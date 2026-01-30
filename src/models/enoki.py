"""
Enoki model implementation extending the base GPT model with:
1. QK-LayerNorm applied to query and key before attention
2. Z-loss applied to final logits for training stability
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import GPTBase, CausalSelfAttention, LayerNorm, Block, MLP


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    # Perform manual "complex" multiplication
    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out, k_out


class EnokiAttention(CausalSelfAttention):
    def __init__(self, config):
        # Determine QKV dimension per head
        self.qkv_dim = config.qkv_dim if config.qkv_dim is not None else config.n_embd // config.n_head
        self.total_qkv_dim = self.qkv_dim * config.n_head

        # Initialize parent class but override c_attn and c_proj
        nn.Module.__init__(self)  # Skip CausalSelfAttention.__init__

        # key, query, value projections for all heads, but with custom QKV dimension
        self.c_attn = nn.Linear(config.n_embd, 3 * self.total_qkv_dim, bias=config.bias)
        # output projection from QKV space back to embedding space
        self.c_proj = nn.Linear(self.total_qkv_dim, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

        # Add LayerNorm for Q and K using the custom QKV dimension (optional)
        self.use_qknorm = not getattr(config, 'no_qknorm', False)
        if self.use_qknorm:
            self.q_layernorm = LayerNorm(self.qkv_dim, bias=config.bias)
            self.k_layernorm = LayerNorm(self.qkv_dim, bias=config.bias)

    def forward(self, x, freqs_cis):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch with custom QKV dimension
        q, k, v = self.c_attn(x).split(self.total_qkv_dim, dim=2)

        # reshape to (B, T, nh, qkv_dim)
        k = k.view(B, T, self.n_head, self.qkv_dim)
        q = q.view(B, T, self.n_head, self.qkv_dim)
        v = v.view(B, T, self.n_head, self.qkv_dim)

        # Apply QK-LayerNorm (if enabled)
        if self.use_qknorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        # Apply RoPE after layer norm operations
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # transpose to (B, nh, T, qkv_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.total_qkv_dim)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class EnokiMLP(MLP):
    def __init__(self, config):
        # Use custom MLP hidden dimension if provided
        if config.mlp_hidden_dim is not None:
            hidden_dim = config.mlp_hidden_dim
        else:
            # Standard calculation: 4x expansion with SwiGLU adjustment
            hidden_dim = config.n_embd * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )

        # Initialize MLP layers with custom hidden dimension
        nn.Module.__init__(self)  # Skip MLP.__init__

        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, {}


class EnokiBlock(Block):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Replace attention with Enoki attention
        self.attn = EnokiAttention(config)
        # Replace MLP with Enoki MLP (supports custom hidden dim)
        if config.moe:
            # Keep MoE support if needed
            from models.moe import MoE
            self.mlp = MoE(config, EnokiMLP)
        else:
            self.mlp = EnokiMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.config.residual_stream_scalar * self.attn(self.ln_1(x), freqs_cis)
        x_, logits_and_experts = self.mlp(self.ln_2(x))
        x = x + self.config.residual_stream_scalar * x_
        return x, logits_and_experts


class Enoki(GPTBase):
    def __init__(self, config):
        # Override weight_tying to False for Enoki
        config.weight_tying = False
        super().__init__(config)

        # Initialize RoPE frequencies
        self.head_dim = self.config.qkv_dim if self.config.qkv_dim is not None else config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        # Remove unused positional embeddings since we're using RoPE
        del self.transformer.wpe

        # Replace blocks with Enoki blocks
        self.transformer.h = nn.ModuleList([EnokiBlock(config) for _ in range(config.n_layer)])

        # Re-apply weight initialization since we replaced blocks
        self.apply(self._init_weights)

        # Apply depth scaling for specific init schemes
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                if config.init_scheme == "KarpathyGPT2":
                    # KarpathyGPT2: std = init_std / sqrt(2 * n_layer)
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=self.config.init_std / math.sqrt(2 * config.n_layer),
                    )
                elif config.init_scheme == "ScaledGPT":
                    # ScaledGPT: std = 1 / sqrt(2 * fan_in * n_layer)
                    fan_in = p.size(1)
                    std = 1.0 / math.sqrt(2 * fan_in * config.n_layer)
                    torch.nn.init.normal_(p, mean=0.0, std=std)

    def compute_z_loss(self, logits):
        """
        Compute z-loss as described in Enoki paper.
        Z-loss = log^2(Z) where Z = sum(exp(logits))
        This encourages log(Z) to remain close to zero for stability.
        """
        # Compute log(Z) = log(sum(exp(logits))) using logsumexp for numerical stability
        log_z = torch.logsumexp(logits, dim=-1, keepdim=False)  # (batch_size, seq_len)

        # Z-loss = (log(Z))^2
        z_loss = log_z.pow(2)

        # Return mean z-loss
        return z_loss.mean()

    def get_num_params(self, non_embedding=True, exclude_embeddings=False):
        """
        Return the number of parameters in the model.
        For Enoki, we don't have positional embeddings since we use RoPE.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wte.weight.numel()  # Token embeddings
            n_params -= self.lm_head.weight.numel()  # LM head
        # Note: No positional embeddings to subtract since we use RoPE
        return n_params

    def forward(self, idx, targets=None, get_logits=False, moe=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        # shape (t,)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        # router logits is a list for each layer's routing, each of shape (b * seq_len, n_experts)
        router_logits = []
        # experts is a list for each layer's selected experts, shape (b * seq_len, topk)
        experts = []

        # forward pass through all the transformer blocks
        for block in self.transformer.h:
            x, logits_and_experts = block(x, freqs_cis)
            if len(logits_and_experts) > 0:
                router_logits.append(logits_and_experts["router_logits"])
                experts.append(logits_and_experts["selected_experts"])
        x = self.transformer.ln_f(x)

        # aux_losses is a dict with keys for different auxiliary losses
        aux_losses = {}

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

            # Add z-loss with coefficient 1e-4 as mentioned in the paper
            z_loss = self.compute_z_loss(logits.view(-1, logits.size(-1)))
            aux_losses["z_loss"] = z_loss
            loss += self.config.z_loss_coeff * z_loss

            # Add Hoyer loss for embedding sparsity
            hoyer_loss = self.compute_hoyer_loss(tok_emb)
            aux_losses["hoyer_loss"] = hoyer_loss
            loss += self.config.hoyer_loss_coeff * hoyer_loss

            if moe and self.config.moe_routing == "standard_gating":
                # calculate the router losses per layer
                for logit, expert_choice in zip(router_logits, experts):
                    router_losses = self.get_router_losses(
                        logit, expert_choice, eval=not self.training
                    )
                    for k, v in router_losses.items():
                        aux_losses[k] = aux_losses.get(k, 0.0) + v
                        if self.training:
                            loss += (
                                v
                                * getattr(self.config, k + "_factor")
                                / self.config.n_layer
                            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None
        router_logits = (
            torch.stack(router_logits, dim=0) if len(router_logits) > 0 else None
        )

        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "router_logits": router_logits,
        }