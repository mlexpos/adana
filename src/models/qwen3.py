"""
Qwen3 model implementation extending the base GPT model with:
1. Elementwise attention output gating
2. QK-LayerNorm applied to query and key before attention
3. Configurable normalization (LayerNorm or RMSNorm)
4. Z-loss applied to final logits for training stability
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import GPTBase, LayerNorm


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Qwen3MLP(nn.Module):
    """SwiGLU MLP as used in Qwen3"""
    def __init__(self, config):
        super().__init__()

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

        # SwiGLU: gate_proj and up_proj
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = self.dropout(x)
        return x, {}


class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Determine QKV dimension per head
        self.qkv_dim = config.qkv_dim if config.qkv_dim is not None else config.n_embd // config.n_head
        self.total_qkv_dim = self.qkv_dim * config.n_head

        # Elementwise gating support
        self.elementwise_attn_output_gate = config.elementwise_attn_output_gate

        # Query projection: 2x size if using elementwise gating
        if self.elementwise_attn_output_gate:
            self.q_proj = nn.Linear(config.n_embd, 2 * self.total_qkv_dim, bias=config.bias)
        else:
            self.q_proj = nn.Linear(config.n_embd, self.total_qkv_dim, bias=config.bias)

        # Key and value projections
        self.k_proj = nn.Linear(config.n_embd, self.total_qkv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.total_qkv_dim, bias=config.bias)

        # Output projection
        self.o_proj = nn.Linear(self.total_qkv_dim, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # Causal mask for non-flash attention
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.sequence_length, config.sequence_length)
                ).view(1, 1, config.sequence_length, config.sequence_length),
            )

        # QK normalization - configurable between LayerNorm and RMSNorm (optional)
        self.use_qknorm = not getattr(config, 'no_qknorm', False)
        if self.use_qknorm:
            norm_type = getattr(config, 'normalization_layer_type', 'rmsnorm')
            if norm_type == 'rmsnorm':
                self.q_norm = RMSNorm(self.qkv_dim, eps=config.rmsnorm_eps)
                self.k_norm = RMSNorm(self.qkv_dim, eps=config.rmsnorm_eps)
            else:  # layernorm
                self.q_norm = LayerNorm(self.qkv_dim, bias=config.bias)
                self.k_norm = LayerNorm(self.qkv_dim, bias=config.bias)

    def forward(self, x, freqs_cis):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # Project to queries, keys, values
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        # Handle elementwise gating if enabled
        if self.elementwise_attn_output_gate:
            # Split query projection into queries and gate scores
            query_states = query_states.view(B, T, self.n_head, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.qkv_dim, self.qkv_dim],
                dim=-1
            )
            # gate_score shape: (B, T, n_head, qkv_dim)
        else:
            query_states = query_states.view(B, T, self.n_head, self.qkv_dim)
            gate_score = None

        # Reshape keys and values
        key_states = key_states.view(B, T, self.n_head, self.qkv_dim)
        value_states = value_states.view(B, T, self.n_head, self.qkv_dim)

        # Apply QK normalization (if enabled)
        if self.use_qknorm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply RoPE after normalization
        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        # Transpose to (B, n_head, T, qkv_dim) for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Compute attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual implementation of attention
            attn_weights = (query_states @ key_states.transpose(-2, -1)) * (1.0 / math.sqrt(self.qkv_dim))
            attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = attn_weights @ value_states  # (B, n_head, T, qkv_dim)

        # Transpose back to (B, T, n_head, qkv_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Apply elementwise gating if enabled
        if self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        # Reshape and project output
        attn_output = attn_output.view(B, T, self.total_qkv_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class Qwen3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Normalization layers - configurable between LayerNorm and RMSNorm
        norm_type = getattr(config, 'normalization_layer_type', 'rmsnorm')
        if norm_type == 'rmsnorm':
            self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        else:  # layernorm
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention and MLP
        self.attn = Qwen3Attention(config)

        if config.moe:
            from models.moe import MoE
            self.mlp = MoE(config, Qwen3MLP)
        else:
            self.mlp = Qwen3MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.config.residual_stream_scalar * self.attn(self.ln_1(x), freqs_cis)
        x_, logits_and_experts = self.mlp(self.ln_2(x))
        x = x + self.config.residual_stream_scalar * x_
        return x, logits_and_experts


class Qwen3(GPTBase):
    def __init__(self, config):
        # Qwen3 can have configurable weight tying
        super().__init__(config)

        # Initialize RoPE frequencies
        self.head_dim = config.qkv_dim if config.qkv_dim is not None else config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        # Remove unused positional embeddings since we're using RoPE
        del self.transformer.wpe

        # Replace final layer norm with configurable norm
        norm_type = getattr(config, 'normalization_layer_type', 'rmsnorm')
        if norm_type == 'rmsnorm':
            self.transformer.ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        else:  # layernorm
            self.transformer.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Replace blocks with Qwen3 blocks
        self.transformer.h = nn.ModuleList([Qwen3Block(config) for _ in range(config.n_layer)])

        # Re-apply weight initialization since we replaced blocks
        self.apply(self._init_weights)

        # Apply depth scaling for specific init schemes
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
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
        Compute z-loss as described in DiLoCo paper.
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
        For Qwen3, we don't have positional embeddings since we use RoPE.
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
