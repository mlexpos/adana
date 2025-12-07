"""
Qwen3Next model implementation extending Qwen3 with:
1. Gated shared expert with learned sigmoid scaling
2. Conditional sparsity (layer-wise MoE/dense alternation)
3. Support for extreme sparsity configurations

Based on Qwen3-Next architecture from Hugging Face transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.qwen3 import Qwen3, Qwen3MLP, Qwen3Attention, RMSNorm
from models.base import LayerNorm
from models.moe import MoE


class Qwen3NextMoE(nn.Module):
    """
    Qwen3Next MoE block combining:
    - Routed experts (sparse, top-k selection)
    - Gated shared expert (dense, learned per-token gating)

    This follows the Qwen3-Next architecture where the shared expert's
    contribution is controlled by a learned sigmoid gate.
    """

    def __init__(self, config, mlp_class):
        super().__init__()
        self.config = config

        # Routed experts using existing MoE implementation
        # Note: If use_gated_shared_expert is True, we don't use the
        # DeepSeek-style ungated shared experts from the base MoE
        moe_config = config
        if config.use_gated_shared_expert:
            # Override to disable ungated shared experts
            moe_config = type('obj', (object,), {**vars(config)})()
            moe_config.moe_num_shared_experts = 0

        # Use expert parallelism if distributed and enabled
        use_expert_parallel = (
            getattr(config, 'expert_parallel', True) and
            dist.is_initialized() and
            dist.get_world_size() > 1
        )

        if use_expert_parallel:
            from models.moe_expert_parallel import ExpertParallelMoE
            self.routed_moe = ExpertParallelMoE(moe_config, mlp_class)
        else:
            self.routed_moe = MoE(moe_config, mlp_class)

        # Gated shared expert (Qwen3Next style)
        if config.use_gated_shared_expert:
            # Create shared expert with potentially different intermediate size
            shared_config = type('obj', (object,), {**vars(config)})()
            if config.shared_expert_intermediate_size is not None:
                shared_config.mlp_hidden_dim = config.shared_expert_intermediate_size

            self.shared_expert = mlp_class(shared_config)
            # Learned gate: projects to single scalar per token
            self.shared_expert_gate = nn.Linear(config.n_embd, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(self, x):
        """
        Forward pass combining routed and gated shared experts.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)

        Returns:
            Tuple of (output, logits_dict) where:
            - output: Combined expert outputs
            - logits_dict: Router logits and selected experts for auxiliary losses
        """
        # Process through routed experts
        routed_output, logits_dict = self.routed_moe(x)

        # Add gated shared expert if enabled
        if self.shared_expert is not None:
            # Flatten for processing
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.view(-1, hidden_dim)

            # Process through shared expert
            shared_output, _ = self.shared_expert(x_flat)

            # Apply learned sigmoid gate
            gate = torch.sigmoid(self.shared_expert_gate(x_flat))
            shared_output = gate * shared_output

            # Reshape and add to routed output
            shared_output = shared_output.view(batch_size, seq_len, hidden_dim)
            final_output = routed_output + shared_output
        else:
            final_output = routed_output

        return final_output, logits_dict


class Qwen3NextBlock(nn.Module):
    """
    Qwen3Next transformer block with conditional sparsity.

    Supports:
    - Layer-wise MoE/dense alternation via decoder_sparse_step
    - Explicit dense layer specification via mlp_only_layers
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Normalization layers - configurable between LayerNorm and RMSNorm
        norm_type = getattr(config, 'normalization_layer_type', 'rmsnorm')
        if norm_type == 'rmsnorm':
            self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        else:  # layernorm
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention (same as Qwen3)
        self.attn = Qwen3Attention(config)

        # MLP or MoE based on conditional sparsity settings
        self.is_moe_layer = self._should_use_moe(config, layer_idx)

        if self.is_moe_layer:
            self.mlp = Qwen3NextMoE(config, Qwen3MLP)
        else:
            self.mlp = Qwen3MLP(config)

    def _should_use_moe(self, config, layer_idx):
        """
        Determine if this layer should use MoE based on configuration.

        Priority:
        1. If layer_idx in mlp_only_layers -> False (force dense)
        2. If not config.moe -> False (MoE disabled globally)
        3. If layer_idx % decoder_sparse_step == 0 -> True (conditional sparsity)
        4. Otherwise -> False (dense)
        """
        # Parse mlp_only_layers if provided
        mlp_only_layers = []
        if hasattr(config, 'mlp_only_layers') and config.mlp_only_layers is not None:
            if isinstance(config.mlp_only_layers, str):
                mlp_only_layers = [int(x.strip()) for x in config.mlp_only_layers.split(',') if x.strip()]
            elif isinstance(config.mlp_only_layers, (list, tuple)):
                mlp_only_layers = list(config.mlp_only_layers)

        # Check if this layer is forced to be dense
        if layer_idx in mlp_only_layers:
            return False

        # Check if MoE is enabled globally
        if not config.moe:
            return False

        # Apply decoder_sparse_step pattern
        decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
        return layer_idx % decoder_sparse_step == 0

    def forward(self, x, freqs_cis):
        """
        Forward pass through the block.

        Args:
            x: Input tensor
            freqs_cis: Rotary position embeddings

        Returns:
            Tuple of (output, logits_and_experts)
        """
        # Attention
        x = x + self.config.residual_stream_scalar * self.attn(self.ln_1(x), freqs_cis)

        # MLP or MoE
        x_, logits_and_experts = self.mlp(self.ln_2(x))
        x = x + self.config.residual_stream_scalar * x_

        return x, logits_and_experts


class Qwen3Next(Qwen3):
    """
    Qwen3Next model with gated shared expert and conditional sparsity.

    Inherits from Qwen3 and replaces blocks with Qwen3NextBlock to support:
    - Gated shared expert with learned sigmoid scaling
    - Layer-wise MoE/dense alternation
    - Extreme sparsity configurations
    """

    def __init__(self, config):
        # Ensure MoE is enabled by default for Qwen3Next
        # User can still explicitly disable with --no-moe if needed
        if not hasattr(config, 'moe'):
            config.moe = True

        # Initialize base Qwen3
        super().__init__(config)

        # Replace blocks with Qwen3NextBlock (with layer indices)
        self.transformer.h = nn.ModuleList([
            Qwen3NextBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])

        # Re-apply weight initialization since we replaced blocks
        self.apply(self._init_weights)

        # Apply depth scaling for specific init schemes (same as Qwen3)
        import math
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                if config.init_scheme == "KarpathyGPT2":
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=self.config.init_std / math.sqrt(2 * config.n_layer),
                    )
                elif config.init_scheme == "ScaledGPT":
                    fan_in = p.size(1)
                    std = 1.0 / math.sqrt(2 * fan_in * config.n_layer)
                    torch.nn.init.normal_(p, mean=0.0, std=std)

    def print_sparsity_pattern(self):
        """
        Print the sparsity pattern (which layers use MoE vs dense MLP).
        Useful for debugging and understanding the model architecture.
        """
        print("\n" + "="*60)
        print("Qwen3Next Sparsity Pattern")
        print("="*60)
        for i, block in enumerate(self.transformer.h):
            layer_type = "MoE" if block.is_moe_layer else "Dense MLP"
            print(f"Layer {i:2d}: {layer_type}")
        print("="*60 + "\n")

        # Count statistics
        n_moe = sum(1 for block in self.transformer.h if block.is_moe_layer)
        n_dense = len(self.transformer.h) - n_moe
        print(f"Total layers: {len(self.transformer.h)}")
        print(f"MoE layers: {n_moe} ({100*n_moe/len(self.transformer.h):.1f}%)")
        print(f"Dense layers: {n_dense} ({100*n_dense/len(self.transformer.h):.1f}%)")

        if self.config.use_gated_shared_expert:
            print(f"Shared expert: Gated (learned sigmoid)")
        elif self.config.moe_num_shared_experts > 0:
            print(f"Shared experts: {self.config.moe_num_shared_experts} (ungated)")
        else:
            print(f"Shared expert: None")
        print("="*60 + "\n")
