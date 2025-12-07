"""
Expert-Parallel Mixture of Experts implementation.

This module implements expert parallelism for MoE layers, distributing experts
across multiple GPUs using all-to-all communication for token routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def all_to_all_single(output_tensor, input_tensor, group=None):
    """
    Wrapper for torch.distributed.all_to_all_single with proper error handling.

    Args:
        output_tensor: Output tensor (will be filled)
        input_tensor: Input tensor to send
        group: Process group for communication
    """
    if not dist.is_initialized():
        # Single GPU mode - no communication needed
        output_tensor.copy_(input_tensor)
        return

    dist.all_to_all_single(output_tensor, input_tensor, group=group)


class ExpertParallelMoE(nn.Module):
    """
    Expert-parallel MoE layer using all-to-all communication.

    This implementation distributes experts across GPUs and uses all-to-all
    communication to route tokens to the GPUs that own their selected experts.

    Args:
        config: Model configuration
        expert_class: Class to instantiate for each expert
    """

    def __init__(self, config, expert_class):
        super().__init__()
        self.config = config

        # Distributed setup
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Expert configuration
        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = config.moe_num_experts_per_tok
        self.softmax_order = config.moe_softmax_order
        self.capacity_factor = config.moe_expert_capacity_factor

        # Each GPU owns a shard of experts
        assert self.num_experts % self.world_size == 0, \
            f"num_experts ({self.num_experts}) must be divisible by world_size ({self.world_size})"

        self.experts_per_rank = self.num_experts // self.world_size
        self.expert_start_idx = self.rank * self.experts_per_rank
        self.expert_end_idx = (self.rank + 1) * self.experts_per_rank

        # Create only local experts (owned by this GPU)
        self.local_experts = nn.ModuleList([
            expert_class(config) for _ in range(self.experts_per_rank)
        ])

        # Router is replicated on all GPUs
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # Capacity calculation for 3D tensor buffers
        # Will be dynamically set based on batch size in forward pass
        self.max_capacity = None

    def _compute_routing(self, inputs):
        """
        Compute routing decisions (which experts to use).

        Args:
            inputs: [num_tokens, hidden_dim]

        Returns:
            weights: [num_tokens, top_k]
            selected_experts: [num_tokens, top_k]
            router_logits: [num_tokens, num_experts]
        """
        num_tokens, hidden_dim = inputs.shape

        # Compute router logits
        router_logits = self.router(inputs)  # [num_tokens, num_experts]

        # Select top-k experts
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, selected_experts = torch.topk(all_probs, self.num_experts_per_tok, dim=-1)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        return weights, selected_experts, router_logits

    def _group_tokens_by_rank(self, tokens, selected_experts, weights, max_capacity):
        """
        Group tokens by which rank (GPU) owns their selected experts.
        Returns 3D padded tensors for torch.compile compatibility.

        Args:
            tokens: [num_tokens, hidden_dim]
            selected_experts: [num_tokens, top_k]
            weights: [num_tokens, top_k]
            max_capacity: Maximum tokens per rank (buffer size)

        Returns:
            tokens_3d: [world_size, max_capacity, hidden_dim]
            expert_ids_3d: [world_size, max_capacity]
            token_indices_3d: [world_size, max_capacity]
            weights_3d: [world_size, max_capacity]
            valid_mask: [world_size, max_capacity] (bool, True for real tokens)
            overflow_count: Number of tokens dropped due to capacity overflow
        """
        num_tokens, hidden_dim = tokens.shape
        device = tokens.device

        # Pre-allocate 3D tensors
        tokens_3d = torch.zeros(self.world_size, max_capacity, hidden_dim, device=device, dtype=tokens.dtype)
        expert_ids_3d = torch.zeros(self.world_size, max_capacity, device=device, dtype=torch.long)
        token_indices_3d = torch.zeros(self.world_size, max_capacity, device=device, dtype=torch.long)
        weights_3d = torch.zeros(self.world_size, max_capacity, device=device, dtype=torch.float32)
        valid_mask = torch.zeros(self.world_size, max_capacity, device=device, dtype=torch.bool)

        overflow_count = 0

        # For each rank, find which tokens need its experts
        for target_rank in range(self.world_size):
            rank_expert_start = target_rank * self.experts_per_rank
            rank_expert_end = (target_rank + 1) * self.experts_per_rank

            # Create mask for tokens that selected this rank's experts
            # Shape: [num_tokens, top_k]
            expert_in_rank = (selected_experts >= rank_expert_start) & \
                           (selected_experts < rank_expert_end)

            # Get token-expert pairs for this rank
            token_idx, expert_slot = torch.where(expert_in_rank)

            num_tokens_for_rank = len(token_idx)

            if num_tokens_for_rank > 0:
                # Handle capacity overflow
                if num_tokens_for_rank > max_capacity:
                    overflow_count += num_tokens_for_rank - max_capacity
                    # Truncate to capacity
                    token_idx = token_idx[:max_capacity]
                    expert_slot = expert_slot[:max_capacity]
                    num_tokens_for_rank = max_capacity

                # Fill 3D tensors
                tokens_3d[target_rank, :num_tokens_for_rank] = tokens[token_idx]
                expert_ids_3d[target_rank, :num_tokens_for_rank] = selected_experts[token_idx, expert_slot]
                token_indices_3d[target_rank, :num_tokens_for_rank] = token_idx
                weights_3d[target_rank, :num_tokens_for_rank] = weights[token_idx, expert_slot]
                valid_mask[target_rank, :num_tokens_for_rank] = True

        return tokens_3d, expert_ids_3d, token_indices_3d, weights_3d, valid_mask, overflow_count

    def _all_to_all_tokens_3d(self, send_3d, send_mask):
        """
        All-to-all communication to redistribute tokens using 3D tensors.
        Torch.compile compatible - no dynamic sizes or .tolist() calls.

        Args:
            send_3d: [world_size, max_capacity, hidden_dim] - tokens to send to each rank
            send_mask: [world_size, max_capacity] - validity mask for send_3d

        Returns:
            recv_3d: [world_size, max_capacity, hidden_dim] - tokens received from each rank
            recv_mask: [world_size, max_capacity] - validity mask for recv_3d
        """
        if self.world_size == 1:
            # Single GPU - no communication needed
            return send_3d, send_mask

        device = send_3d.device
        world_size, max_capacity, hidden_dim = send_3d.shape

        # Prepare output tensors
        recv_3d = torch.zeros_like(send_3d)
        recv_mask = torch.zeros_like(send_mask)

        # Perform all-to-all on flattened tensors
        # Reshape [world_size, max_capacity, hidden_dim] -> [world_size * max_capacity, hidden_dim]
        all_to_all_single(
            recv_3d.view(-1, hidden_dim),
            send_3d.view(-1, hidden_dim)
        )

        # Also exchange masks to know which tokens are valid
        # Reshape [world_size, max_capacity] -> [world_size * max_capacity]
        # Convert bool to float for communication, then back to bool
        all_to_all_single(
            recv_mask.view(-1, 1).float(),
            send_mask.view(-1, 1).float()
        )
        recv_mask = recv_mask.view(world_size, max_capacity).bool()

        return recv_3d, recv_mask

    def _process_local_experts_3d(self, received_tokens_3d, received_expert_ids_3d,
                                  received_weights_3d, received_mask):
        """
        Process tokens through local experts using 3D tensors.

        Args:
            received_tokens_3d: [world_size, max_capacity, hidden_dim]
            received_expert_ids_3d: [world_size, max_capacity]
            received_weights_3d: [world_size, max_capacity]
            received_mask: [world_size, max_capacity] - validity mask

        Returns:
            results_3d: [world_size, max_capacity, hidden_dim] - processed results
        """
        device = received_tokens_3d.device
        world_size, max_capacity, hidden_dim = received_tokens_3d.shape

        # Pre-allocate results tensor
        results_3d = torch.zeros_like(received_tokens_3d)

        # Process tokens from each source rank
        for source_rank in range(world_size):
            tokens_from_source = received_tokens_3d[source_rank]  # [max_capacity, hidden_dim]
            expert_ids_from_source = received_expert_ids_3d[source_rank]  # [max_capacity]
            weights_from_source = received_weights_3d[source_rank]  # [max_capacity]
            valid_from_source = received_mask[source_rank]  # [max_capacity]

            # Skip if no valid tokens
            if not valid_from_source.any():
                continue

            # Process each local expert
            for local_expert_idx, expert in enumerate(self.local_experts):
                global_expert_id = self.expert_start_idx + local_expert_idx

                # Find tokens for this expert (only among valid tokens)
                expert_mask = (expert_ids_from_source == global_expert_id) & valid_from_source

                if expert_mask.any():
                    expert_tokens = tokens_from_source[expert_mask]
                    expert_weights = weights_from_source[expert_mask]

                    # Process through expert
                    expert_output, _ = expert(expert_tokens)

                    # Apply weights and accumulate
                    # Use masked scatter to put results back
                    results_3d[source_rank][expert_mask] = expert_weights.unsqueeze(-1) * expert_output

        return results_3d

    def forward(self, inputs):
        """
        Forward pass with expert parallelism using 3D tensors.

        Args:
            inputs: [batch_size, seq_len, hidden_dim]

        Returns:
            outputs: [batch_size, seq_len, hidden_dim]
            aux_dict: Dictionary with router_logits, selected_experts, and overflow_count
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        inputs_flat = inputs.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        num_tokens = inputs_flat.shape[0]

        # Calculate max capacity for 3D buffers
        # capacity_factor × avg_tokens_per_rank with some headroom
        avg_tokens_per_rank = num_tokens * self.num_experts_per_tok / self.world_size
        max_capacity = int(self.capacity_factor * avg_tokens_per_rank) + 1

        # Step 1: Compute routing (local to each GPU)
        weights, selected_experts, router_logits = self._compute_routing(inputs_flat)

        # Step 2: Group tokens by target rank (returns 3D tensors)
        (send_tokens_3d, send_expert_ids_3d, send_token_indices_3d,
         send_weights_3d, send_mask, overflow_count) = self._group_tokens_by_rank(
            inputs_flat, selected_experts, weights, max_capacity
        )

        # Step 3: All-to-all to send tokens to expert owners
        recv_tokens_3d, recv_mask = self._all_to_all_tokens_3d(send_tokens_3d, send_mask)

        # Also send expert IDs and weights
        recv_expert_ids_3d, _ = self._all_to_all_tokens_3d(
            send_expert_ids_3d.unsqueeze(-1).float(),
            send_mask
        )
        recv_expert_ids_3d = recv_expert_ids_3d.squeeze(-1).long()

        recv_weights_3d, _ = self._all_to_all_tokens_3d(
            send_weights_3d.unsqueeze(-1),
            send_mask
        )
        recv_weights_3d = recv_weights_3d.squeeze(-1)

        # Step 4: Process local experts (3D version)
        results_3d = self._process_local_experts_3d(
            recv_tokens_3d,
            recv_expert_ids_3d,
            recv_weights_3d,
            recv_mask
        )

        # Step 5: All-to-all to send results back
        final_results_3d, final_mask = self._all_to_all_tokens_3d(results_3d, recv_mask)

        # Step 6: Reconstruct output in original order
        outputs_flat = torch.zeros_like(inputs_flat)

        for source_rank in range(self.world_size):
            results_from_rank = final_results_3d[source_rank]
            token_indices = send_token_indices_3d[source_rank]
            valid = send_mask[source_rank]

            # Only accumulate valid results
            if valid.any():
                # Use masked indexing
                valid_indices = token_indices[valid]
                valid_results = results_from_rank[valid]
                outputs_flat.index_add_(0, valid_indices, valid_results)

        outputs = outputs_flat.view(batch_size, seq_len, hidden_dim)

        return outputs, {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "overflow_count": overflow_count,
        }
