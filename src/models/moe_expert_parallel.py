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

    def _group_tokens_by_rank(self, tokens, selected_experts, weights):
        """
        Group tokens by which rank (GPU) owns their selected experts.

        Args:
            tokens: [num_tokens, hidden_dim]
            selected_experts: [num_tokens, top_k]
            weights: [num_tokens, top_k]

        Returns:
            tokens_per_rank: List of [num_tokens_for_rank, hidden_dim] tensors
            expert_ids_per_rank: List of expert IDs for each rank
            token_indices_per_rank: List of original token indices
            weights_per_rank: List of weights for each rank
        """
        num_tokens = tokens.shape[0]
        device = tokens.device

        # Storage for each rank
        tokens_per_rank = []
        expert_ids_per_rank = []
        token_indices_per_rank = []
        weights_per_rank = []

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

            if len(token_idx) > 0:
                # Gather tokens, expert IDs, and weights
                rank_tokens = tokens[token_idx]
                rank_expert_ids = selected_experts[token_idx, expert_slot]
                rank_weights = weights[token_idx, expert_slot]

                tokens_per_rank.append(rank_tokens)
                expert_ids_per_rank.append(rank_expert_ids)
                token_indices_per_rank.append(token_idx)
                weights_per_rank.append(rank_weights)
            else:
                # No tokens for this rank - send empty tensors
                tokens_per_rank.append(torch.empty(0, tokens.shape[1], device=device))
                expert_ids_per_rank.append(torch.empty(0, dtype=torch.long, device=device))
                token_indices_per_rank.append(torch.empty(0, dtype=torch.long, device=device))
                weights_per_rank.append(torch.empty(0, dtype=torch.float32, device=device))

        return tokens_per_rank, expert_ids_per_rank, token_indices_per_rank, weights_per_rank

    def _all_to_all_tokens(self, tokens_per_rank):
        """
        All-to-all communication to redistribute tokens.

        Args:
            tokens_per_rank: List of [num_tokens_for_rank, hidden_dim] tensors

        Returns:
            received_tokens_list: List of tensors received from each rank
            received_sizes: Number of tokens received from each rank
        """
        if self.world_size == 1:
            # Single GPU - no communication needed
            return tokens_per_rank, [t.shape[0] for t in tokens_per_rank]

        device = tokens_per_rank[0].device
        hidden_dim = tokens_per_rank[0].shape[1]

        # Get sizes to send to each rank
        send_sizes = [t.shape[0] for t in tokens_per_rank]

        # Exchange sizes (so each rank knows how much to receive)
        send_sizes_tensor = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_tensor = torch.zeros(self.world_size, dtype=torch.long, device=device)

        # All-gather sizes
        dist.all_to_all_single(
            recv_sizes_tensor,
            send_sizes_tensor
        )
        recv_sizes = recv_sizes_tensor.tolist()

        # Prepare send tensor (concatenate all tokens)
        total_send = sum(send_sizes)
        send_tensor = torch.cat(tokens_per_rank, dim=0) if total_send > 0 else \
                     torch.empty(0, hidden_dim, device=device)

        # Prepare receive tensor
        total_recv = sum(recv_sizes)
        recv_tensor = torch.empty(total_recv, hidden_dim, device=device)

        # All-to-all communication
        # Split send_tensor and recv_tensor according to sizes
        send_splits = list(send_tensor.split(send_sizes, dim=0))
        recv_splits = list(recv_tensor.split(recv_sizes, dim=0))

        # Pad to same size for all-to-all
        max_size = max(max(send_sizes), max(recv_sizes)) if send_sizes and recv_sizes else 0

        if max_size > 0:
            # Pad and stack
            send_padded = torch.zeros(self.world_size, max_size, hidden_dim, device=device)
            recv_padded = torch.zeros(self.world_size, max_size, hidden_dim, device=device)

            for i, tensor in enumerate(send_splits):
                if tensor.shape[0] > 0:
                    send_padded[i, :tensor.shape[0]] = tensor

            # Perform all-to-all
            all_to_all_single(
                recv_padded.view(-1, hidden_dim),
                send_padded.view(-1, hidden_dim)
            )

            # Extract received tensors
            received_tokens_list = []
            for i, size in enumerate(recv_sizes):
                if size > 0:
                    received_tokens_list.append(recv_padded[i, :size])
                else:
                    received_tokens_list.append(torch.empty(0, hidden_dim, device=device))
        else:
            received_tokens_list = [torch.empty(0, hidden_dim, device=device)
                                   for _ in range(self.world_size)]

        return received_tokens_list, recv_sizes

    def _process_local_experts(self, received_tokens_list, received_expert_ids_list,
                               received_weights_list, received_token_indices_list):
        """
        Process tokens through local experts.

        Returns:
            results_per_source_rank: List of result tensors to send back to each rank
        """
        device = self.local_experts[0].gate_proj.weight.device
        hidden_dim = self.config.n_embd

        results_per_source_rank = []

        # Process tokens from each source rank
        for source_rank in range(self.world_size):
            tokens_from_source = received_tokens_list[source_rank]
            expert_ids_from_source = received_expert_ids_list[source_rank]
            weights_from_source = received_weights_list[source_rank]

            if tokens_from_source.shape[0] == 0:
                # No tokens from this rank
                results_per_source_rank.append(torch.empty(0, hidden_dim, device=device))
                continue

            # Results for tokens from this source rank
            results = torch.zeros_like(tokens_from_source)

            # Process each local expert
            for local_expert_idx, expert in enumerate(self.local_experts):
                global_expert_id = self.expert_start_idx + local_expert_idx

                # Find tokens for this expert
                expert_mask = (expert_ids_from_source == global_expert_id)

                if expert_mask.any():
                    expert_tokens = tokens_from_source[expert_mask]
                    expert_weights = weights_from_source[expert_mask]

                    # Process through expert
                    expert_output, _ = expert(expert_tokens)

                    # Apply weights and accumulate
                    results[expert_mask] = expert_weights.unsqueeze(-1) * expert_output

            results_per_source_rank.append(results)

        return results_per_source_rank

    def forward(self, inputs):
        """
        Forward pass with expert parallelism.

        Args:
            inputs: [batch_size, seq_len, hidden_dim]

        Returns:
            outputs: [batch_size, seq_len, hidden_dim]
            aux_dict: Dictionary with router_logits and selected_experts
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        inputs_flat = inputs.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        num_tokens = inputs_flat.shape[0]

        # Step 1: Compute routing (local to each GPU)
        weights, selected_experts, router_logits = self._compute_routing(inputs_flat)

        # Step 2: Group tokens by target rank
        (tokens_per_rank, expert_ids_per_rank,
         token_indices_per_rank, weights_per_rank) = self._group_tokens_by_rank(
            inputs_flat, selected_experts, weights
        )

        # Step 3: All-to-all to send tokens to expert owners
        received_tokens_list, recv_sizes = self._all_to_all_tokens(tokens_per_rank)

        # Also send expert IDs and weights
        received_expert_ids_list, _ = self._all_to_all_tokens(
            [ids.unsqueeze(-1).float() for ids in expert_ids_per_rank]
        )
        received_expert_ids_list = [ids.squeeze(-1).long() for ids in received_expert_ids_list]

        received_weights_list, _ = self._all_to_all_tokens(
            [w.unsqueeze(-1) for w in weights_per_rank]
        )
        received_weights_list = [w.squeeze(-1) for w in received_weights_list]

        received_token_indices_list, _ = self._all_to_all_tokens(
            [idx.unsqueeze(-1).float() for idx in token_indices_per_rank]
        )
        received_token_indices_list = [idx.squeeze(-1).long() for idx in received_token_indices_list]

        # Step 4: Process local experts
        results_per_source_rank = self._process_local_experts(
            received_tokens_list,
            received_expert_ids_list,
            received_weights_list,
            received_token_indices_list
        )

        # Step 5: All-to-all to send results back
        final_results_list, _ = self._all_to_all_tokens(results_per_source_rank)

        # Step 6: Reconstruct output in original order
        outputs_flat = torch.zeros_like(inputs_flat)

        for source_rank in range(self.world_size):
            results_from_rank = final_results_list[source_rank]
            token_indices = token_indices_per_rank[source_rank]

            if results_from_rank.shape[0] > 0:
                outputs_flat.index_add_(0, token_indices, results_from_rank)

        outputs = outputs_flat.view(batch_size, seq_len, hidden_dim)

        return outputs, {
            "router_logits": router_logits,
            "selected_experts": selected_experts
        }
