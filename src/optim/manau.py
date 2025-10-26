"""
Manau - Hybrid optimizer combining Distributed Muon with DANA-STAR-MK4
Based on:
- DistributedMuon: https://github.com/KellerJordan/modded-nanogpt
- DANA-STAR-MK4: Custom adaptive optimizer

For 2D parameters (weights), uses Muon with Newton-Schulz orthogonalization.
For non-2D parameters (biases, norms, embeddings), uses DANA-STAR-MK4.
Optional dana_momentum flag replaces fixed momentum with DANA-STAR-MK4 style EMA.
"""

import math
import torch
import torch.distributed as dist
from typing import Dict, Tuple

# Import Newton-Schulz function from muon
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def normalize_range(range: Tuple[int, int], start):
    return (range[0] - start, range[1] - start)


class MuonDistMeta:
    """Metadata for distributed Muon parameters"""

    buffer_idx: int = 0
    bucket_idx: int = 0
    shape: torch.Size = None
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    local_range: Tuple[int, int] = None

    def __init__(
        self,
        buffer_idx: int,
        bucket_idx: int,
        shape: torch.Size,
        global_range: Tuple[int, int],
        tp_split_dim: int,
    ):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (
            (start, end)
            if start < end
            else (local_buffer_range[0], local_buffer_range[0])
        )


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    """Adjust learning rate based on parameter dimensions"""
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


class Manau(torch.optim.Optimizer):
    """
    Manau - Hybrid optimizer combining Muon and DANA-STAR-MK4

    For 2D weight matrices (attention, MLP), uses Muon with Newton-Schulz orthogonalization.
    For other parameters (1D biases/norms, embeddings, heads), uses DANA-STAR-MK4 adaptive optimizer.

    IMPORTANT: param_groups must include 'param_names' key mapping to parameter names.
    This is required to correctly identify embeddings (wte, wpe) and heads (lm_head).

    Arguments:
        param_groups: The parameters to be optimized. Each group dict must include:
                     - 'params': list of parameter tensors
                     - 'param_names': list of parameter names (same order as params)
        lr: The learning rate.
        momentum: The momentum used by Muon (ignored if dana_momentum=True).
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match.
        nesterov: Whether to use Nesterov-style momentum in Muon (ignored if dana_momentum=True).
        ns_steps: The number of Newton-Schulz iterations to run.
        weight_decay: Weight decay coefficient.
        dana_momentum: If True, replace fixed momentum with DANA-STAR-MK4 style EMA.
        delta: Delta parameter for DANA-STAR-MK4 EMA (used when dana_momentum=True).
        kappa: Kappa parameter for DANA-STAR-MK4.
        mk4A: mk4A parameter for DANA-STAR-MK4.
        mk4B: mk4B parameter for DANA-STAR-MK4.
        clipsnr: ClipSNR parameter for DANA-STAR-MK4.
        epsilon: Epsilon for DANA-STAR-MK4.
        weight_time: Whether to use weighted time in DANA-STAR-MK4.
        wd_decaying: Whether to use decaying weight decay in DANA-STAR-MK4.
        wd_ts: Weight decay time scale for DANA-STAR-MK4.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        dana_momentum=False,
        delta=8.0,
        kappa=1.0,
        mk4A=0.0,
        mk4B=0.0,
        clipsnr=1.0,
        epsilon=1e-8,
        weight_time=False,
        wd_decaying=False,
        wd_ts=1.0,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            dana_momentum=dana_momentum,
            delta=delta,
            kappa=kappa,
            mk4A=mk4A,
            mk4B=mk4B,
            clipsnr=clipsnr,
            epsilon=epsilon,
            weight_time=weight_time,
            wd_decaying=wd_decaying,
            wd_ts=wd_ts,
            weighted_step_count=0,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        self.lr = lr  # Store for weight_time calculation

        # Sort parameters into those for which we will use Muon, and those for which we will use DANA-STAR-MK4
        # Parameter names MUST be provided in the group for correct embedding/head detection

        # Tracking for logging
        muon_params = []
        dana_params = []
        muon_total_params = 0
        dana_total_params = 0

        for group in self.param_groups:
            param_names = group.get("param_names", None)
            if param_names is None:
                raise ValueError(
                    "Manau optimizer requires 'param_names' to be provided in param_groups. "
                    "This is needed to correctly identify embeddings and head layers."
                )

            for idx, p in enumerate(group["params"]):
                # Determine if this is an embedding or head layer by name
                is_embedding_or_head = False
                if idx < len(param_names):
                    pname = param_names[idx]
                    # Check if parameter name contains embedding or head identifiers
                    if any(identifier in pname for identifier in ['wte.weight', 'wpe.weight', 'lm_head.weight']):
                        is_embedding_or_head = True

                # Use Muon for 2D parameters that are not embeddings/heads
                # Use DANA-STAR-MK4 for 1D params (biases, norms) or embeddings/heads
                if not is_embedding_or_head and p.ndim >= 2:
                    self.state[p]["use_muon"] = True
                    muon_params.append((pname, p.shape, p.numel()))
                    muon_total_params += p.numel()
                else:
                    self.state[p]["use_muon"] = False
                    dana_params.append((pname, p.shape, p.numel()))
                    dana_total_params += p.numel()

        # Print summary
        print("\n" + "="*80)
        print("Manau Optimizer Parameter Assignment")
        print("="*80)
        print(f"\nMuon (Newton-Schulz) parameters: {len(muon_params)}")
        print(f"Total Muon params: {muon_total_params:,} ({muon_total_params/1e6:.2f}M)")
        print("-" * 80)
        for pname, shape, numel in muon_params:
            print(f"  [Muon] {pname:50s} {str(shape):20s} {numel:>12,} params")

        print(f"\nDANA-STAR-MK4 parameters: {len(dana_params)}")
        print(f"Total DANA params: {dana_total_params:,} ({dana_total_params/1e6:.2f}M)")
        print("-" * 80)
        for pname, shape, numel in dana_params:
            print(f"  [DANA] {pname:50s} {str(shape):20s} {numel:>12,} params")

        print("\n" + "="*80)
        print(f"Total parameters: {muon_total_params + dana_total_params:,} ({(muon_total_params + dana_total_params)/1e6:.2f}M)")
        print(f"Muon ratio: {100*muon_total_params/(muon_total_params + dana_total_params):.1f}%")
        print(f"DANA ratio: {100*dana_total_params/(muon_total_params + dana_total_params):.1f}%")
        print("="*80 + "\n")

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        dist_group,
        tp_group,
        dist_metas: Dict[torch.nn.Parameter, MuonDistMeta],
    ):
        """
        Enable distributed mode for Muon parameters.
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (
                    local_buffer_start,
                    local_buffer_start + local_buffer_size,
                )
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def _clip_to_half(self, tau: torch.Tensor) -> torch.Tensor:
        """Clip tau values to at most 0.5 (DANA-STAR-MK4)"""
        return torch.clamp(tau, max=0.5)

    def _tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Tau regularization function (DANA-STAR-MK4)"""
        clipped_tau = self._clip_to_half(tau)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = torch.full_like(tau, 1.0 / (1.0 + step))
        return torch.maximum(p_estimate, min_p)

    def _root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Square root of tau regularization (DANA-STAR-MK4)"""
        return torch.sqrt(self._tau_reg(tau, step))

    def _effective_time(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Compute effective time for tau regularization (DANA-STAR-MK4)"""
        return torch.maximum(tau * step, torch.ones_like(tau))

    def _tau_updater(
        self,
        g: torch.Tensor,  # gradient
        v: torch.Tensor,  # second moment
        epsilon: float,
    ) -> torch.Tensor:
        """Update tau estimates (DANA-STAR-MK4)"""
        return torch.abs(g) / (torch.abs(g) + torch.sqrt(v) + epsilon)

    def _norm_term(
        self,
        v: torch.Tensor,
        tau: torch.Tensor,
        step: int,
        epsilon: float,
    ) -> torch.Tensor:
        """Compute normalization term (DANA-STAR-MK4)"""
        root_tau_reg = self._root_tau_reg(tau, step)
        return root_tau_reg / (torch.sqrt(v) + epsilon)

    def step(self):

        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # =====================================================================
        # MUON MOMENTUM UPDATE (with optional DANA-STAR-MK4 style EMA)
        # =====================================================================
        for group in self.param_groups:

            momentum = group["momentum"]
            dana_momentum = group["dana_momentum"]
            delta = group["delta"]
            weight_time = group["weight_time"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                # 1-dim grad for distributed mode
                assert self.distributed_mode or g.dim() == 2

                # prepare muon buffer in state
                state = self.state[p]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                    state["step"] = 0

                buf = state["muon_buffer"]
                state["step"] += 1

                if dana_momentum:
                    # DANA-STAR-MK4 style EMA update
                    step = state["step"]
                    time_factor = group["lr"] / self.lr if weight_time else 1.0
                    group["weighted_step_count"] += time_factor

                    if weight_time:
                        step = group["weighted_step_count"]
                        alpha = delta / (delta + step) * time_factor
                    else:
                        alpha = delta / (delta + step)

                    buf.mul_(1 - alpha).add_(g, alpha=alpha)
                else:
                    # Standard fixed momentum update
                    buf.mul_(momentum).add_(g)

                # save to ns input
                if dana_momentum:
                    # When using dana_momentum, we don't use nesterov
                    g = buf
                else:
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_inputs[p] = g.bfloat16()

        # =====================================================================
        # DISTRIBUTED ALL-GATHER FOR MUON PARAMETERS
        # =====================================================================
        if self.distributed_mode:

            # initialize buffers
            ns_input_local_buffers = [
                [
                    torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes
                ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [
                    torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes
                ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_range = normalize_range(
                    dist_meta.local_range, local_buffer_range[0]
                )
                ns_input_local_buffer[local_range[0] : local_range[1]].copy_(
                    ns_input.view(-1)
                )

            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(
                ns_input_global_buffers, ns_input_local_buffers
            ):
                for ns_input_global_bucket, ns_input_local_bucket in zip(
                    ns_input_global_buffer, ns_input_local_buffer
                ):
                    dist.all_gather_into_tensor(
                        ns_input_global_bucket,
                        ns_input_local_bucket,
                        group=self.dist_group,
                    )

            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # =====================================================================
        # MUON PARAMETER UPDATE (Newton-Schulz orthogonalization)
        # =====================================================================
        for group in self.param_groups:

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    ns_input_shards = [
                        torch.empty_like(ns_input) for _ in range(tp_world_size)
                    ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                # only local buffer part
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    update = update.reshape(-1)[
                        local_range_in_global_range[0] : local_range_in_global_range[1]
                    ]

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)

                #  adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # =====================================================================
        # DANA-STAR-MK4 UPDATE FOR NON-MUON PARAMETERS
        # =====================================================================
        for group in self.param_groups:

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            delta = group["delta"]
            kappa = group["kappa"]
            mk4A = group["mk4A"]
            mk4B = group["mk4B"]
            clipsnr = group["clipsnr"]
            epsilon = group["epsilon"]
            weight_time = group["weight_time"]
            wd_decaying = group["wd_decaying"]
            wd_ts = group["wd_ts"]
            params = group["params"]

            for p in params:

                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                # State initialization for DANA-STAR-MK4
                if "dana_m" not in state:
                    state["dana_m"] = torch.zeros_like(g)
                    state["dana_v"] = torch.zeros_like(g)
                    state["dana_tau"] = torch.zeros_like(g)
                    state["step"] = 0

                m = state["dana_m"]
                v = state["dana_v"]
                tau = state["dana_tau"]
                state["step"] += 1

                # Calculate alpha for EMA
                time_factor = lr / self.lr if weight_time else 1.0
                group["weighted_step_count"] += time_factor

                if weight_time:
                    step = group["weighted_step_count"]
                    alpha = delta / (delta + step) * time_factor
                else:
                    step = state["step"]
                    alpha = delta / (delta + step)

                # Update first moment
                m.mul_(1 - alpha).add_(g, alpha=alpha)
                # Update second moment
                v.mul_(1 - alpha).addcmul_(g, g, value=alpha)

                # Update tau using the specified tau updater
                tau_update = self._tau_updater(g, v, epsilon)
                tau.mul_(1 - alpha).add_(tau_update, alpha=alpha)

                # Compute effective time
                effective_time = self._effective_time(tau, step)

                # Store current alpha for logging
                state["current_alpha"] = alpha

                # Compute momentum terms
                norm_term = self._norm_term(v, tau, step, epsilon)

                # DANA-STAR-MK4 formula 13
                mfac = norm_term * torch.abs(m) / self._tau_reg(tau, step)
                alpha_factor = torch.clamp(
                    (effective_time ** (1 - kappa))
                    * (mfac ** (2 * mk4B + 1))
                    * (norm_term ** (2 * (-mk4A - mk4B))),
                    max=clipsnr,
                )
                g3_term = lr * (
                    self._tau_reg(tau, step) * torch.sign(m) * alpha_factor
                    + 1.0 * m * norm_term
                )
                state["current_alpha_factor"] = alpha_factor.mean().detach()
                state["gradient_norm"] = g.norm().detach()
                state["auto_factor_mean"] = mfac.mean().detach()
                state["m_norm"] = m.norm().detach()

                # Compute parameter updates
                g2_term = lr * g * norm_term

                # Apply the main update
                update = -(g2_term + g3_term)

                # Decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    if wd_decaying:
                        p.add_(p, alpha=-weight_decay / (1 + step / wd_ts) * lr)
                    else:
                        p.add_(p, alpha=-weight_decay * lr)

                # Apply update to parameters
                p.add_(update)
