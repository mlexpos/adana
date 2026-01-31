import math
import os
from contextlib import contextmanager

import torch
from torch.distributed import (destroy_process_group, get_world_size,
                               init_process_group)

from .backend import DistributedBackend

# FSDP2 composable API (PyTorch 2.4+)
try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


class FSDPDistributedBackend(DistributedBackend):
    """FSDP2 (per-parameter sharding) distributed backend.

    Uses the composable ``fully_shard`` API with FULL_SHARD strategy.
    Mixed precision: bf16 compute, fp32 params.
    Checkpoints use ``torch.distributed.checkpoint`` for sharded state dicts.

    When ``tp_size > 1``, a 2-D DeviceMesh (dp, tp) is created. Tensor
    parallelism is applied per-block *before* FSDP wrapping, and FSDP uses
    the ``dp`` sub-mesh so that only data-parallel ranks shard parameters.
    """

    _is_fsdp = True

    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", -1))
        assert self.rank != -1, "FSDP backend requires RANK to be set (use torchrun)"
        assert "cuda" in args.device, "FSDP backend requires CUDA devices"
        init_process_group(backend="nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])

        # Mixed precision policy: compute in bf16, keep params in fp32
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
        )

        # Tensor parallelism configuration
        self.tp_size = getattr(args, "tp_size", 1)
        world_size = self.get_world_size()

        if self.tp_size > 1:
            assert world_size % self.tp_size == 0, (
                f"World size ({world_size}) must be divisible by tp_size ({self.tp_size})"
            )
            from torch.distributed.device_mesh import init_device_mesh
            self.device_mesh = init_device_mesh(
                "cuda", (world_size // self.tp_size, self.tp_size),
                mesh_dim_names=("dp", "tp"),
            )
            self.tp_mesh = self.device_mesh["tp"]
            self.dp_mesh = self.device_mesh["dp"]
        else:
            self.device_mesh = None
            self.tp_mesh = None
            self.dp_mesh = None

    @property
    def dp_size(self):
        """Number of data-parallel ranks (world_size // tp_size)."""
        return self.get_world_size() // self.tp_size

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        # Only data-parallel ranks split the batch; TP ranks process the same data.
        dp_world = self.dp_size
        if effective_batch_size % dp_world != 0:
            raise ValueError(
                f"Effective batch size {effective_batch_size} is not divisible "
                f"by the data-parallel world size {dp_world} "
                f"(world_size={self.get_world_size()}, tp_size={self.tp_size})."
            )
        acc_steps_div = math.gcd(args.acc_steps, dp_world)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (dp_world // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.local_rank
        args.data_seed = args.data_seed
        return args

    def transform_model(self, model):
        # Apply TP first (if enabled), then FSDP.
        if self.tp_size > 1:
            from torch.distributed.tensor.parallel import parallelize_module

            # Validate n_head divisibility
            n_head = model.config.n_head
            assert n_head % self.tp_size == 0, (
                f"n_head ({n_head}) must be divisible by tp_size ({self.tp_size})"
            )

            tp_plan = model.get_tp_plan()

            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                for block in model.transformer.h:
                    parallelize_module(block, self.tp_mesh, tp_plan)

        # Apply FSDP per-block, then to the full model.
        # All model architectures use model.transformer.h for blocks.
        fsdp_kwargs = dict(mp_policy=self.mp_policy)
        if self.dp_mesh is not None:
            fsdp_kwargs["mesh"] = self.dp_mesh

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for block in model.transformer.h:
                fully_shard(block, **fsdp_kwargs)

        fully_shard(model, **fsdp_kwargs)
        return model

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0):
        """Gradient clipping that handles mixed-mesh DTensors from 2D parallelism.

        When tp_size > 1, TP parameters live on the (dp, tp) mesh while
        FSDP-only parameters (embeddings, norms) live on the (dp,) sub-mesh.
        Standard clip_grad_norm_ fails because torch.stack can't combine
        DTensors from different meshes.  We compute per-param norms as plain
        tensors first, then clip.
        """
        if self.tp_size <= 1:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        from torch.distributed.tensor import DTensor

        params = [p for p in parameters if p.grad is not None]
        if len(params) == 0:
            return torch.tensor(0.0)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        device = params[0].grad.device

        # Compute per-parameter gradient norms, converting DTensor scalars to
        # plain tensors so they can be stacked regardless of source mesh.
        norms = []
        for p in params:
            g = p.grad.detach()
            if isinstance(g, DTensor):
                # full_tensor() materialises the global tensor (cheap for a
                # scalar result of norm, which is Replicate after the reduction).
                n = g.full_tensor().norm(norm_type)
            else:
                n = g.norm(norm_type)
            norms.append(n.to(device))

        total_norm = torch.norm(torch.stack(norms), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in params:
            p.grad.detach().mul_(clip_coef_clamped)
        return total_norm

    @contextmanager
    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        # FSDP2 composable API: set_requires_gradient_sync controls all-reduce.
        # Skip sync on non-final microsteps for gradient accumulation.
        if microstep_idx < gradient_accumulation_steps - 1:
            model.set_requires_gradient_sync(False)
        else:
            model.set_requires_gradient_sync(True)
        try:
            yield
        finally:
            pass

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        # FSDP2 composable API modifies model in-place, no wrapper to unwrap.
        return model

    def translate_model_parameter_name_for_node(self, parameter_name):
        # FSDP2 composable API preserves parameter names.
        return [parameter_name]

    def get_world_size(self):
        return get_world_size()

    def all_ranks_checkpoint(self):
        """FSDP requires all ranks to participate in checkpoint save/load."""
        return True

    def finalize(self):
        destroy_process_group()
