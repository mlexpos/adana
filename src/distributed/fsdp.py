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

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(
                f"Effective batch size {effective_batch_size} is not divisible "
                f"by the world size {world_size}."
            )
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.local_rank
        args.data_seed = args.data_seed
        return args

    def transform_model(self, model):
        # Apply FSDP per-block, then to the full model.
        # All model architectures use model.transformer.h for blocks.
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for block in model.transformer.h:
                fully_shard(block, mp_policy=self.mp_policy)

        fully_shard(model, mp_policy=self.mp_policy)
        return model

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
