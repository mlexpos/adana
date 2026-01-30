from . import ddp, fsdp, single

BACKEND_TYPE_TO_MODULE_MAP = {
    "nccl": ddp.DataParallelDistributedBackend,  # GPU DDP
    "gloo": ddp.DataParallelDistributedBackend,  # CPU DDP
    "fsdp": fsdp.FSDPDistributedBackend,         # GPU FSDP2
    "single": single.SinlgeNodeBackend,          # Explicit single-process backend
    None: single.SinlgeNodeBackend,               # Backward-compatible default
}


def make_backend_from_args(args):
    return BACKEND_TYPE_TO_MODULE_MAP[args.distributed_backend](args)


def registered_backends():
    return BACKEND_TYPE_TO_MODULE_MAP.keys()
