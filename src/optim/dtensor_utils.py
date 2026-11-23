"""Helpers for optimizers whose parameters may be FSDP2 DTensors (each matrix sharded by rows across the data-parallel ranks).

Under DDP every tensor is a plain torch.Tensor and every helper is the identity.  Under FSDP2 parameters, gradients and optimizer
state are DTensors: elementwise updates work on them as they are, but whole-matrix operations (Newton-Schulz, the Lanczos products
of the SLQ refresh) need the full matrix, and per-tensor reductions are better done locally and summed in one collective.
"""
import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:                       # older torch without DTensor: nothing is ever sharded
    DTensor = Replicate = None


def is_dtensor(t):
    return DTensor is not None and isinstance(t, DTensor)


def full(t):
    """The whole tensor as a plain tensor (an all-gather if t is sharded)."""
    return t.full_tensor() if is_dtensor(t) else t


def local(t):
    """This rank's shard as a plain tensor (t itself if not sharded)."""
    return t.to_local() if is_dtensor(t) else t


def like(full_t, ref):
    """A plain tensor of ref's global shape, placed like ref: this rank's shard if ref is a DTensor (Replicate -> Shard is a local
    slice, no communication)."""
    if not is_dtensor(ref):
        return full_t
    mesh = ref.device_mesh
    return DTensor.from_local(full_t, mesh, [Replicate()] * mesh.ndim, run_check=False).redistribute(mesh, ref.placements)


def matrix_fn(fn, x):
    """fn applied to the whole matrix x (gathered once if x is sharded), returned placed like x."""
    if not is_dtensor(x):
        return fn(x)
    return like(fn(x.full_tensor()), x)


def sum_local_scalars(vals, sharded):
    """Per-tensor scalars computed on local shards -> their global values, in one all-reduce.  vals: list of 0-d tensors."""
    if not vals:
        return []
    v = torch.stack([x.float() for x in vals])
    if sharded and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(v)
    return v.tolist()


def materialize(shadow, model):
    """Load the full weights and buffers of `model` (an FSDP2-sharded module) into `shadow`, an unwrapped copy of the same
    architecture built on the meta device.  Every rank gathers every parameter (collective, same order on all ranks).  The
    shadow's parameters are fresh fp32 leaf tensors on the model's device; returns {name: parameter} of the shadow."""
    mods = dict(shadow.named_modules())
    out = {}
    for name, p in model.named_parameters():
        mod, _, leaf = name.rpartition(".")
        t = full(p.detach()).float().clone()
        mods[mod]._parameters[leaf] = torch.nn.Parameter(t, requires_grad=p.requires_grad)
        out[name] = mods[mod]._parameters[leaf]
    for name, b in model.named_buffers():
        mod, _, leaf = name.rpartition(".")
        mods[mod]._buffers[leaf] = full(b).clone()
    # plain tensor attributes that are neither parameters nor buffers (e.g. rotary tables such as freqs_cis): FSDP2 leaves them
    # alone, so the sharded model holds the real ones; the shadow points at the same tensors (no copy)
    for mname, m in model.named_modules():
        if mname not in mods:
            continue
        sm = mods[mname]
        for k, v in vars(m).items():
            if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter) and not k.startswith("_"):
                object.__setattr__(sm, k, v)
    # parameters shared between modules (weight tying) appear once in named_parameters: point every alias at the same tensor
    for mname, m in shadow.named_modules():
        for leaf, p in list(m._parameters.items()):
            if p is not None and p.device.type == "meta":
                key = f"{mname}.{leaf}" if mname else leaf
                raise RuntimeError(f"materialize: shadow parameter {key} has no counterpart in the sharded model")
    return out


def release(shadow):
    """Free the shadow's tensors (back to the meta device)."""
    shadow.to("meta")
