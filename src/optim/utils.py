import math
import os
import random
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb


# Background rsync thread handle for checkpoint reliability
_rsync_thread = None


def _rsync_checkpoint_to_project(local_ckpt_dir, project_ckpt_dir):
    """Background rsync from local SLURM_TMPDIR checkpoint to project directory.

    Uses atomic pattern: rsync to temp dir, then mv to final location.
    On failure, leaves temp dir intact (does not corrupt existing checkpoint).
    """
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    project_parent = Path(project_ckpt_dir).parent
    project_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = project_parent / f".ckpt_tmp_{job_id}"
    final_dir = Path(project_ckpt_dir)

    try:
        # rsync local checkpoint to temp dir
        subprocess.run(
            ["rsync", "-a", "--delete", f"{local_ckpt_dir}/", f"{tmp_dir}/"],
            check=True,
            capture_output=True,
        )
        # Atomic rename: remove old final, move temp to final
        if final_dir.exists():
            # Rename old to .old, then rename new, then remove old
            old_dir = project_parent / f".ckpt_old_{job_id}"
            if old_dir.exists():
                subprocess.run(["rm", "-rf", str(old_dir)], check=False)
            final_dir.rename(old_dir)
            tmp_dir.rename(final_dir)
            subprocess.run(["rm", "-rf", str(old_dir)], check=False)
        else:
            tmp_dir.rename(final_dir)
        print(f"[Checkpoint] rsync complete: {local_ckpt_dir} -> {final_dir}")
    except Exception as e:
        print(f"[Checkpoint] WARNING: rsync failed ({e}), temp dir left at {tmp_dir}")


def _background_rsync(local_ckpt_dir, project_ckpt_dir):
    """Launch background rsync, waiting for any previous rsync to complete first."""
    global _rsync_thread
    if _rsync_thread is not None and _rsync_thread.is_alive():
        _rsync_thread.join()  # Wait for previous rsync before starting new one
    _rsync_thread = threading.Thread(
        target=_rsync_checkpoint_to_project,
        args=(str(local_ckpt_dir), str(project_ckpt_dir)),
        daemon=True,
    )
    _rsync_thread.start()


def wait_for_rsync():
    """Wait for any in-flight background rsync to complete."""
    global _rsync_thread
    if _rsync_thread is not None and _rsync_thread.is_alive():
        _rsync_thread.join()


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    moe=False,
    get_router_logits=False,
    cfg=None,
):
    assert model.training == False

    loss_list_val, acc_list, loss_list_aux_val = [], [], {}
    router_logits = []

    for idx in range(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, moe=moe)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        # auxiliary losses are optional
        for k, v in outputs["aux_losses"].items():
            loss_list_aux_val[k] = loss_list_aux_val.get(k, [])
            loss_list_aux_val[k].append(v)

        # router logits for MoE visualization
        if get_router_logits:
            # shape [layers, batch_size * sequence_length, num_experts]
            logits = outputs["router_logits"]
            # shape [max_batches, layers, batch_size * sequence_length, num_experts]
            router_logits.append(logits)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss
    val_aux_losses = {
        f"val/{k}": torch.stack(v).mean().item() for k, v in loss_list_aux_val.items()
    }

    if get_router_logits:
        # filter out the router logits that are not of the expected shape (happens for the last batch in
        # dataloader has a different batch size than the others)
        if cfg:
            intended_size = cfg.batch_size * cfg.sequence_length
        else:
            intended_size = x.shape[0] * x.shape[1]
        # shape [batches - 1, layers, batch_size * sequence_length, num_experts]
        router_logits = (
            torch.stack(
                [rl for rl in router_logits if rl.shape[1] == intended_size],
                dim=0,
            )
            .detach()
            .cpu()
        )

    return val_acc, val_loss, val_perplexity, val_aux_losses, router_logits


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def _is_fsdp_backend(distributed_backend):
    return distributed_backend is not None and getattr(distributed_backend, '_is_fsdp', False)


def _get_local_ckpt_dir(ckpt_dir: Path):
    """If SLURM_TMPDIR is set, return a local NVMe path for fast writes.
    Otherwise return ckpt_dir unchanged. Also returns the project dir."""
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    project_ckpt_dir = ckpt_dir
    if slurm_tmpdir:
        local_base = Path(slurm_tmpdir) / "ckpts"
        ckpt_dir = local_base / ckpt_dir.name
    return ckpt_dir, project_ckpt_dir, slurm_tmpdir is not None


def rsync_if_slurm(ckpt_dir: Path):
    """If SLURM_TMPDIR is set, kick off a background rsync of the local
    checkpoint dir to the project directory. Returns immediately.
    Should be called by rank 0 only, AFTER all ranks have finished writing."""
    local_dir, project_dir, is_slurm = _get_local_ckpt_dir(ckpt_dir)
    if is_slurm:
        _background_rsync(local_dir, project_dir)


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path, distributed_backend=None):
    # Determine if we should write to SLURM_TMPDIR first for speed + reliability.
    local_dir, project_ckpt_dir, is_slurm = _get_local_ckpt_dir(ckpt_dir)
    ckpt_dir = local_dir

    if is_slurm:
        # Wait for any in-flight rsync of the previous checkpoint to finish
        # before we overwrite the local dir with new data.
        wait_for_rsync()
        rank = 0 if not dist.is_initialized() else dist.get_rank()
        if rank == 0:
            print(f"[Checkpoint] Writing to local: {ckpt_dir} (will rsync to {project_ckpt_dir})")

    ckpt_dir.mkdir(exist_ok=True, parents=True)

    if _is_fsdp_backend(distributed_backend):
        # FSDP: use torch.distributed.checkpoint for sharded state dicts.
        # All ranks must participate in this call.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
        )

        model_sd = get_model_state_dict(model)
        opt_sd = get_optimizer_state_dict(model, opt)
        state_dict = {
            "model": model_sd,
            "optimizer": opt_sd,
        }
        dcp.save(state_dict, checkpoint_id=str(ckpt_dir / "sharded"))

        # Save scheduler and iteration on rank 0 only (small, not sharded)
        rank = 0 if not dist.is_initialized() else dist.get_rank()
        if rank == 0:
            meta = {
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "itr": itr,
            }
            torch.save(meta, ckpt_dir / "meta.pt")
    else:
        # DDP / single: standard full state dict save
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "itr": itr,
        }
        torch.save(checkpoint, ckpt_dir / "main.pt")

    # NOTE: rsync is NOT done here. The caller (base.py) must call
    # rsync_if_slurm() after all ranks finish save_worker_state too.


def load_checkpoint(model, opt, scheduler, ckpt_path, device, distributed_backend=None):
    if _is_fsdp_backend(distributed_backend):
        # FSDP: use torch.distributed.checkpoint for sharded state dicts.
        # ckpt_path is expected to be e.g. <ckpt_dir>/main.pt; derive ckpt_dir.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, get_optimizer_state_dict,
            set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
        )

        ckpt_dir = ckpt_path.parent if hasattr(ckpt_path, 'parent') else Path(ckpt_path).parent

        model_sd = get_model_state_dict(model)
        opt_sd = get_optimizer_state_dict(model, opt)
        state_dict = {
            "model": model_sd,
            "optimizer": opt_sd,
        }
        dcp.load(state_dict, checkpoint_id=str(ckpt_dir / "sharded"))

        set_model_state_dict(model, state_dict["model"])
        set_optimizer_state_dict(model, opt, state_dict["optimizer"])

        # Load scheduler and iteration from meta (saved by rank 0)
        try:
            meta = torch.load(ckpt_dir / "meta.pt", map_location='cpu')
        except Exception:
            meta = torch.load(ckpt_dir / "meta.pt", map_location='cpu', weights_only=False)
        if scheduler is not None and meta.get("scheduler") is not None:
            scheduler.load_state_dict(meta["scheduler"])
        return meta["itr"]
    else:
        # DDP / single: standard full state dict load
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        except Exception:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        itr = ckpt["itr"]
        return itr


def save_worker_state(ckpt_dir: Path, train_reader=None):
    # Write to SLURM_TMPDIR if available (same dir as save_checkpoint),
    # so the rsync in rsync_if_slurm() picks up everything atomically.
    local_dir, _, _ = _get_local_ckpt_dir(ckpt_dir)
    ckpt_dir = local_dir

    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    # Get GPU RNG state only if CUDA is available
    try:
        worker_state["rng_torch_gpu"] = torch.cuda.get_rng_state()
    except (AttributeError, RuntimeError):
        worker_state["rng_torch_gpu"] = None
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    # Save dataloader state if available
    if train_reader is not None and hasattr(train_reader, 'state_dict'):
        dataloader_state = train_reader.state_dict()
        worker_state["dataloader_state"] = dataloader_state
        
        # Print checkpoint info for each rank
        seed = getattr(train_reader, 'seed', None)
        step = dataloader_state.get('step', None)
        if hasattr(train_reader, 'current_file_idx'):
            # MultiFileDataReader case
            file_idx = dataloader_state.get('current_file_idx', None)
            # Compute actual seed used for current file: seed + file_idx * 1000
            file_seed = seed + file_idx * 1000 if (seed is not None and file_idx is not None) else None
            if file_idx is not None and hasattr(train_reader, 'file_paths') and file_idx < len(train_reader.file_paths):
                file_name = train_reader.file_paths[file_idx].name
                print(f"[Rank {rank}] Checkpoint: data_seed={seed}, file_idx={file_idx} ({file_name}), file_seed={file_seed}, step={step}")
            else:
                print(f"[Rank {rank}] Checkpoint: data_seed={seed}, file_idx={file_idx}, file_seed={file_seed}, step={step}")
            
            # Preview next data points (from underlying reader)
            if hasattr(train_reader, 'current_reader') and train_reader.current_reader is not None:
                if hasattr(train_reader.current_reader, '_preview_next_samples'):
                    train_reader.current_reader._preview_next_samples(num_preview=3)
        else:
            # Regular DataReader
            print(f"[Rank {rank}] Checkpoint: data_seed={seed}, step={step}")
            
            # Preview next data points
            if hasattr(train_reader, '_preview_next_samples'):
                train_reader._preview_next_samples(num_preview=3)
    
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path, train_reader=None):
    rank = 0 if not dist.is_initialized() else dist.get_rank()

    # PyTorch 2.6 defaults to weights_only=True; worker state includes numpy/python RNG states
    # Try safe default first, then fall back to weights_only=False if needed
    try:
        worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt")
    except Exception:
        worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt", weights_only=False)
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    if worker_state.get("rng_torch_gpu", None) is not None:
        try:
            torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
        except (AttributeError, RuntimeError):
            pass  # CUDA not available, skip GPU RNG restoration
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])
    
    # Restore dataloader state if available
    if train_reader is not None and hasattr(train_reader, 'load_state_dict'):
        if "dataloader_state" in worker_state:
            dataloader_state = worker_state["dataloader_state"]
            train_reader.load_state_dict(dataloader_state)
            
            # Print restored state info for each rank
            seed = getattr(train_reader, 'seed', None)
            step = dataloader_state.get('step', None)
            if hasattr(train_reader, 'current_file_idx'):
                # MultiFileDataReader case
                file_idx = dataloader_state.get('current_file_idx', None)
                # Compute actual seed used for current file: seed + file_idx * 1000
                file_seed = seed + file_idx * 1000 if (seed is not None and file_idx is not None) else None
                if file_idx is not None and hasattr(train_reader, 'file_paths') and file_idx < len(train_reader.file_paths):
                    file_name = train_reader.file_paths[file_idx].name
                    print(f"[Rank {rank}] Restored: data_seed={seed}, file_idx={file_idx} ({file_name}), file_seed={file_seed}, step={step}")
                else:
                    print(f"[Rank {rank}] Restored: data_seed={seed}, file_idx={file_idx}, file_seed={file_seed}, step={step}")
            else:
                # Regular DataReader
                print(f"[Rank {rank}] Restored: data_seed={seed}, step={step}")
        else:
            print(f"[Rank {rank}] Warning: No dataloader state in checkpoint")


def get_parameter_norms(model, order=2):
    model_norm = 0
    for p in model.parameters():
        param_data = p.detach().data
        if order == float("inf"):
            param_norm = param_data.norm(p=order)
            model_norm = max(model_norm, param_norm.item())
        else:
            param_norm = param_data.norm(p=order)
            model_norm += param_norm.item() ** order

    if order != float("inf"):
        model_norm = model_norm ** (1.0 / order)

    return model_norm


def log_prodigy_lr(opt):
    effective_lrs = []

    for group in opt.param_groups:
        d = group["d"]
        lr = group["lr"]
        if group["use_bias_correction"]:
            k = group["k"]
            beta1, beta2 = group["betas"]
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1
        effective_lr = d * lr * bias_correction
        effective_lrs.append(effective_lr)

    return effective_lrs


def visualize_routing(router_logits, extra_args):
    # router_logits: [batches, layers, batch_size * sequence_length, num_experts]
    logs = {}

    n_layers = extra_args.n_layer
    num_experts = extra_args.moe_num_experts
    num_experts_per_tok = extra_args.moe_num_experts_per_tok

    # histogram over all logits to see distribution
    logs["router/logits"] = wandb.Histogram(
        router_logits.type(torch.float32).flatten().cpu().numpy()
    )

    # distribution over experts for layer 0, layer n/2, n-1
    for layer in [0, n_layers // 2, n_layers - 1]:
        router_logits_layer = router_logits[:, layer]
        # shape [batches, batch_size * sequence_length, num_experts_per_tok]
        weights, selected_experts = torch.topk(
            router_logits_layer, num_experts_per_tok, dim=-1
        )
        # shape [batches, batch_size * sequence_length, num_experts_per_tok, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [batches, batch_size * sequence_length, num_experts]
        expert_mask, _ = torch.max(expert_mask, dim=-2)
        # shape [num_experts]
        tokens_per_expert = torch.mean(expert_mask, dim=(0, 1), dtype=torch.float32)
        layer_token_routing = {
            f"router/layer_{layer}_expert_{i}_selection": tokens_per_expert[i].item()
            for i in range(num_experts)
        }
        logs.update(layer_token_routing)
    return logs


def log_optimizer_schedules(optimizer, optimizer_name):
    """
    Log optimizer-specific scheduling parameters to wandb.

    For AdEMAMix: logs alpha(t) and 1-beta_3(t) schedules
    For DANA/DANA-STAR: logs alpha factor and (1+t)**(1-kappa) schedules
    """
    logs = {}

    def _to_float(v):
        """Safely convert a value to float, handling DTensors and unallocated tensors."""
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return v.item() if hasattr(v, 'item') else float(v)
        except (RuntimeError, ValueError):
            return None
    
    if optimizer_name in ["ademamix"]:
        # Log AdEMAMix schedules: alpha(t) and 1-beta_3(t)
        alpha_values = []
        one_minus_beta3_values = []

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in optimizer.state and "current_alpha" in optimizer.state[param]:
                    state = optimizer.state[param]
                    v = _to_float(state["current_alpha"])
                    if v is not None:
                        alpha_values.append(v)
                    v = _to_float(state["current_one_minus_beta3"])
                    if v is not None:
                        one_minus_beta3_values.append(v)

        if alpha_values:
            logs["optimizer/alpha_schedule"] = sum(alpha_values) / len(alpha_values)
            if one_minus_beta3_values:
                logs["optimizer/one_minus_beta3_schedule"] = sum(one_minus_beta3_values) / len(one_minus_beta3_values)
    
    elif optimizer_name in ["adana", "dana-mk4", "dana-star", "dana-star-mk4"]:
        # Log DANA schedules: alpha factor and (1+t)**(1-kappa)
        alpha_values = []
        kappa_factor_values = []
        auto_factor_values = []
        g2_gradient_norm_values = []
        m_norm_values = []

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in optimizer.state and "current_alpha" in optimizer.state[param]:
                    state = optimizer.state[param]
                    if "current_alpha" in state:
                        v = _to_float(state["current_alpha"])
                        if v is not None:
                            alpha_values.append(v)
                    if "current_kappa_factor" in state:
                        v = _to_float(state["current_kappa_factor"])
                        if v is not None:
                            kappa_factor_values.append(v)
                    if "auto_factor_mean" in state:
                        v = _to_float(state["auto_factor_mean"])
                        if v is not None:
                            auto_factor_values.append(v)
                    if "g2_gradient_norm" in state:
                        v = _to_float(state["g2_gradient_norm"])
                        if v is not None:
                            g2_gradient_norm_values.append(v)
                    if "m_norm" in state:
                        v = _to_float(state["m_norm"])
                        if v is not None:
                            m_norm_values.append(v)

        if alpha_values:
            logs["optimizer/alpha_schedule"] = sum(alpha_values) / len(alpha_values)
            if kappa_factor_values:
                logs["optimizer/kappa_factor_schedule"] = sum(kappa_factor_values) / len(kappa_factor_values)
            if auto_factor_values:
                logs["optimizer/auto_factor"] = sum(auto_factor_values) / len(auto_factor_values)
            if g2_gradient_norm_values:
                logs["optimizer/g2_gradient_norm"] = sum(g2_gradient_norm_values) / len(g2_gradient_norm_values)
            if m_norm_values:
                logs["optimizer/m_norm"] = sum(m_norm_values) / len(m_norm_values)
    
    return logs
