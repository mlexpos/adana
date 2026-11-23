import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml
import torch.distributed as dist
from logger.logger import DynamicsLogger

from .utils import (eval, get_batch, get_parameter_norms, load_checkpoint,
                    load_worker_state, log_prodigy_lr, log_optimizer_schedules,
                    save_checkpoint, save_worker_state, rsync_if_slurm,
                    visualize_routing, wait_for_rsync, all_ranks_ok, has_tensor_state,
                    CKPT_WAIT, CKPT_FINAL_WAIT)
from .tau_stats_collector import TauStatsCollector


class _StepProfiler:
    """ADANA_PROFILE="start:steps": a torch.profiler trace of optimizer steps start .. start+steps-1 (rank 0 only; the other ranks run
    unprofiled), for performance analysis.  Writes <exp_dir>/profile_rank0.json: every kernel and every aten op with device time,
    grouped by name and input shapes (GEMMs can then be attributed to layers by shape), and the wall time of the profiled steps;
    ADANA_PROFILE_TRACE=1 also writes a Chrome trace.  Off unless the variable is set."""

    def __init__(self, exp_dir, master):
        spec = __import__("os").environ.get("ADANA_PROFILE", "")
        self.on = bool(spec) and master
        if self.on:
            a, b = spec.split(":")
            self.start, self.steps, self.exp_dir, self.prof = int(a), int(b), Path(exp_dir), None

    def tick(self, curr_iter):
        if not self.on:
            return
        if curr_iter == self.start:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            self.prof.__enter__()
            self.t0 = time.perf_counter()
        elif curr_iter == self.start + self.steps and self.prof is not None:
            torch.cuda.synchronize()
            wall = time.perf_counter() - self.t0
            self.prof.__exit__(None, None, None)
            import json
            import os
            rows = []
            for e in self.prof.key_averages(group_by_input_shape=True):
                dt = float(getattr(e, "device_time_total", 0) or 0)
                sdt = float(getattr(e, "self_device_time_total", 0) or 0)
                if dt <= 0 and sdt <= 0:
                    continue
                rows.append(dict(key=e.key, shapes=str(e.input_shapes), count=int(e.count), self_us=sdt, total_us=dt,
                                 device_type=str(getattr(e, "device_type", ""))))
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with open(self.exp_dir / "profile_rank0.json", "w") as f:
                json.dump(dict(start=self.start, steps=self.steps, wall_s=wall, rows=rows), f)
            if os.environ.get("ADANA_PROFILE_TRACE") == "1":
                self.prof.export_chrome_trace(str(self.exp_dir / "profile_rank0.trace.json"))
            print(f"[profile] steps {self.start}..{self.start + self.steps - 1}: {wall / self.steps:.3f} s/step; "
                  f"{len(rows)} rows -> {self.exp_dir / 'profile_rank0.json'}", flush=True)
            self.on = False


def _all_ranks_checkpoint(backend):
    """Check if checkpoint save/load requires all ranks (e.g. FSDP)."""
    return hasattr(backend, 'all_ranks_checkpoint') and backend.all_ranks_checkpoint()


def _checkpoint(label, ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend, train_reader,
                fsdp_all_ranks, wait=CKPT_WAIT):
    """Write one checkpoint; never stop training.

    Every rank calls this. All ranks agree to skip the checkpoint if the copy of the previous one to the project disk
    is still running after CKPT_WAIT seconds (the local files are still being read). If any rank fails to write its
    part, the checkpoint is not copied and the previous copy on the project disk is kept.
    """
    master = distributed_backend.is_master_process()
    if fsdp_all_ranks and not has_tensor_state(opt):
        # FSDP: before the first step there is nothing to save, and torch's get_optimizer_state_dict would create the missing
        # state by taking one optimizer step with zero gradients, which advances step counts (and the adana-slq estimator) of
        # the run itself
        if master:
            print(f"[Checkpoint] SKIP  {label} iter={curr_iter}: no optimizer state yet")
        return
    if not all_ranks_ok(wait_for_rsync(wait) if master else True, cfg.device):
        if master:
            print(f"[Checkpoint] SKIP  {label} iter={curr_iter}: previous copy still running after {wait:.0f}s; "
                  f"training continues")
        return
    if master:
        print(f"[Checkpoint] START {label} iter={curr_iter} -> {ckpt_dir}")
    ok = True
    try:
        if master or fsdp_all_ranks:
            save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, distributed_backend=distributed_backend)
        if master and cfg.wandb and wandb.run is not None:
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "wandb_run_id.txt", "w") as f:
                f.write(wandb.run.id)
        # All ranks save their own worker state (RNG + dataloader position)
        save_worker_state(ckpt_dir, train_reader=train_reader)
    except Exception as e:
        ok = False
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Checkpoint] WARNING rank {rank}: writing {label} iter={curr_iter} failed ({e!r}); training continues")
    # Also the barrier: all ranks have finished writing before rank 0 copies
    ok = all_ranks_ok(ok, cfg.device)
    if master:
        if ok:
            rsync_if_slurm(ckpt_dir)
            print(f"[Checkpoint] END   {label} iter={curr_iter} -> {ckpt_dir}")
        else:
            print(f"[Checkpoint] ABANDON {label} iter={curr_iter}: not copied; the previous copy is kept")


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()
    # Initialize datareaders early so we can restore their state if resuming
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    
    _fsdp_all_ranks = _all_ranks_checkpoint(distributed_backend)

    if cfg.resume_from :
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
            distributed_backend=distributed_backend,
        )
        # Load worker state including dataloader state
        load_worker_state(ckpt_dir, train_reader=train_reader)
        substep = curr_iter * cfg.acc_steps
    else:
        curr_iter = 0
        # Set initial step for non-resume case
        substep = curr_iter * cfg.acc_steps
        train_reader.set_step(substep)
    
    # Initialize iterations_in_run counter for iterations_to_run (always starts from 0, independent of checkpoint)
    iterations_in_run = 0

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    # Initialize tau statistics collector if requested
    tau_stats_collector = None
    if distributed_backend.is_master_process() and getattr(cfg, 'collect_tau_stats', False) and cfg.opt in ("dana-star-mk4", "dana-star"):
        # Extract parameter names from optimizer groups
        param_names = []
        for group in opt.param_groups:
            if 'param_names' in group:
                param_names.extend(group['param_names'])

        tau_stats_collector = TauStatsCollector(
            exp_dir=exp_dir,
            cfg=cfg,
            param_names=param_names,
        )

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    grad_norms = []
    model.train()

    # adana-slq: split-half gradient statistics (needs >= 2 micro-steps) and quadrature refreshes
    # split-batch statistics read p.grad half-way through the accumulation window.  DDP: the local half is all-reduced below.
    # FSDP2: gradients are only reduced (and p.grad populated) at a syncing micro-step, so the middle micro-step syncs as well (one
    # more reduce-scatter per step, as DDP's extra all-reduce), for the optimizers whose split statistics run on sharded gradients
    # (adana-slq) and whose rule reads them (split_needed: under FSDP the extra reduce-scatter is not spent on the B / B_noise
    # diagnostic alone); off for the others (version 0's rule does not use them)
    fsdp = getattr(distributed_backend, "_is_fsdp", False)
    slq_split = (hasattr(opt, "set_split_stats") and getattr(opt, "split_stats", True) and cfg.acc_steps >= 2
                 and (not fsdp or (getattr(opt, "fsdp_split_ok", False) and getattr(opt, "split_needed", True))))
    if slq_split and fsdp:
        distributed_backend.sync_microsteps = {cfg.acc_steps // 2 - 1}
    if hasattr(opt, "set_split_stats") and getattr(opt, "split_stats", True) and cfg.acc_steps < 2 and distributed_backend.is_master_process():
        print("[adana-slq] acc_steps < 2: no split-half noise statistics (mu = 1, water-filling falls back to global)")
    if hasattr(opt, "log_dir") and opt.log_dir is None and distributed_backend.is_master_process():
        opt.log_dir = Path(exp_dir)

    profiler = _StepProfiler(exp_dir, distributed_backend.is_master_process())
    deadline = float(__import__("os").environ.get("ADANA_DEADLINE", "0") or 0)
    if deadline and distributed_backend.is_master_process():
        print(f"[deadline] checkpoint and stop at {time.strftime('%F %T', time.localtime(deadline))}")
    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                _checkpoint("permanent", ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend, train_reader, _fsdp_all_ranks)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                _checkpoint("latest", ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend, train_reader, _fsdp_all_ranks)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        is_eval_step = (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        )
        if is_eval_step:
            eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )
            # FSDP with reshard_after_forward=False leaves the evaluation's gathered parameters in place; reshard so that
            # checkpoints and the optimizer always see sharded parameters (no-op for other backends)
            if hasattr(distributed_backend, "reshard_all"):
                distributed_backend.reshard_all(model)

            # Collect tau statistics if enabled
            if tau_stats_collector is not None and tau_stats_collector.should_collect(curr_iter, is_eval_step):
                tau_stats_collector.collect(opt, curr_iter)

        # Check if we've reached iterations_to_run (if set) - check at start of iteration
        # iterations_to_run is independent of checkpoint, always counts from 0
        if cfg.iterations_to_run is not None and iterations_in_run >= cfg.iterations_to_run:
            # Save checkpoint before stopping
            ckpt_dir = exp_dir / "ckpts" / "latest"
            _checkpoint(f"latest (iterations_to_run={iterations_in_run} reached)", ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend, train_reader, _fsdp_all_ranks)
            # Also save permanent checkpoint if interval is set
            if cfg.permanent_ckpt_interval > 0:
                if curr_iter % cfg.permanent_ckpt_interval == 0:
                    ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                    _checkpoint("permanent", ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend, train_reader, _fsdp_all_ranks)
            break

        # ADANA_DEADLINE (unix time; scripts/tamia_job.sh sets it a margin before the job's time limit): write the latest checkpoint
        # and stop, for the next job of the chain to resume.  Checked every 10 steps; every rank must agree (the FSDP checkpoint is
        # a collective), so the decision is all-reduced.
        if (deadline and iterations_in_run > 0 and curr_iter % 10 == 0 and curr_iter < cfg.iterations
                and not all_ranks_ok(time.time() < deadline, cfg.device)):
            ckpt_dir = exp_dir / "ckpts" / "latest"
            _checkpoint("latest (job deadline)", ckpt_dir, curr_iter, model, opt, scheduler, exp_dir, cfg, distributed_backend,
                        train_reader, _fsdp_all_ranks, wait=900)    # the margin covers a copy in flight
            if distributed_backend.is_master_process():
                print(f"[deadline] stopped at iteration {curr_iter} of {cfg.iterations}; the next job resumes from {ckpt_dir}")
            break

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        profiler.tick(curr_iter)

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, moe=cfg.moe)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1
            if slq_split and microstep_idx == cfg.acc_steps // 2 - 1:
                slq_half = [p.grad.detach().clone() if p.grad is not None else None for p in opt.plist]

        if slq_split:
            if not fsdp and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                live = [h for h in slq_half if h is not None]
                flat = torch.cat([h.reshape(-1).float() for h in live])
                dist.all_reduce(flat)
                flat /= dist.get_world_size()
                k = 0
                for h in live:
                    n = h.numel(); h.copy_(flat[k:k + n].view_as(h)); k += n
                del flat
            opt.set_split_stats(slq_half, [p.grad for p in opt.plist])
            del slq_half

        if hasattr(opt, "needs_refresh") and opt.needs_refresh():
            raw_for_slq = distributed_backend.get_raw_model(not_compiled_model)
            if getattr(opt, "cbatch", 0):
                # hybrid-v3: the step's training loss (tau / P) and a curvature batch from a stream the training never sees
                # (the validation reader; every evaluation resets it to step 0, so evaluations and training data are unchanged)
                opt.last_loss = float(loss.detach()) * cfg.acc_steps

                def curv(n):
                    # n sequences for this rank; the validation reader is identical on every rank, so draw world x n and take this
                    # rank's slice (distinct curvature sequences per GPU; the GN products are averaged across ranks)
                    ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
                    rk = dist.get_rank() if ws > 1 else 0
                    xs, ys = [], []
                    while sum(t.shape[0] for t in xs) < n * ws:
                        xv, yv = get_batch(val_reader, device=cfg.device); xs.append(xv); ys.append(yv)
                    return torch.cat(xs)[rk * n:(rk + 1) * n], torch.cat(ys)[rk * n:(rk + 1) * n]
                opt.refresh(raw_for_slq, x, y, curv=curv)
            else:
                opt.refresh(raw_for_slq, x, y)
            if hasattr(opt, "refresh_message"):
                msg = opt.refresh_message()
                if msg and distributed_backend.is_master_process():
                    print(msg)
            elif distributed_backend.is_master_process():
                lr_ = opt.last_refresh
                print(f"[adana-slq] refresh t={lr_['t']}: m={lr_['m_used']:.0f} gap={lr_['gap']:.3f} "
                      f"lam_max={lr_['lam_max']:.3g} T_eff={opt.teff:.1f} N={opt.N:.4g} S={opt.S:.4g} "
                      f"mu={opt.mu:.3g} ratio={min(opt.S / max(opt.N, 1e-12), opt.cap):.3g} ({lr_['seconds']:.1f}s)")

        if cfg.grad_clip != 0.0:
            raw_model = distributed_backend.get_raw_model(model)
            grad_norm = distributed_backend.clip_grad_norm_(
                raw_model.parameters(), cfg.grad_clip
            )
            # Convert DTensor to float immediately; calling .item() on a DTensor
            # inside a rank-0-only block would deadlock FSDP (implicit collective).
            grad_norms.append(grad_norm.item())

        if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
            opt.train()
        (
            opt.step()
            if cfg.opt != "sophiag"
            else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
        )
        if cfg.scheduler != "none":
            scheduler.step()
        if cfg.opt == "sophiag":
            opt.zero_grad(set_to_none=True)
            if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                sample_again = model(x, targets=y, get_logits=True)
                samp_dist = torch.distributions.Categorical(
                    logits=sample_again["logits"]
                )
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(
                    sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                    y_sample.view(-1),
                    ignore_index=-1,
                )
                (loss_sampled / cfg.acc_steps).backward()
                opt.update_hessian()
                opt.zero_grad(set_to_none=True)
                model.zero_grad()
        elif cfg.opt == "mars":
            opt.zero_grad(set_to_none=True)
            opt.update_last_grad()
        else:
            opt.zero_grad(set_to_none=True)

        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        iterations_in_run += 1

        # Report peak GPU memory after first full iteration
        if curr_iter == 1 and distributed_backend.is_master_process():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            reserved_mem = torch.cuda.max_memory_reserved() / 1024**3
            print(f"[Memory] GPU 0: peak_allocated={peak_mem:.2f} GiB, peak_reserved={reserved_mem:.2f} GiB")

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            if cfg.opt == "prodigy":
                prodigy_efective_lrs = log_prodigy_lr(opt)
            
            # Log optimizer-specific schedules for AdEMAMix and DANA
            optimizer_schedule_logs = log_optimizer_schedules(opt, cfg.opt)

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )
            if cfg.opt == "prodigy":
                print(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

            if cfg.wandb:
                wandb_logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "iter_dt": dt,
                    "max_grad_norm": max(grad_norms) if grad_norms else 0,
                    "mean_grad_norm": (
                        sum(grad_norms) / len(grad_norms) if grad_norms else 0
                    ),
                    **train_aux_losses,
                }

                if cfg.opt == "prodigy":
                    wandb_logs["effective_lr"] = prodigy_efective_lrs[0]

                # Add optimizer schedule logs to wandb
                if optimizer_schedule_logs:
                    wandb_logs.update(optimizer_schedule_logs)

                if cfg.log_parameter_norms:
                    raw_model = distributed_backend.get_raw_model(model)
                    model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
                    wandb_logs["model_norm"] = model_norm

                wandb.log(wandb_logs, step=curr_iter)

            grad_norms = []

    # Wait for the last checkpoint copy, but not forever: an unfinished copy is abandoned and the older one kept
    if distributed_backend.is_master_process() and not wait_for_rsync(CKPT_FINAL_WAIT):
        print(f"[Checkpoint] WARNING: last copy still running after {CKPT_FINAL_WAIT:.0f}s; abandoned, older copy kept")

    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    # With FSDP, all ranks must participate in forward passes (parameter all-gather).
    # For DDP/single, only master rank evaluates (original behavior).
    _fsdp = _all_ranks_checkpoint(distributed_backend)

    if not _fsdp and not distributed_backend.is_master_process():
        return

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    # Only master logs results
    if distributed_backend.is_master_process():
        print(
            f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
            f"val_loss={val_loss:.3f} "
            f"val_pp={val_perplexity:.3f} "
            f"val_acc={val_acc:3f}"
        )

        if cfg.wandb:
            if curr_iter == cfg.iterations or full_eval:
                logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "final-val/loss": val_loss,
                    "final-val/perplexity": val_perplexity,
                    "final-val/acc": val_acc,
                    **val_aux_losses,
                }
            else:
                logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity,
                    "val/acc": val_acc,
                    **val_aux_losses,
                }
            if cfg.moe and cfg.plot_router_logits:
                routing_logs = visualize_routing(router_logits, cfg)
                logs = {**logs, **routing_logs}

            wandb.log(logs, step=curr_iter)
            if cfg.eval_seq_prefix != "none" and (
                curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
            ):
                text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                out_str = distributed_backend.get_raw_model(model).generate_from_string(
                    cfg.eval_seq_prefix,
                    max_new_tokens=40,
                    temperature=0.9,
                    top_k=None,
                )
                text_table.add_data(curr_iter, val_perplexity, out_str)
                # why a copy? see github.com/wandb/wandb/issues/2981
                wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)}, step=curr_iter)
    model.train()
