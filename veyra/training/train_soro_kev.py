"""System-One Soro-Kev Decision Model Training Loop.

Trains dual linear pointer projection heads + LoRA adapter on a causal backbone chosen with
--base-model (default Qwen/Qwen3-0.6B). The released checkpoints use Qwen/Qwen3.5-0.8B-Base
(soro-kev-0.8b) and Qwen/Qwen3-4B (soro-kev-4b), per their adapter_config.json. Training uses a weighted multi-task objective: route CE, Brier scoring, permutation consistency, and
auxiliary decision heads, followed by periodic split-conformal validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soro.routing.conformal import conformal_quantile, _wilson
from soro.routing.kev_model import (
    MockCausalBackbone,
    SoroKevModel,
    compute_rlcd_loss,
    save_soro_kev_checkpoint,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "soro_kev_dataset"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "soro_kev_run"


class RLCDLoss(nn.Module):
    """Legacy-named multi-task routing objective used by the training step.

    The batched forward pass reduces each term. This module is the single
    weighted sum: CE + 0.25 Brier + 0.30 permutation JSD + auxiliary CEs.
    The name is retained for checkpoint/import compatibility; it is not the
    RLCD alignment method from Yang et al. (ICLR 2024).
    """

    def forward(
        self,
        route_ce: torch.Tensor,
        brier: torch.Tensor,
        jsd: torch.Tensor,
        arg_ce: torch.Tensor,
        missing_ce: torch.Tensor,
        mutation_ce: torch.Tensor,
        intent_ce: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_rlcd_loss(route_ce, brier, jsd, arg_ce, missing_ce, mutation_ce, intent_ce)


def _legacy_example_rlcd(
    route_logits: torch.Tensor,
    route_gold_idx: int,
    perm_logits: torch.Tensor,
    perm_to_route_perm: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single-example RLCD. Kept so a unit check can run the formula without a batch."""
    p_route = F.softmax(route_logits, dim=-1)
    ce_loss = -torch.log(torch.clamp(p_route[route_gold_idx], min=1e-9))
    num_options = route_logits.shape[-1]
    gold_one_hot = torch.zeros(num_options, device=route_logits.device)
    gold_one_hot[route_gold_idx] = 1.0
    brier_loss = torch.mean((p_route - gold_one_hot) ** 2)
    p_perm_aligned = F.softmax(perm_logits, dim=-1)[perm_to_route_perm]
    p_safe = torch.clamp(p_route, min=1e-9)
    q_safe = torch.clamp(p_perm_aligned, min=1e-9)
    m = 0.5 * (p_safe + q_safe)
    log_m = torch.log(m)
    kl1 = torch.sum(p_safe * (torch.log(p_safe) - log_m))
    kl2 = torch.sum(q_safe * (torch.log(q_safe) - log_m))
    jsd_loss = torch.clamp(0.5 * (kl1 + kl2), min=0.0)
    zero = route_logits.sum() * 0.0
    total_loss = compute_rlcd_loss(ce_loss, brier_loss, jsd_loss, zero, zero, zero)
    is_hit = (torch.argmax(route_logits).item() == route_gold_idx)
    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_ce": float(ce_loss.item()),
        "loss_brier": float(brier_loss.item()),
        "loss_jsd": float(jsd_loss.item()),
        "prob_gold": float(p_route[route_gold_idx].item()),
        "hit": 1.0 if is_hit else 0.0,
    }
    return total_loss, metrics


def stratified_val_sample(records: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Even sample across format and delex so a 400-row mid-epoch eval matches the file."""
    if len(records) <= n:
        return list(records)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec.get("format"), bool(rec.get("delexicalized")))
        groups.setdefault(key, []).append(rec)
    rng = random.Random(seed)
    for bucket in groups.values():
        rng.shuffle(bucket)
    picked: list[dict[str, Any]] = []
    order = list(groups)
    while len(picked) < n and any(groups.values()):
        for key in order:
            bucket = groups[key]
            if bucket and len(picked) < n:
                picked.append(bucket.pop())
    return picked


def run_split_conformal_val(
    model: SoroKevModel,
    val_records: list[dict[str, Any]],
    cal_size: int = 300,
    alpha: float = 0.10,
    batch_size: int = 4,
    max_seq_len: int = 1024,
) -> dict[str, Any]:
    """Run split conformal validation on held-out zero-shot records using batched forward passes."""
    model.eval()
    cal_records = val_records[:cal_size]
    test_records = val_records[cal_size:]

    # Calibration Phase: compute non-conformity scores
    nonconf_scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(cal_records), batch_size):
            b_items = cal_records[i : i + batch_size]
            out = model.forward_batch(b_items, max_seq_len=max_seq_len)
            if out.get("skipped") or not out["logits_route_batch"]:
                continue
            for j, rec_i in enumerate(out.get("kept_indices") or []):
                r = b_items[rec_i]
                logits = out["logits_route_batch"][j]
                q_route = r["questions"]["tool_route"]
                keys = list(q_route["criteria"].keys())
                gold = q_route["label"]
                probs = F.softmax(logits, dim=-1).cpu().tolist()
                p_gold = probs[keys.index(gold)] if gold in keys else 0.0
                nonconf_scores.append(1.0 - p_gold)

    q_hat = conformal_quantile(np.array(nonconf_scores, dtype=np.float64), alpha=alpha)
    threshold = max(0.0, 1.0 - q_hat)

    # Test Phase: evaluate empirical coverage and set size
    correct = 0
    covered = 0
    set_sizes: list[int] = []

    with torch.no_grad():
        for i in range(0, len(test_records), batch_size):
            b_items = test_records[i : i + batch_size]
            out = model.forward_batch(b_items, max_seq_len=max_seq_len)
            if out.get("skipped") or not out["logits_route_batch"]:
                continue
            for j, rec_i in enumerate(out.get("kept_indices") or []):
                r = b_items[rec_i]
                logits = out["logits_route_batch"][j]
                q_route = r["questions"]["tool_route"]
                keys = list(q_route["criteria"].keys())
                gold = q_route["label"]
                probs = F.softmax(logits, dim=-1).cpu().tolist()
                pred_key = keys[int(torch.argmax(logits).item())]
                if pred_key == gold:
                    correct += 1

                p_set = [keys[k] for k, p in enumerate(probs) if p >= threshold]
                sz = len(p_set)
                set_sizes.append(sz)
                if gold in p_set:
                    covered += 1

    n_scored = len(set_sizes)
    lo, hi = _wilson(covered, n_scored) if n_scored else (0.0, 0.0)
    singletons = sum(1 for s in set_sizes if s == 1)
    q_value = float(q_hat) if n_scored and np.isfinite(q_hat) else float("nan")

    model.train()
    return {
        "val_route_acc": correct / n_scored if n_scored else 0.0,
        "empirical_coverage": covered / n_scored if n_scored else 0.0,
        "wilson_ci": [lo, hi],
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
        "singleton_rate": singletons / n_scored if n_scored else 0.0,
        "q_hat": q_value,
        "threshold": float(threshold) if n_scored else 0.0,
        "n_scored": n_scored,
        "n_skipped": len(test_records) - n_scored,
    }


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train Soro-Kev System-One Decision Model")
    parser.add_argument("--profile", type=str, default="cpu_smoke", choices=["cpu_smoke", "kaggle_t4"])
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B", help="Base backbone (e.g. Qwen/Qwen3-0.6B, Qwen/Qwen3.5-0.8B-Base, Qwen/Qwen3.5-4B-Base)")
    parser.add_argument("--batch-size", type=int, default=4, help="GPU mini-batch size (number of packed records per step)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    args = parser.parse_args()

    print(f"=== Initializing Soro-Kev Training (Profile: {args.profile} | Model: {args.base_model} | Batch Size: {args.batch_size}) ===")

    out_path = Path(args.output_dir)
    data_path = Path(args.data_dir)
    train_file = data_path / "train.jsonl"
    val_file = data_path / "val_zeroshot.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Dataset not found at {train_file}. Run build_universal_dataset.py first.")

    train_records = []
    with train_file.open(encoding="utf-8") as f:
        for line in f:
            train_records.append(json.loads(line))
    extra_file = data_path / "external_intent.jsonl"
    if extra_file.exists():
        extra_n = 0
        with extra_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
                    extra_n += 1
        print(f"Added {extra_n} external intent rows from {extra_file.name}")

    val_records = []
    if val_file.exists():
        with val_file.open(encoding="utf-8") as f:
            for line in f:
                val_records.append(json.loads(line))

    is_smoke = (args.profile == "cpu_smoke")
    # T4 is Turing sm_75: 16 GB, fp16 tensor cores, no bf16.
    # 2x T4 is launched with torchrun --nproc_per_node=2. Each rank holds the
    # full 0.6B LoRA model. A 4B backbone does not fit this profile.
    load_in_4bit = "4b" in args.base_model.lower()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    dist = None
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Qwen3-0.6B fp16 fits batch 4 on a 16 GB T4.
    # Qwen3.5-0.8B does not: each record is two sequences, and its linear-attention
    # fallback fills the card at batch 4. One record per GPU, accumulate to the
    # same effective batch of 16 across 2 GPUs.
    batch_size = 2 if is_smoke else args.batch_size
    max_seq_len = 1536 if args.profile == "kaggle_t4" else 1024
    if args.profile == "kaggle_t4" and "3.5" in args.base_model:
        if batch_size > 1 and local_rank == 0:
            print(
                f"Qwen3.5 on a T4: lowering batch {batch_size} -> 1 and "
                f"grad_accum {args.grad_accum} -> {max(args.grad_accum, 8)} so the linear-attention forward fits."
            )
        batch_size = 1
        args.grad_accum = max(args.grad_accum, 8)
    qwen35_4b = args.profile == "kaggle_t4" and "4b" in args.base_model.lower() and "3.5" in args.base_model
    plain_4b = args.profile == "kaggle_t4" and "4b" in args.base_model.lower() and "3.5" not in args.base_model
    if qwen35_4b:
        max_seq_len = 256
        if local_rank == 0:
            print("Qwen3.5-4B on a T4: max_seq_len=256 and one sequence per step.")
    if plain_4b:
        max_seq_len = 512
        batch_size = 1
        args.grad_accum = max(args.grad_accum, 8)
        if args.lr > 5e-5:
            args.lr = 5e-5
        if local_rank == 0:
            print("Qwen3-4B on a T4: max_seq_len=512, one sequence, batch 1, lr 5e-5, no GradScaler.")
    if is_smoke:
        train_records = train_records[:20]
        val_records = val_records[:40]
    if distributed:
        train_records = train_records[local_rank::world_size]
        val_records = val_records if local_rank == 0 else []

    micro_per_epoch = max(1, math.ceil(len(train_records) / batch_size))
    # Optimizer steps, not micro-batches. grad_accum micro-steps make one update.
    total_updates = max(1, micro_per_epoch // args.grad_accum) * (1 if is_smoke else args.epochs)
    if is_smoke:
        total_updates = 1
    total_steps = total_updates * args.grad_accum

    device = f"cuda:{local_rank}" if torch.cuda.is_available() and not is_smoke else "cpu"
    use_fp16 = args.profile == "kaggle_t4" and device.startswith("cuda")
    if use_fp16:
        major, _minor = torch.cuda.get_device_capability(local_rank)
        if major >= 8 and local_rank == 0:
            print("This GPU can run bf16. Profile kaggle_t4 stays on fp16 to match T4 tensor cores.")
    print(
        f"Loaded {len(train_records)} train records | {len(val_records)} zero-shot val records | "
        f"Batch/GPU: {batch_size} | Grad accum: {args.grad_accum} | World: {world_size} | Device: {device} | "
        f"dtype: {'fp16' if use_fp16 else 'fp32'}"
    )

    model = SoroKevModel(
        model_name=args.base_model,
        use_mock=is_smoke,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        allow_mock_fallback=is_smoke,
        load_in_4bit=load_in_4bit and not is_smoke,
    )
    if load_in_4bit:
        # The quantized backbone is already on this GPU. The pointer heads are
        # created on CPU. DDP rejects a mix of the two.
        for child in (model.q_proj_head, model.opt_proj_head, model.binary_head, model.intent_head):
            child.to(device)
        model.log_temp.data = model.log_temp.data.to(device)
        model.single_sequence = True
    else:
        model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    raw_model = model.module if distributed else model
    criterion = RLCDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 4-bit weights are already fp16 inside the matmul. A GradScaler on top of that overflows to NaN.
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16 and not load_in_4bit)

    warmup_steps = 1 if is_smoke else 100
    best_val_score = -1.0
    best_q_hat = 0.90
    step_idx = 0
    t0 = time.time()

    running_loss = 0.0
    running_route = 0.0
    running_brier = 0.0
    running_jsd = 0.0
    running_hits = 0.0
    logged_steps = 0

    print(f"Starting batched training loop ({total_steps} steps | Profile: {args.profile})...")

    optimizer.zero_grad()

    while step_idx < total_steps:
        # Current learning rate schedule
        cur_lr = get_cosine_lr(step_idx, total_steps, warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Mini-batch slice
        batch_start = (step_idx * batch_size) % len(train_records)
        batch_items = [train_records[(batch_start + j) % len(train_records)] for j in range(batch_size)]

        # Batched forward pass: packs 2 sequences per record, single backbone execution
        if use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = raw_model.forward_batch(batch_items, max_seq_len=max_seq_len)
        else:
            out = raw_model.forward_batch(batch_items, max_seq_len=max_seq_len)
        skipped = bool(out.get("skipped"))
        if skipped:
            # Keep the autograd graph so this rank still steps. The eval check
            # below must still run, or this rank walks past step 400 and the
            # other rank waits at the barrier until NCCL kills the job.
            loss = sum((p.sum() * 0.0) for p in model.parameters())
        else:
            loss = criterion(*out["parts"])
        scaled_loss = loss / args.grad_accum
        scaler.scale(scaled_loss).backward()

        if not skipped:
            running_loss += float(out["loss"].item())
            running_route += out["loss_route"]
            running_brier += out["loss_brier"]
            running_jsd += out["loss_jsd"]
            running_hits += out["metrics"]["route_acc"]
            logged_steps += 1

        grad_norm = 0.0
        if (step_idx + 1) % args.grad_accum == 0 or is_smoke:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            if math.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                print(f"Step {step_idx + 1}: non-finite grad {grad_norm}. Skipping update.", flush=True)
            scaler.update()
            optimizer.zero_grad()

        step_idx += 1

        # Periodic logging every 50 steps (or on smoke steps)
        if logged_steps >= 50 or is_smoke:
            n_sub = max(logged_steps, 1)
            speed = (n_sub * batch_size) / max(time.time() - t0, 0.001)
            print(
                f"Step {step_idx:05d}/{total_steps} | "
                f"Loss: {running_loss / n_sub:.4f} (Route: {running_route / n_sub:.4f}, Brier: {running_brier / n_sub:.4f}, JSD: {running_jsd / n_sub:.6f}) | "
                f"Train Route Acc: {min(100.0, running_hits / n_sub * 100):.1f}% | "
                f"Grad Norm: {grad_norm:.2f} | "
                f"Throughput: {speed:.1f} rec/s",
                flush=True,
            )
            running_loss = running_route = running_brier = running_jsd = running_hits = 0.0
            logged_steps = 0
            t0 = time.time()

        # Mid-epoch eval is a 400-row stratified slice. Full val runs at epoch boundaries.
        # Rank 0 holds the val file. The other rank waits, or the next training
        # allreduce starts while rank 0 is still scoring and the job hangs.
        steps_per_epoch = max(1, math.ceil(len(train_records) / batch_size))
        at_epoch_end = step_idx % steps_per_epoch == 0 or step_idx == total_steps
        should_eval = (step_idx % args.eval_every == 0) or at_epoch_end
        if should_eval and dist is not None:
            dist.barrier()
        if should_eval and local_rank == 0 and len(val_records) >= 10:
            if at_epoch_end:
                eval_records = val_records
                eval_label = "full"
            else:
                eval_records = stratified_val_sample(val_records, 80, seed=step_idx)
                eval_label = "subsample-80"
            print(f"--- Split conformal val ({eval_label}, n={len(eval_records)}, step {step_idx}) ---")
            val_res = run_split_conformal_val(
                raw_model,
                eval_records,
                cal_size=min(150 if not at_epoch_end else 300, max(1, len(eval_records) // 2)),
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            print(
                f"Zero-Shot Route Acc: {val_res['val_route_acc'] * 100:.2f}% | "
                f"Coverage: {val_res['empirical_coverage'] * 100:.2f}% (Wilson: [{val_res['wilson_ci'][0]*100:.1f}%, {val_res['wilson_ci'][1]*100:.1f}%]) | "
                f"Mean |S|: {val_res['mean_set_size']:.2f} | "
                f"Singleton Rate: {val_res['singleton_rate'] * 100:.2f}% | "
                f"q_hat: {val_res['q_hat']:.4f} | scored {val_res['n_scored']} skipped {val_res['n_skipped']}",
                flush=True,
            )
            score = val_res["val_route_acc"] + val_res["singleton_rate"]
            if val_res["n_scored"] < 20 or not math.isfinite(val_res["q_hat"]):
                print("--> Validation scored too few rows. Not saving.", flush=True)
            elif score > best_val_score:
                best_val_score = score
                best_q_hat = val_res["q_hat"]
                print(f"--> New best model found! Saving checkpoint to {out_path}...")
                if local_rank == 0:
                    save_soro_kev_checkpoint(raw_model, out_path, q_hat=best_q_hat)
        if should_eval and dist is not None:
            dist.barrier()

    if local_rank == 0:
        print(f"=== Training Complete! Saving final checkpoint to {out_path} ===")
        save_soro_kev_checkpoint(raw_model, out_path, q_hat=best_q_hat)
    print("Checkpoint saved successfully: head.pt + config.json + temperature.json.")


if __name__ == "__main__":
    main()
