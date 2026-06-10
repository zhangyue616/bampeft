import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from b2_stage2_common import (
    BACKBONES,
    DATASETS,
    JsonlLogger,
    MemoryMonitor,
    SequenceHeadModel,
    apply_data_scaler,
    batch_ranges,
    build_lora_config,
    compute_data_scaler,
    count_parameters,
    evaluate_loss,
    load_sprint_freesolv_split_if_available,
    make_batch,
    masked_loss,
    read_dataset,
    save_checkpoint,
    split_indices,
    subset,
    timestamp,
    write_split_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2 Stage-2 Phase 2 one-epoch LoRA timing pilot")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--method", choices=["lora"], required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    spec = BACKBONES[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    backbone = AutoModel.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.0,
        bias="none",
    )
    backbone = get_peft_model(backbone, peft_config)
    return tokenizer, SequenceHeadModel(backbone, hidden_size=backbone.config.hidden_size, out_dim=out_dim(args.dataset))


def out_dim(dataset_name: str) -> int:
    if dataset_name == "sider":
        return 27
    return 1


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(args.save_dir)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        logger.log(f"phase2_run_start backbone={args.backbone} method={args.method} dataset={args.dataset} seed={args.seed}")
        logger.log(f"config epochs={args.epochs} batch_size={args.batch_size} max_len={args.max_len} lr={args.lr} device={device}")

        with MemoryMonitor(interval=1.0) as monitor:
            dataset = read_dataset(args.dataset)
            splits = None
            split_source = "chemprop_random_seed"
            if args.dataset == "freesolv":
                splits = load_sprint_freesolv_split_if_available(args.seed)
                if splits is not None:
                    split_source = "sprint_freesolv_seed0_indices"
            if splits is None:
                splits = split_indices(len(dataset.smiles), args.seed)
            split_hashes = write_split_artifacts(args.save_dir, dataset, splits)
            logger.log(
                "split_counts "
                + " ".join(f"{name}={len(indices)}" for name, indices in splits.items())
                + f" source={split_source}"
            )
            logger.log("split_hashes " + json.dumps(split_hashes, sort_keys=True))
            logger.verbose("split", {"counts": {k: len(v) for k, v in splits.items()}, "hashes": split_hashes, "source": split_source})

            train_smiles, train_targets_raw = subset(dataset, splits["train"])
            val_smiles, val_targets_raw = subset(dataset, splits["val"])
            test_smiles, test_targets_raw = subset(dataset, splits["test"])

            data_scaler = compute_data_scaler(train_targets_raw, dataset.task)
            train_targets = apply_data_scaler(train_targets_raw, data_scaler)
            val_targets = apply_data_scaler(val_targets_raw, data_scaler)
            test_targets = apply_data_scaler(test_targets_raw, data_scaler)

            tokenizer, model = build_model(args)
            model.to(device)
            model.train()
            total_params, trainable_params = count_parameters(model)
            logger.log(f"model_params total={total_params} trainable={trainable_params}")
            logger.verbose("model", {"total_params": total_params, "trainable_params": trainable_params})

            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            best_val = float("inf")
            best_epoch = -1
            bad_epochs = 0
            early_stop_triggered = False
            val_curve: List[float] = []
            batch_times: List[float] = []
            train_loss_curve: List[float] = []

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for epoch in range(args.epochs):
                epoch_losses: List[float] = []
                epoch_start = time.time()
                for batch_id, batch in enumerate(batch_ranges(len(train_smiles), args.batch_size, shuffle_seed=args.seed + epoch)):
                    batch_start = time.time()
                    batch_smiles = [train_smiles[i] for i in batch]
                    batch_targets = train_targets[np.array(batch, dtype=np.int64)]
                    input_ids, attention_mask, y = make_batch(tokenizer, batch_smiles, batch_targets, args.max_len, device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = masked_loss(logits, y, dataset.task)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non_finite_loss epoch={epoch} batch={batch_id} loss={loss}")
                    loss.backward()
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.time() - batch_start
                    batch_times.append(elapsed)
                    epoch_losses.append(float(loss.detach().cpu()))
                    logger.verbose(
                        "batch",
                        {"epoch": epoch, "batch": batch_id, "loss": epoch_losses[-1], "elapsed_sec": elapsed},
                    )

                epoch_elapsed = time.time() - epoch_start
                train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                val_loss = evaluate_loss(
                    model,
                    tokenizer,
                    val_smiles,
                    val_targets,
                    dataset.task,
                    args.batch_size,
                    args.max_len,
                    device,
                )
                train_loss_curve.append(train_loss)
                val_curve.append(val_loss)
                logger.log(f"epoch={epoch} train_loss={train_loss:.8f} val_loss={val_loss:.8f} elapsed_sec={epoch_elapsed:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    save_checkpoint(
                        os.path.join(args.save_dir, "model.pt"),
                        model,
                        checkpoint_args(args, dataset.task, split_source),
                        data_scaler,
                        build_lora_config(args.method, args.backbone, BACKBONES[args.backbone]["hf_path"]),
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= args.patience:
                        early_stop_triggered = True
                        logger.log(f"early_stop_triggered epoch={epoch} patience={args.patience}")
                        break

            test_loss = evaluate_loss(
                model,
                tokenizer,
                test_smiles,
                test_targets,
                dataset.task,
                args.batch_size,
                args.max_len,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_total = time.time() - start_time
            torch_peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0
            timing = summarize_times(batch_times)
            result: Dict[str, object] = {
                "status": "PASS",
                "backbone": args.backbone,
                "method": args.method,
                "dataset": args.dataset,
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "epochs_completed": len(val_curve),
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "split_hashes": split_hashes,
                "split_source": split_source,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "train_loss_curve": train_loss_curve,
                "val_loss_curve": val_curve,
                "test_loss": test_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "early_stop_triggered": early_stop_triggered,
                "per_batch": timing,
                "elapsed_sec": elapsed_total,
                "nvidia_smi_peak_mib": monitor.peak_mib,
                "torch_peak_allocated_mib": torch_peak_mib,
                "model_path": os.path.join(args.save_dir, "model.pt"),
                "training_log": os.path.join(args.save_dir, "training.log"),
                "verbose_log": os.path.join(args.save_dir, "verbose.log"),
                "timestamp_end": timestamp(),
            }
            metrics_path = os.path.join(args.save_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            logger.log("phase2_run_result " + json.dumps(result, sort_keys=True))
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        logger.close()


def summarize_times(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def checkpoint_args(args: argparse.Namespace, task: str, split_source: str) -> Dict[str, object]:
    spec = BACKBONES[args.backbone]
    return {
        "backbone": args.backbone,
        "hf_path": spec["hf_path"],
        "trust_remote_code": spec["trust_remote_code"],
        "hf_model_revision": spec["model_revision"],
        "remote_code_revision": spec["remote_code_revision"],
        "method": args.method,
        "dataset": args.dataset,
        "dataset_type": task,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "split_source": split_source,
        "stage": "B2 Stage-2 Phase 2 timing pilot",
    }


if __name__ == "__main__":
    main()
