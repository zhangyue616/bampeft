import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from b2_phase4_common import (
    BACKBONES,
    DATASETS,
    PHASE4_BATCH_SIZE,
    PHASE4_EPOCHS,
    PHASE4_LR,
    PHASE4_PATIENCE,
    JsonlLogger,
    MemoryMonitor,
    SequenceHeadModel,
    apply_data_scaler,
    batch_ranges,
    build_lora_config,
    compute_data_scaler,
    compute_metric,
    count_parameters,
    environment_provenance,
    evaluate_loss_and_predictions,
    make_batch,
    masked_loss,
    read_dataset,
    save_checkpoint,
    split_indices,
    subset,
    summarize_times,
    timestamp,
    write_json,
    write_split_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2 Stage-2 Phase 4 production training run")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--method", choices=["head", "lora", "dora"], required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=PHASE4_EPOCHS)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=PHASE4_BATCH_SIZE)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--lr", type=float, default=PHASE4_LR)
    parser.add_argument("--patience", type=int, default=PHASE4_PATIENCE)
    parser.add_argument("--run_index", type=int, default=-1)
    parser.add_argument("--total_runs", type=int, default=108)
    return parser.parse_args()


def out_dim(dataset_name: str) -> int:
    if dataset_name == "sider":
        return 27
    return 1


def build_model(args: argparse.Namespace):
    spec = BACKBONES[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    backbone = AutoModel.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    if args.method == "head":
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.0,
            bias="none",
            use_dora=args.method == "dora",
        )
        backbone = get_peft_model(backbone, peft_config)
    hidden_size = backbone.config.hidden_size
    return tokenizer, SequenceHeadModel(backbone, hidden_size=hidden_size, out_dim=out_dim(args.dataset))


def checkpoint_args(args: argparse.Namespace, task: str, metric: str) -> Dict[str, object]:
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
        "metric": metric,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "patience": args.patience,
        "run_index": args.run_index,
        "total_runs": args.total_runs,
        "stage": "B2 Stage-2 Phase 4 production",
    }


def trainable_param_hash(model: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                digest.update(name.encode("utf-8"))
                digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_best_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])


def main() -> None:
    args = parse_args()
    if args.max_len is None:
        args.max_len = BACKBONES[args.backbone]["native_max_len"]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(save_dir)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        logger.log(
            f"phase4_run_start run_index={args.run_index} backbone={args.backbone} "
            f"method={args.method} dataset={args.dataset} seed={args.seed}"
        )
        logger.log(
            f"config epochs={args.epochs} patience={args.patience} batch_size={args.batch_size} "
            f"max_len={args.max_len} lr={args.lr} device={device}"
        )
        provenance = environment_provenance(args.backbone)
        logger.log("provenance " + json.dumps(provenance, sort_keys=True))

        with MemoryMonitor(interval=1.0) as monitor:
            dataset = read_dataset(args.dataset)
            splits = split_indices(len(dataset.smiles), args.seed)
            split_hashes = write_split_artifacts(save_dir, dataset, splits)
            logger.log(
                "split_counts "
                + " ".join(f"{name}={len(indices)}" for name, indices in splits.items())
                + " source=chemprop_random_seed"
            )
            logger.log("split_hashes " + json.dumps(split_hashes, sort_keys=True))
            logger.verbose("split", {"counts": {k: len(v) for k, v in splits.items()}, "hashes": split_hashes})

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
            before_hash = trainable_param_hash(model)
            logger.log(f"model_params total={total_params} trainable={trainable_params}")
            logger.verbose("model", {"total_params": total_params, "trainable_params": trainable_params})

            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            best_val = float("inf")
            best_epoch = -1
            bad_epochs = 0
            early_stop_triggered = False
            early_stop_epoch = None
            val_curve: List[float] = []
            train_loss_curve: List[float] = []
            epoch_records: List[Dict[str, float]] = []
            batch_times: List[float] = []
            model_path = save_dir / "model.pt"

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
                    batch_elapsed = time.time() - batch_start
                    batch_times.append(batch_elapsed)
                    epoch_losses.append(float(loss.detach().cpu()))
                    logger.verbose(
                        "batch",
                        {"epoch": epoch, "batch": batch_id, "loss": epoch_losses[-1], "elapsed_sec": batch_elapsed},
                    )

                epoch_elapsed = time.time() - epoch_start
                train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                val_loss, val_pred = evaluate_loss_and_predictions(
                    model,
                    tokenizer,
                    val_smiles,
                    val_targets,
                    dataset.task,
                    args.batch_size,
                    args.max_len,
                    device,
                )
                val_metric, val_metric_detail = compute_metric(dataset, val_pred, val_targets_raw, data_scaler)
                train_loss_curve.append(train_loss)
                val_curve.append(val_loss)
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metric": val_metric,
                    "elapsed_sec": epoch_elapsed,
                }
                epoch_records.append(record)
                logger.log(
                    f"epoch={epoch} train_loss={train_loss:.8f} val_loss={val_loss:.8f} "
                    f"val_metric={val_metric:.8f} elapsed_sec={epoch_elapsed:.4f}"
                )
                logger.verbose("epoch", {**record, "val_metric_detail": val_metric_detail})

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    save_checkpoint(
                        model_path,
                        model,
                        checkpoint_args(args, dataset.task, dataset.metric),
                        data_scaler,
                        build_lora_config(args.method, args.backbone),
                        provenance,
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= args.patience:
                        early_stop_triggered = True
                        early_stop_epoch = epoch
                        logger.log(f"early_stop_triggered epoch={epoch} patience={args.patience}")
                        break

            restore_best_checkpoint(model, model_path, device)
            test_loss, test_pred = evaluate_loss_and_predictions(
                model,
                tokenizer,
                test_smiles,
                test_targets,
                dataset.task,
                args.batch_size,
                args.max_len,
                device,
            )
            final_metric, final_metric_detail = compute_metric(dataset, test_pred, test_targets_raw, data_scaler)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_total = time.time() - start_time
            torch_peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0
            after_hash = trainable_param_hash(model)
            result: Dict[str, object] = {
                "status": "PASS",
                "phase": "B2 Stage-2 Phase 4 production",
                "run_index": args.run_index,
                "total_runs": args.total_runs,
                "backbone": args.backbone,
                "method": args.method,
                "dataset": args.dataset,
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "epochs_completed": len(val_curve),
                "early_stop_triggered": early_stop_triggered,
                "early_stop_epoch": early_stop_epoch,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "metric_type": dataset.metric,
                "final_metric": final_metric,
                "final_metric_detail": final_metric_detail,
                "test_loss": test_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "split_hashes": split_hashes,
                "split_source": "chemprop_random_seed",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "train_loss_curve": train_loss_curve,
                "val_loss_curve": val_curve,
                "epoch_records": epoch_records,
                "per_batch": summarize_times(batch_times),
                "elapsed_sec": elapsed_total,
                "nvidia_smi_peak_mib": monitor.peak_mib,
                "torch_peak_allocated_mib": torch_peak_mib,
                "trainable_hash_before": before_hash,
                "trainable_hash_after": after_hash,
                "model_path": str(model_path),
                "training_log": str(save_dir / "training.log"),
                "verbose_log": str(save_dir / "verbose.log"),
                "provenance": provenance,
                "timestamp_end": timestamp(),
            }
            metrics_path = save_dir / "metrics.json"
            write_json(metrics_path, result)
            summary = {
                "status": result["status"],
                "backbone": result["backbone"],
                "method": result["method"],
                "dataset": result["dataset"],
                "seed": result["seed"],
                "epochs_completed": result["epochs_completed"],
                "early_stop_epoch": result["early_stop_epoch"],
                "final_metric": result["final_metric"],
                "metric_type": result["metric_type"],
                "elapsed_sec": result["elapsed_sec"],
                "nvidia_smi_peak_mib": result["nvidia_smi_peak_mib"],
            }
            logger.log("phase4_run_result " + json.dumps(summary, sort_keys=True))
            print("RESULT_JSON_PATH=" + str(metrics_path), flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
