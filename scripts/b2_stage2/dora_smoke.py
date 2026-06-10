import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from b2_stage2_common import (
    BACKBONES,
    DATASETS,
    JsonlLogger,
    MemoryMonitor,
    REPO_ROOT,
    SequenceHeadModel,
    apply_data_scaler,
    build_lora_config,
    compute_data_scaler,
    count_parameters,
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
    parser = argparse.ArgumentParser(description="B2 Stage-2 Phase 3 DoRA 5-step smoke")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def out_dim(dataset_name: str) -> int:
    return 27 if dataset_name == "sider" else 1


def build_dora_model(args: argparse.Namespace):
    spec = BACKBONES[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    backbone = AutoModel.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.0,
        bias="none",
        use_dora=True,
    )
    backbone = get_peft_model(backbone, config)
    model = SequenceHeadModel(backbone, hidden_size=backbone.config.hidden_size, out_dim=out_dim(args.dataset))
    return tokenizer, model


def load_phase3_splits(dataset_name: str, seed: int):
    if dataset_name == "freesolv":
        splits = load_sprint_freesolv_split_if_available(seed)
        if splits is not None:
            return splits, "sprint_freesolv_seed0_indices"
    if dataset_name == "sider" and seed == 0:
        split_dir = os.path.join(REPO_ROOT, "dumped/baseline_sider_lora_chemberta2_seed0/split")
        paths = {name: os.path.join(split_dir, f"{name}_indices.txt") for name in ("train", "val", "test")}
        if all(os.path.exists(path) for path in paths.values()):
            loaded = {}
            for name, path in paths.items():
                with open(path, encoding="utf-8") as f:
                    loaded[name] = [int(line.strip()) for line in f if line.strip()]
            return loaded, "phase2_sider_seed0_indices"
    dataset = read_dataset(dataset_name)
    return split_indices(len(dataset.smiles), seed), "chemprop_random_seed"


def magnitude_items(model) -> List[Tuple[str, torch.nn.Parameter]]:
    items = []
    for name, param in model.named_parameters():
        if "lora_magnitude_vector" in name or "lora_m" in name or "magnitude" in name:
            items.append((name, param))
    return items


def digest_params(items: List[Tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, param in items:
        digest.update(name.encode("utf-8"))
        digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def count_state_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "lora_A": sum(1 for key in state_dict if "lora_A" in key),
        "lora_B": sum(1 for key in state_dict if "lora_B" in key),
        "lora_m": sum(1 for key in state_dict if "lora_m" in key),
        "lora_magnitude_vector": sum(1 for key in state_dict if "lora_magnitude_vector" in key),
        "magnitude_any": sum(1 for key in state_dict if "magnitude" in key or "lora_m" in key),
    }


def reload_smoke(args: argparse.Namespace, checkpoint_path: str, batch_smiles, batch_targets, data_scaler, task: str):
    tokenizer, model = build_dora_model(args)
    state = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    targets = apply_data_scaler(batch_targets, data_scaler)
    with torch.no_grad():
        input_ids, attention_mask, _ = make_batch(tokenizer, batch_smiles, targets, args.max_len, device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    return {
        "reload_forward_ok": True,
        "reload_output_shape": list(logits.shape),
        "reload_missing_keys": list(load_result.missing_keys),
        "reload_unexpected_keys": list(load_result.unexpected_keys),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(args.save_dir)
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        logger.log(f"phase3_dora_smoke_start backbone={args.backbone} dataset={args.dataset} seed={args.seed} steps={args.steps}")
        logger.log(f"config batch_size={args.batch_size} max_len={args.max_len} lr={args.lr} device={device}")
        with MemoryMonitor(interval=1.0) as monitor:
            dataset = read_dataset(args.dataset)
            splits, split_source = load_phase3_splits(args.dataset, args.seed)
            split_hashes = write_split_artifacts(args.save_dir, dataset, splits)
            logger.log(
                "split_counts "
                + " ".join(f"{name}={len(indices)}" for name, indices in splits.items())
                + f" source={split_source}"
            )
            train_smiles, train_targets_raw = subset(dataset, splits["train"])
            data_scaler = compute_data_scaler(train_targets_raw, dataset.task)
            train_targets = apply_data_scaler(train_targets_raw, data_scaler)
            batch_indices = list(range(min(args.batch_size, len(train_smiles))))
            batch_smiles = [train_smiles[i] for i in batch_indices]
            batch_targets = train_targets[np.array(batch_indices, dtype=np.int64)]
            batch_targets_raw = train_targets_raw[np.array(batch_indices, dtype=np.int64)]

            tokenizer, model = build_dora_model(args)
            model.to(device)
            model.train()
            total_params, trainable_params = count_parameters(model)
            magnitude_before_items = magnitude_items(model)
            magnitude_before = digest_params(magnitude_before_items)
            magnitude_param_count = len(magnitude_before_items)
            magnitude_param_numel = sum(param.numel() for _, param in magnitude_before_items)
            logger.log(
                f"model_params total={total_params} trainable={trainable_params} "
                f"magnitude_params={magnitude_param_count} magnitude_numel={magnitude_param_numel}"
            )

            input_ids, attention_mask, y = make_batch(tokenizer, batch_smiles, batch_targets, args.max_len, device)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            losses: List[float] = []
            forward_ok = False
            backward_ok = False
            optimizer_ok = False
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                forward_ok = True
                loss = masked_loss(logits, y, dataset.task)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non_finite_loss step={step} loss={loss}")
                loss.backward()
                backward_ok = True
                optimizer.step()
                optimizer_ok = True
                if device.type == "cuda":
                    torch.cuda.synchronize()
                value = float(loss.detach().cpu())
                losses.append(value)
                logger.verbose("step", {"step": step, "loss": value})

            magnitude_after_items = magnitude_items(model)
            magnitude_after = digest_params(magnitude_after_items)
            magnitude_updated = magnitude_before != magnitude_after
            checkpoint_path = os.path.join(args.save_dir, "model.pt")
            lora_config = build_lora_config(
                "dora",
                args.backbone,
                BACKBONES[args.backbone]["hf_path"],
                use_dora=True,
                stage="B2 Stage-2 Phase 3 DoRA smoke",
            )
            save_checkpoint(
                checkpoint_path,
                model,
                checkpoint_args(args, dataset.task, split_source),
                data_scaler,
                lora_config,
            )
            state = torch.load(checkpoint_path, map_location="cpu")
            key_counts = count_state_keys(state["state_dict"])
            reload_result = reload_smoke(args, checkpoint_path, batch_smiles, batch_targets_raw, data_scaler, dataset.task)
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch_peak_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0
            monotonic = all(losses[i + 1] <= losses[i] for i in range(len(losses) - 1))
            result = {
                "status": "PASS",
                "backbone": args.backbone,
                "dataset": args.dataset,
                "seed": args.seed,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "magnitude_param_count": magnitude_param_count,
                "magnitude_param_numel": magnitude_param_numel,
                "magnitude_digest_before": magnitude_before,
                "magnitude_digest_after": magnitude_after,
                "magnitude_updated": magnitude_updated,
                "losses": losses,
                "loss_decrease_ok": losses[-1] < losses[0],
                "loss_monotonic_nonincreasing": monotonic,
                "forward_ok": forward_ok,
                "backward_ok": backward_ok,
                "optimizer_ok": optimizer_ok,
                "state_key_counts": key_counts,
                "checkpoint_keys": sorted(state.keys()),
                "checkpoint_lora_config": state["lora_config"],
                "checkpoint_path": checkpoint_path,
                "reload": reload_result,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "split_hashes": split_hashes,
                "split_source": split_source,
                "nvidia_smi_peak_mib": monitor.peak_mib,
                "torch_peak_allocated_mib": torch_peak_mib,
                "elapsed_sec": time.time() - start,
                "timestamp_end": timestamp(),
            }
            if not all([forward_ok, backward_ok, optimizer_ok, magnitude_updated, result["loss_decrease_ok"], monotonic]):
                result["status"] = "FAIL"
            if key_counts["magnitude_any"] != magnitude_param_count:
                result["status"] = "FAIL"
            if not reload_result["reload_forward_ok"] or reload_result["reload_missing_keys"] or reload_result["reload_unexpected_keys"]:
                result["status"] = "FAIL"
            metrics_path = os.path.join(args.save_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            logger.log("phase3_dora_smoke_result " + json.dumps(result, sort_keys=True))
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
            if result["status"] != "PASS":
                raise RuntimeError("DoRA smoke failed checks; see RESULT_JSON")
    finally:
        logger.close()


def checkpoint_args(args: argparse.Namespace, task: str, split_source: str) -> Dict[str, object]:
    spec = BACKBONES[args.backbone]
    return {
        "backbone": args.backbone,
        "hf_path": spec["hf_path"],
        "trust_remote_code": spec["trust_remote_code"],
        "hf_model_revision": spec["model_revision"],
        "remote_code_revision": spec["remote_code_revision"],
        "method": "dora",
        "dataset": args.dataset,
        "dataset_type": task,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "split_source": split_source,
        "stage": "B2 Stage-2 Phase 3 DoRA smoke",
    }


if __name__ == "__main__":
    main()
