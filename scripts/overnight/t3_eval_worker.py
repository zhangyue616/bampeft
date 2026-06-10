#!/usr/bin/env python3
"""Evaluate one classification checkpoint and emit calibration statistics."""

from __future__ import annotations
import os

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "overnight"))

from overnight_common import DATASET_INFO, N5_SPLIT_DIR, ece_brier, read_indices, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def random_split_indices(n_items: int, seed: int) -> Dict[str, List[int]]:
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    train_size = int(0.8 * n_items)
    train_val_size = int(0.9 * n_items)
    return {"train": indices[:train_size], "val": indices[train_size:train_val_size], "test": indices[train_val_size:]}


def load_scaffold_test_indices(dataset: str, seed: int) -> List[int]:
    path = N5_SPLIT_DIR / f"{dataset}_scaffold_seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(i) for i in payload["splits"]["test"]]


def checkpoint_split_test_indices(checkpoint: Path, dataset: str, protocol: str, seed: int, n_items: int) -> List[int]:
    split_dir = checkpoint.parent / "split"
    test_path = split_dir / "test_indices.txt"
    if test_path.exists():
        return read_indices(test_path)
    if protocol == "scaffold":
        return load_scaffold_test_indices(dataset, seed)
    return random_split_indices(n_items, seed)["test"]


def eval_cmpnn(args: argparse.Namespace) -> Dict[str, object]:
    from chemprop.data import MoleculeDataset
    from chemprop.data.utils import get_data, split_data
    from chemprop.train.predict import predict
    from chemprop.utils import load_checkpoint

    path = Path(args.checkpoint)
    model = load_checkpoint(str(path), cuda=False)
    model.eval()
    data = get_data(DATASET_INFO[args.dataset]["path"], skip_invalid_smiles=True)
    if args.protocol == "scaffold":
        test_indices = load_scaffold_test_indices(args.dataset, args.seed)
        test_data = MoleculeDataset([data[i] for i in test_indices])
    else:
        _, _, test_data = split_data(data, split_type="random", sizes=(0.8, 0.1, 0.1), seed=args.seed)
        test_indices = list(range(len(test_data)))
    probs = np.asarray(predict(model, test_data, batch_size=50, scaler=None), dtype=float)
    targets = np.asarray(test_data.targets(), dtype=float)
    ece, brier, bins = ece_brier(probs, targets, n_bins=10)
    return {
        "status": "PASS",
        "n_test": int(len(test_data)),
        "ECE_10bin": ece,
        "BrierScore": brier,
        "bins": bins,
    }


def eval_transformer(args: argparse.Namespace) -> Dict[str, object]:
    sys.path.insert(0, str(REPO / "scripts" / "b2_stage2"))
    from b2_phase4_common import BACKBONES, SequenceHeadModel, batch_ranges, make_batch, read_dataset, subset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    checkpoint = Path(args.checkpoint)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    spec = BACKBONES[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf_path"],
        trust_remote_code=spec["trust_remote_code"],
        local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        spec["hf_path"],
        trust_remote_code=spec["trust_remote_code"],
        local_files_only=True,
    )
    lora_config = state.get("lora_config")
    if lora_config:
        peft_config = LoraConfig(
            r=int(lora_config["rank"]),
            lora_alpha=float(lora_config["alpha"]),
            target_modules=list(lora_config.get("target_modules", ["query", "key", "value"])),
            lora_dropout=0.0,
            bias="none",
            use_dora=bool(lora_config.get("use_dora")),
        )
        backbone = get_peft_model(backbone, peft_config)
    out_dim = 27 if args.dataset == "sider" else 1
    model = SequenceHeadModel(backbone, hidden_size=backbone.config.hidden_size, out_dim=out_dim)
    model.load_state_dict(state["state_dict"])
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    dataset = read_dataset(args.dataset)
    test_indices = checkpoint_split_test_indices(checkpoint, args.dataset, args.protocol, args.seed, len(dataset.smiles))
    test_smiles, test_targets = subset(dataset, test_indices)
    preds = []
    with torch.no_grad():
        for batch in batch_ranges(len(test_smiles), 50, shuffle_seed=None):
            batch_smiles = [test_smiles[i] for i in batch]
            batch_targets = test_targets[np.array(batch, dtype=np.int64)]
            input_ids, attention_mask, _ = make_batch(tokenizer, batch_smiles, batch_targets, spec["native_max_len"], device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(preds, axis=0)
    ece, brier, bins = ece_brier(probs, test_targets, n_bins=10)
    return {
        "status": "PASS",
        "n_test": int(len(test_smiles)),
        "ECE_10bin": ece,
        "BrierScore": brier,
        "bins": bins,
    }


def main() -> None:
    args = parse_args()
    try:
        result = eval_cmpnn(args) if args.backbone == "cmpnn" else eval_transformer(args)
        result.update(
            {
                "dataset": args.dataset,
                "backbone": args.backbone,
                "method": args.method,
                "protocol": args.protocol,
                "seed": args.seed,
                "checkpoint": args.checkpoint,
            }
        )
    except Exception as exc:
        result = {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "method": args.method,
            "protocol": args.protocol,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "status": "FAIL",
            "n_test": "",
            "ECE_10bin": "",
            "BrierScore": "",
            "bins": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(Path(args.out), result)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
