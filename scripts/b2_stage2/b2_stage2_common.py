from pathlib import Path
import csv
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


REPO_ROOT = os.environ.get("BAM_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
TMP_DIR = os.environ.get("BAM_TMP_DIR", os.path.join(REPO_ROOT, "outputs", "tmp"))


DATASETS = {
    "freesolv": {
        "path": "data/freesolv.csv",
        "smiles_col": "smiles",
        "task": "regression",
    },
    "sider": {
        "path": "data/sider.csv",
        "smiles_col": "smiles",
        "task": "classification",
    },
}


BACKBONES = {
    "chemberta2": {
        "hf_path": "DeepChem/ChemBERTa-77M-MLM",
        "trust_remote_code": False,
        "model_revision": None,
        "remote_code_revision": None,
    },
    "molformer_c3": {
        "hf_path": "DeepChem/MoLFormer-c3-1.1B",
        "trust_remote_code": True,
        "model_revision": "9f1b9ea3590833bd0ea1a70e789c5d3da11ba7ed",
        "remote_code_revision": "7b12d946c181a37f6012b9dc3b002275de070314",
    },
}


@dataclass
class DatasetBundle:
    smiles: List[str]
    targets: np.ndarray
    target_cols: List[str]
    task: str


class JsonlLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.training_path = os.path.join(save_dir, "training.log")
        self.verbose_path = os.path.join(save_dir, "verbose.log")
        self._training = open(self.training_path, "w", encoding="utf-8", buffering=1)
        self._verbose = open(self.verbose_path, "w", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"{timestamp()} {message}"
        print(line, flush=True)
        self._training.write(line + "\n")

    def verbose(self, event: str, payload: Dict[str, object]) -> None:
        record = {"time": timestamp(), "event": event, **payload}
        self._verbose.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        self._training.close()
        self._verbose.close()


class MemoryMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.peak_mib = 0
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                value = int(output.splitlines()[0].strip())
                self.samples.append(value)
                self.peak_mib = max(self.peak_mib, value)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)


class SequenceHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, out_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_dataset(dataset_name: str) -> DatasetBundle:
    spec = DATASETS[dataset_name]
    path = os.path.join(REPO_ROOT, spec["path"])
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    smiles_col = spec["smiles_col"]
    target_cols = [c for c in fieldnames if c != smiles_col]
    smiles = [row[smiles_col] for row in rows]
    values: List[List[float]] = []
    for row in rows:
        row_values = []
        for col in target_cols:
            value = row[col]
            row_values.append(float(value) if value != "" else float("nan"))
        values.append(row_values)
    return DatasetBundle(
        smiles=smiles,
        targets=np.array(values, dtype=np.float32),
        target_cols=target_cols,
        task=spec["task"],
    )


def split_indices(n_items: int, seed: int) -> Dict[str, List[int]]:
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    train_size = int(0.8 * n_items)
    train_val_size = int(0.9 * n_items)
    return {
        "train": indices[:train_size],
        "val": indices[train_size:train_val_size],
        "test": indices[train_val_size:],
    }


def load_sprint_freesolv_split_if_available(seed: int) -> Optional[Dict[str, List[int]]]:
    if seed != 0:
        return None
    paths = {
        split: os.path.join(TMP_DIR, f"freesolv_seed0_{split}_indices.txt")
        for split in ("train", "val", "test")
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    loaded: Dict[str, List[int]] = {}
    for split, path in paths.items():
        with open(path, encoding="utf-8") as f:
            loaded[split] = [int(line.strip()) for line in f if line.strip()]
    return loaded


def write_split_artifacts(save_dir: str, dataset: DatasetBundle, splits: Dict[str, List[int]]) -> Dict[str, str]:
    split_dir = os.path.join(save_dir, "split")
    os.makedirs(split_dir, exist_ok=True)
    hashes: Dict[str, str] = {}
    for split_name, indices in splits.items():
        smiles_path = os.path.join(split_dir, f"{split_name}_smiles.txt")
        index_path = os.path.join(split_dir, f"{split_name}_indices.txt")
        with open(smiles_path, "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(dataset.smiles[idx] + "\n")
        with open(index_path, "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(f"{idx}\n")
        hashes[split_name] = sha256_file(smiles_path)
    return hashes


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset(dataset: DatasetBundle, indices: Sequence[int]) -> Tuple[List[str], np.ndarray]:
    return [dataset.smiles[i] for i in indices], dataset.targets[np.array(indices, dtype=np.int64)]


def batch_ranges(n_items: int, batch_size: int, shuffle_seed: Optional[int] = None) -> Iterable[List[int]]:
    order = list(range(n_items))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    for start in range(0, n_items, batch_size):
        yield order[start : start + batch_size]


def make_batch(tokenizer, smiles: Sequence[str], targets: np.ndarray, max_len: int, device: torch.device):
    encoded = tokenizer(
        list(smiles),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    return input_ids, attention_mask, y


def masked_loss(logits: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    mask = torch.isfinite(y)
    if task == "regression":
        logits = logits.view_as(y)
        diff = (logits - torch.nan_to_num(y, nan=0.0))[mask]
        return torch.mean(diff * diff)
    target = torch.nan_to_num(y, nan=0.0)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return raw[mask].mean()


def evaluate_loss(model, tokenizer, smiles: List[str], targets: np.ndarray, task: str, batch_size: int, max_len: int, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    weights: List[int] = []
    with torch.no_grad():
        for batch in batch_ranges(len(smiles), batch_size):
            batch_smiles = [smiles[i] for i in batch]
            batch_targets = targets[np.array(batch, dtype=np.int64)]
            input_ids, attention_mask, y = make_batch(tokenizer, batch_smiles, batch_targets, max_len, device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_loss(logits, y, task)
            losses.append(float(loss.detach().cpu()))
            weights.append(len(batch))
    model.train()
    return float(np.average(losses, weights=weights)) if losses else float("nan")


def compute_data_scaler(train_targets: np.ndarray, task: str) -> Optional[Dict[str, List[float]]]:
    if task != "regression":
        return None
    means = np.nanmean(train_targets, axis=0)
    stds = np.nanstd(train_targets, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return {"means": means.astype(float).tolist(), "stds": stds.astype(float).tolist()}


def apply_data_scaler(targets: np.ndarray, scaler: Optional[Dict[str, List[float]]]) -> np.ndarray:
    if scaler is None:
        return targets
    means = np.array(scaler["means"], dtype=np.float32)
    stds = np.array(scaler["stds"], dtype=np.float32)
    return (targets - means) / stds


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_lora_config(
    method: str,
    backbone: str,
    hf_path: str,
    use_dora: bool = False,
    stage: str = "B2 Stage-2 Phase 2 timing pilot",
) -> Dict[str, object]:
    method_name = "DoRA + Head" if use_dora else "LoRA + Head"
    return {
        "method": method_name,
        "rank": 8,
        "alpha": 16,
        "target_modules": ["query", "key", "value"],
        "ffn_policy": "trainable",
        "bias_policy": "none",
        "use_dora": use_dora,
        "ablation_method": None,
        "backbone": backbone,
        "hf_path": hf_path,
        "stage": stage,
        "peft_method": method,
    }


def save_checkpoint(path: str, model: nn.Module, args: Dict[str, object], data_scaler: Optional[Dict[str, List[float]]], lora_config: Dict[str, object]) -> None:
    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "features_scaler": None,
        "lora_config": lora_config,
    }
    torch.save(state, path)
