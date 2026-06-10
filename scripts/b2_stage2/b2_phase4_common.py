import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
TMP_DIR = Path(os.environ.get("BAM_TMP_DIR", REPO_ROOT / "outputs" / "tmp"))
TOTAL_PHASE4_RUNS = 108
PHASE4_EPOCHS = 30
PHASE4_BATCH_SIZE = 50
PHASE4_PATIENCE = 10
PHASE4_LR = 1e-3

DATASETS = {
    "freesolv": {
        "path": "data/freesolv.csv",
        "smiles_col": "smiles",
        "task": "regression",
        "metric": "rmse",
    },
    "esol": {
        "path": "data/esol.csv",
        "smiles_col": "smiles",
        "task": "regression",
        "metric": "rmse",
    },
    "bace": {
        "path": "data/bace.csv",
        "smiles_col": "smiles",
        "task": "classification",
        "metric": "auc",
    },
    "bbbp": {
        "path": "data/bbbp.csv",
        "smiles_col": "smiles",
        "task": "classification",
        "metric": "auc",
    },
    "sider": {
        "path": "data/sider.csv",
        "smiles_col": "smiles",
        "task": "classification",
        "metric": "mean_auc",
    },
    "lipo": {
        "path": "data/lipo.csv",
        "smiles_col": "smiles",
        "task": "regression",
        "metric": "rmse",
    },
}

BACKBONES = {
    "chemberta2": {
        "hf_path": "DeepChem/ChemBERTa-77M-MLM",
        "trust_remote_code": False,
        "model_revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "remote_code_revision": None,
        "env_name": "kapt-chemberta2",
        "native_max_len": 512,
    },
    "molformer_c3": {
        "hf_path": "DeepChem/MoLFormer-c3-1.1B",
        "trust_remote_code": True,
        "model_revision": "9f1b9ea3590833bd0ea1a70e789c5d3da11ba7ed",
        "remote_code_revision": "7b12d946c181a37f6012b9dc3b002275de070314",
        "remote_code_source": "ibm-research/MoLFormer-XL-both-10pct",
        "env_name": "kapt-molformer-c3",
        "native_max_len": 202,
    },
}

METHODS = ("head", "lora", "dora")
SEEDS = (0, 1, 2)
BACKBONE_ORDER = ("chemberta2", "molformer_c3")
DATASET_ORDER = ("freesolv", "esol", "bace", "bbbp", "sider", "lipo")


@dataclass
class DatasetBundle:
    smiles: List[str]
    targets: np.ndarray
    target_cols: List[str]
    task: str
    metric: str


class JsonlLogger:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.training_path = save_dir / "training.log"
        self.verbose_path = save_dir / "verbose.log"
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


def output_dir(dataset: str, method: str, backbone: str, seed: int) -> Path:
    return REPO_ROOT / "dumped" / f"baseline_{dataset}_{method}_{backbone}_seed{seed}"


def read_dataset(dataset_name: str) -> DatasetBundle:
    spec = DATASETS[dataset_name]
    path = REPO_ROOT / spec["path"]
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
        metric=spec["metric"],
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


def write_split_artifacts(save_dir: Path, dataset: DatasetBundle, splits: Dict[str, List[int]]) -> Dict[str, str]:
    split_dir = save_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    for split_name, indices in splits.items():
        smiles_path = split_dir / f"{split_name}_smiles.txt"
        index_path = split_dir / f"{split_name}_indices.txt"
        with open(smiles_path, "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(dataset.smiles[idx] + "\n")
        with open(index_path, "w", encoding="utf-8") as f:
            for idx in indices:
                f.write(f"{idx}\n")
        hashes[split_name] = sha256_file(smiles_path)
    return hashes


def sha256_file(path: Path) -> str:
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


def inverse_data_scaler(values: np.ndarray, scaler: Optional[Dict[str, List[float]]]) -> np.ndarray:
    if scaler is None:
        return values
    means = np.array(scaler["means"], dtype=np.float32)
    stds = np.array(scaler["stds"], dtype=np.float32)
    return values * stds + means


def evaluate_loss_and_predictions(
    model,
    tokenizer,
    smiles: List[str],
    targets_scaled: np.ndarray,
    task: str,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    model.eval()
    losses: List[float] = []
    weights: List[int] = []
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for batch in batch_ranges(len(smiles), batch_size):
            batch_smiles = [smiles[i] for i in batch]
            batch_targets = targets_scaled[np.array(batch, dtype=np.int64)]
            input_ids, attention_mask, y = make_batch(tokenizer, batch_smiles, batch_targets, max_len, device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_loss(logits, y, task)
            losses.append(float(loss.detach().cpu()))
            weights.append(len(batch))
            predictions.append(logits.detach().cpu().numpy())
    model.train()
    pred = np.concatenate(predictions, axis=0) if predictions else np.empty((0, targets_scaled.shape[1]))
    return float(np.average(losses, weights=weights)) if losses else float("nan"), pred


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y = y_true[mask].astype(int)
    scores = y_score[mask].astype(np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_metric(
    dataset: DatasetBundle,
    predictions_scaled: np.ndarray,
    targets_raw: np.ndarray,
    data_scaler: Optional[Dict[str, List[float]]],
) -> Tuple[float, Dict[str, float]]:
    if dataset.metric == "rmse":
        predictions_raw = inverse_data_scaler(predictions_scaled, data_scaler)
        mask = np.isfinite(targets_raw)
        diff = predictions_raw[mask] - targets_raw[mask]
        rmse = float(math.sqrt(float(np.mean(diff * diff))))
        return rmse, {"rmse": rmse}

    scores = sigmoid_np(predictions_scaled)
    aucs: Dict[str, float] = {}
    for col_idx, col_name in enumerate(dataset.target_cols):
        aucs[col_name] = binary_auc(targets_raw[:, col_idx], scores[:, col_idx])
    valid = [v for v in aucs.values() if math.isfinite(v)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    key = "auc" if len(dataset.target_cols) == 1 else "mean_auc"
    return mean_auc, {key: mean_auc, "per_task_auc": aucs}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_lora_config(method: str, backbone: str, stage: str = "B2 Stage-2 Phase 4 production") -> Dict[str, object]:
    spec = BACKBONES[backbone]
    use_dora = method == "dora"
    if method == "head":
        method_name = "Head-only"
        target_modules: List[str] = []
        rank = 0
        alpha = 0
    else:
        method_name = "DoRA + Head" if use_dora else "LoRA + Head"
        target_modules = ["query", "key", "value"]
        rank = 8
        alpha = 16
    return {
        "method": method_name,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "ffn_policy": "frozen",
        "bias_policy": "none",
        "use_dora": use_dora,
        "ablation_method": None,
        "backbone": backbone,
        "hf_path": spec["hf_path"],
        "stage": stage,
        "peft_method": method,
    }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def gpu_info() -> Dict[str, object]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        name, total, free = [part.strip() for part in output.splitlines()[0].split(",")]
        return {"name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment_provenance(backbone: str) -> Dict[str, object]:
    spec = BACKBONES[backbone]
    return {
        "stage": "B2 Stage-2 Phase 4 production",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "numpy": np.__version__,
        "pandas": package_version("pandas"),
        "rdkit": package_version("rdkit") or package_version("rdkit-pypi"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": gpu_info(),
        "hf_path": spec["hf_path"],
        "hf_model_revision": spec["model_revision"],
        "remote_code_revision": spec["remote_code_revision"],
        "remote_code_source": spec.get("remote_code_source"),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: Dict[str, object],
    data_scaler: Optional[Dict[str, List[float]]],
    lora_config: Dict[str, object],
    provenance: Dict[str, object],
) -> None:
    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "features_scaler": None,
        "lora_config": lora_config,
        "provenance": provenance,
    }
    torch.save(state, path)


def gate_status(projected_days: float) -> str:
    if projected_days > 6.0:
        return "red / hard STOP"
    if projected_days >= 5.0:
        return "yellow"
    return "green"


def all_run_specs(
    datasets: Sequence[str] = DATASET_ORDER,
    methods: Sequence[str] = METHODS,
    backbones: Sequence[str] = BACKBONE_ORDER,
    seeds: Sequence[int] = SEEDS,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for seed in seeds:
        for dataset in datasets:
            for backbone in backbones:
                for method in methods:
                    specs.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "backbone": backbone,
                            "seed": seed,
                            "max_len": BACKBONES[backbone]["native_max_len"],
                            "save_dir": str(output_dir(dataset, method, backbone, seed)),
                        }
                    )
    return specs


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


def load_json(path: Path) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
