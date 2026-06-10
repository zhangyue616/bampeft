#!/usr/bin/env python3
"""Shared helpers for the 2026-05-19 overnight supplementary experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_ROOT = REPO / "docs" / "overnight_2026-05-19"
T1_DIR = OUT_ROOT / "t1_n6_rank_boundary"
T2_DIR = OUT_ROOT / "t2_n7_alpha_sensitivity"
T3_DIR = OUT_ROOT / "t3_calibration_metrics"
T4_DIR = OUT_ROOT / "t4_compute_cost"
T5_DIR = OUT_ROOT / "t5_bootstrap_stability"
T6_DIR = OUT_ROOT / "t6_sklearn_baselines"
CRITICAL_FINDINGS = OUT_ROOT / "critical_findings.md"

N5_PAIRED_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_vs_r8_baseline_paired_comparison.csv"
N5_RESULTS_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_results_table.csv"
N5_SPLIT_DIR = REPO / "dumped" / "n5_rank_sensitivity_splits"
STAGE2_BASELINE_JSON = REPO / "docs" / "stage3" / "scaffold_split_extension_results.json"

DATASETS = ["freesolv", "esol", "lipo", "bace", "bbbp", "sider"]
CLASSIFICATION_DATASETS = ["bace", "bbbp", "sider"]
REGRESSION_DATASETS = ["freesolv", "esol", "lipo"]
SEEDS = [0, 1, 2]
METHODS = ["lora", "dora"]
BACKBONES = ["cmpnn", "chemberta2", "molformer_c3"]
TRANSFORMER_BACKBONES = ["chemberta2", "molformer_c3"]
ENV_BY_BACKBONE = {
    "cmpnn": "kapt-5090",
    "chemberta2": "kapt-chemberta2",
    "molformer_c3": "kapt-molformer-c3",
}

DATASET_INFO = {
    "freesolv": {"path": "data/freesolv.csv", "dataset_type": "regression", "metric": "rmse"},
    "esol": {"path": "data/esol.csv", "dataset_type": "regression", "metric": "rmse"},
    "lipo": {"path": "data/lipo.csv", "dataset_type": "regression", "metric": "rmse"},
    "bace": {"path": "data/bace.csv", "dataset_type": "classification", "metric": "auc"},
    "bbbp": {"path": "data/bbbp.csv", "dataset_type": "classification", "metric": "auc"},
    "sider": {"path": "data/sider.csv", "dataset_type": "classification", "metric": "auc"},
}

FINAL_RE = re.compile(r"Final test (rmse|auc|mean_auc) = ([0-9.eE+-]+)")
EPOCH_RE = re.compile(r"Epoch ([0-9]+)/([0-9]+)")
TRAINABLE_RE = re.compile(r"Trainable params: ([0-9,]+)")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def quote(value: Any) -> str:
    return shlex.quote(str(value))


def ensure_output_dirs() -> None:
    for path in [OUT_ROOT, T1_DIR, T2_DIR, T3_DIR, T4_DIR, T5_DIR, T6_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    (T3_DIR / "t3_reliability_diagrams").mkdir(parents=True, exist_ok=True)


def read_text(path: Path, limit: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n... [truncated]\n"
    return text


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_capture(command: str, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return {"command": command, "returncode": int(proc.returncode), "output": proc.stdout}


def conda_command(env_name: str, inner: str, hide_cuda: bool = False) -> str:
    prefix = "CUDA_VISIBLE_DEVICES='' " if hide_cuda else ""
    return (
        "set -o pipefail; "
        "source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {quote(env_name)} && "
        f"cd {quote(REPO)} && "
        f"{prefix}{inner}"
    )


def gpu_snapshot() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        parts = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "memory_total_mib": int(float(parts[1])),
            "memory_used_mib": int(float(parts[2])),
            "memory_free_mib": int(float(parts[3])),
            "temperature_c": int(float(parts[4])),
            "raw": output,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def git_status_lines() -> List[str]:
    result = shell_capture("git status --short --untracked-files=all")
    lines = []
    for line in result["output"].splitlines():
        if not line.strip():
            continue
        # Conda activation can put libtinfo warnings on stdout/stderr in this
        # environment. Only porcelain-short status lines are meaningful here.
        if len(line) >= 3 and line[2] == " ":
            lines.append(line)
    return lines


def has_tracked_git_change() -> Tuple[bool, List[str]]:
    lines = git_status_lines()
    tracked = [line for line in lines if not line.startswith("?? ")]
    return bool(tracked), tracked


def allowed_untracked(line: str) -> bool:
    if not line.startswith("?? "):
        return False
    path = line[3:]
    return (
        path.startswith("docs/overnight_2026-05-19/")
        or path == "docs/overnight_2026-05-19"
        or path.startswith("dumped/n6_rank_boundary_")
        or path.startswith("dumped/n7_alpha_sensitivity_")
        or path.startswith("scripts/overnight/")
        or path == "scripts/overnight"
    )


def worktree_gate() -> Dict[str, Any]:
    lines = git_status_lines()
    tracked = [line for line in lines if not line.startswith("?? ")]
    unexpected = [line for line in lines if line.startswith("?? ") and not allowed_untracked(line)]
    return {
        "lines": lines,
        "tracked_changes": tracked,
        "unexpected_untracked": unexpected,
        "pass": not tracked and not unexpected,
    }


def lower_is_better(metric_type: str) -> bool:
    return metric_type in {"rmse", "mae", "mse", "loss", "cross_entropy"}


def paired_bad_diff(variant_metric: float, baseline_metric: float, metric_type: str) -> float:
    if lower_is_better(metric_type):
        return float(variant_metric) - float(baseline_metric)
    return float(baseline_metric) - float(variant_metric)


def load_n5_scope() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_csv(N5_PAIRED_CSV)
    unique: Dict[Tuple[str, str, str, int, str], Dict[str, Any]] = {}
    ranks = sorted({int(row["rank"]) for row in rows})
    for row in rows:
        key = (row["backbone"], row["dataset"], row["method"], int(row["seed"]), row["protocol"])
        current = unique.get(key)
        candidate = {
            "backbone": row["backbone"],
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "protocol": row["protocol"],
            "metric_type": row["metric_type"],
            "task_type": row["task_type"],
            "baseline_metric": float(row["r8_baseline_metric"]),
            "baseline_checkpoint": row["r8_checkpoint"],
        }
        if current is None:
            unique[key] = candidate
        else:
            if abs(float(current["baseline_metric"]) - float(candidate["baseline_metric"])) > 1e-12:
                raise RuntimeError(f"Inconsistent r8 baseline metric for {key}")
            if current["baseline_checkpoint"] != candidate["baseline_checkpoint"]:
                raise RuntimeError(f"Inconsistent r8 checkpoint for {key}")
    base_rows = sorted(unique.values(), key=lambda r: (BACKBONES.index(r["backbone"]), DATASETS.index(r["dataset"]), METHODS.index(r["method"]), r["seed"]))
    info = {
        "n5_paired_rows": len(rows),
        "n5_unique_ranks": ranks,
        "base_cells_without_rank": len(base_rows),
        "protocols": sorted({row["protocol"] for row in rows}),
        "datasets": sorted({row["dataset"] for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "backbones": sorted({row["backbone"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
    }
    return base_rows, info


def mean_n5_elapsed_sec(default: float = 70.0) -> float:
    if not N5_RESULTS_CSV.exists():
        return default
    values = []
    for row in read_csv(N5_RESULTS_CSV):
        try:
            if row.get("status") == "PASS":
                values.append(float(row["elapsed_sec"]))
        except Exception:
            pass
    return float(np.mean(values)) if values else default


def split_path(dataset: str, seed: int) -> Path:
    return N5_SPLIT_DIR / f"{dataset}_scaffold_seed{seed}.json"


def save_dir(experiment: str, backbone: str, dataset: str, method: str, variant: Any, seed: int, protocol: str = "scaffold") -> Path:
    if experiment == "t1":
        return REPO / "dumped" / f"n6_rank_boundary_{dataset}_{method}_r{int(variant)}_{backbone}_{protocol}_seed{seed}"
    if experiment == "t2":
        return REPO / "dumped" / f"n7_alpha_sensitivity_{dataset}_{method}_alpha{int(variant)}_{backbone}_{protocol}_seed{seed}"
    raise ValueError(experiment)


def checkpoint_path(experiment: str, backbone: str, dataset: str, method: str, variant: Any, seed: int) -> Path:
    base = save_dir(experiment, backbone, dataset, method, variant, seed)
    if backbone == "cmpnn":
        return base / "run_0" / "model_0" / "model.pt"
    return base / "model.pt"


def cmpnn_log_path(experiment: str, dataset: str, method: str, variant: Any, seed: int) -> Path:
    if experiment == "t1":
        stem = f"n6_rank_boundary_{dataset}_{method}_r{int(variant)}_cmpnn_scaffold"
        return T1_DIR / "cmpnn_logs" / stem / f"seed{seed}" / "training.log"
    if experiment == "t2":
        stem = f"n7_alpha_sensitivity_{dataset}_{method}_alpha{int(variant)}_cmpnn_scaffold"
        return T2_DIR / "cmpnn_logs" / stem / f"seed{seed}" / "training.log"
    raise ValueError(experiment)


def cmpnn_command(experiment: str, dataset: str, method: str, variant: Any, seed: int) -> str:
    info = DATASET_INFO[dataset]
    if experiment == "t1":
        rank = int(variant)
        alpha = 16
        exp_stem = f"n6_rank_boundary_{dataset}_{method}_r{rank}_cmpnn_scaffold"
        exp_name = f"../docs/overnight_2026-05-19/t1_n6_rank_boundary/cmpnn_logs/{exp_stem}"
    elif experiment == "t2":
        rank = 8
        alpha = int(variant)
        exp_stem = f"n7_alpha_sensitivity_{dataset}_{method}_alpha{alpha}_cmpnn_scaffold"
        exp_name = f"../docs/overnight_2026-05-19/t2_n7_alpha_sensitivity/cmpnn_logs/{exp_stem}"
    else:
        raise ValueError(experiment)

    args = [
        "python",
        "train.py",
        "--data_path",
        info["path"],
        "--dataset_type",
        info["dataset_type"],
        "--metric",
        info["metric"],
        "--epochs",
        "50",
        "--batch_size",
        "50",
        "--seed",
        str(seed),
        "--split_type",
        "scaffold_balanced",
        "--split_sizes",
        "0.8",
        "0.1",
        "0.1",
        "--checkpoint_path",
        str(Path(os.environ.get("BAM_CMPNN_PRETRAINED", Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])) / "pretrained" / "original_CMPN_0623_1350_14000th_epoch.pkl"))),
        "--save_dir",
        rel(save_dir(experiment, "cmpnn", dataset, method, variant, seed)),
        "--exp_name",
        exp_name,
        "--exp_id",
        f"seed{seed}",
        "--peft_method",
        method,
        "--lora_rank",
        str(rank),
        "--lora_alpha",
        str(alpha),
    ]
    return conda_command(ENV_BY_BACKBONE["cmpnn"], " ".join(quote(arg) for arg in args))


def transformer_command(experiment: str, backbone: str, dataset: str, method: str, variant: Any, seed: int, run_index: int, total_runs: int) -> str:
    if experiment == "t1":
        rank = int(variant)
        alpha = 16
        stage = "N6 rank boundary production"
    elif experiment == "t2":
        rank = 8
        alpha = int(variant)
        stage = "N7 alpha sensitivity production"
    else:
        raise ValueError(experiment)

    args = [
        "python",
        "scripts/overnight/train_transformer_variant.py",
        "--backbone",
        backbone,
        "--method",
        method,
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--rank",
        str(rank),
        "--lora_alpha",
        str(alpha),
        "--epochs",
        "30",
        "--batch_size",
        "50",
        "--save_dir",
        str(save_dir(experiment, backbone, dataset, method, variant, seed)),
        "--lr",
        "0.001",
        "--patience",
        "10",
        "--run_index",
        str(run_index),
        "--total_runs",
        str(total_runs),
        "--split_indices_path",
        str(split_path(dataset, seed)),
        "--stage",
        stage,
    ]
    return conda_command(ENV_BY_BACKBONE[backbone], " ".join(quote(arg) for arg in args))


def command_for_spec(experiment: str, spec: Dict[str, Any], total_runs: int) -> str:
    if spec["backbone"] == "cmpnn":
        return cmpnn_command(experiment, spec["dataset"], spec["method"], spec["variant"], int(spec["seed"]))
    return transformer_command(
        experiment,
        spec["backbone"],
        spec["dataset"],
        spec["method"],
        spec["variant"],
        int(spec["seed"]),
        int(spec["index"]),
        total_runs,
    )


def parse_cmpnn_result(experiment: str, dataset: str, method: str, variant: Any, seed: int) -> Dict[str, Any]:
    log_path = cmpnn_log_path(experiment, dataset, method, variant, seed)
    text = read_text(log_path)
    finals = [(m.group(1), float(m.group(2))) for m in FINAL_RE.finditer(text)]
    epochs = [(int(m.group(1)), int(m.group(2))) for m in EPOCH_RE.finditer(text)]
    trainable = None
    for match in TRAINABLE_RE.finditer(text):
        trainable = int(match.group(1).replace(",", ""))
    ckpt = checkpoint_path(experiment, "cmpnn", dataset, method, variant, seed)
    metric_type, metric = finals[-1] if finals else (DATASET_INFO[dataset]["metric"], None)
    return {
        "metric_type": metric_type,
        "metric": metric,
        "complete": bool(metric is not None and "Training completed successfully!" in text and ckpt.exists()),
        "epochs_completed": epochs[-1][0] if epochs else None,
        "epochs_requested": epochs[-1][1] if epochs else None,
        "trainable_params": trainable,
        "checkpoint": rel(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "training_log": rel(log_path),
    }


def parse_transformer_result(experiment: str, backbone: str, dataset: str, method: str, variant: Any, seed: int) -> Dict[str, Any]:
    out_dir = save_dir(experiment, backbone, dataset, method, variant, seed)
    metrics_path = out_dir / "metrics.json"
    ckpt = checkpoint_path(experiment, backbone, dataset, method, variant, seed)
    if not metrics_path.exists():
        return {"metric_type": DATASET_INFO[dataset]["metric"], "metric": None, "complete": False, "checkpoint": rel(ckpt), "checkpoint_exists": ckpt.exists()}
    data = read_json(metrics_path)
    return {
        "metric_type": data.get("metric_type", DATASET_INFO[dataset]["metric"]),
        "metric": data.get("final_metric"),
        "complete": bool(data.get("status") == "PASS" and ckpt.exists()),
        "epochs_completed": data.get("epochs_completed"),
        "epochs_requested": data.get("epochs_requested"),
        "trainable_params": data.get("trainable_params"),
        "elapsed_sec": data.get("elapsed_sec"),
        "peak_mib": data.get("nvidia_smi_peak_mib"),
        "checkpoint": rel(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "training_log": rel(out_dir / "training.log"),
    }


def parse_result(experiment: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    if spec["backbone"] == "cmpnn":
        return parse_cmpnn_result(experiment, spec["dataset"], spec["method"], spec["variant"], int(spec["seed"]))
    return parse_transformer_result(experiment, spec["backbone"], spec["dataset"], spec["method"], spec["variant"], int(spec["seed"]))


def contains_oom(text: str) -> bool:
    lower = text.lower()
    return "out of memory" in lower or "outofmemoryerror" in lower or "cuda error: out of memory" in lower


class ResourceGuard:
    def __init__(self, started_epoch: Optional[float] = None, max_wall_sec: float = 8.5 * 3600):
        self.started_epoch = started_epoch
        self.max_wall_sec = max_wall_sec
        self.hot_since: Optional[float] = None
        self.high_mem_since: Optional[float] = None
        self.max_gpu_used_mib = 0
        self.max_temperature_c = 0

    def check(self) -> Optional[str]:
        if self.started_epoch is not None and time.time() - self.started_epoch > self.max_wall_sec:
            return "global wall-clock exceeded 8.5 hours"
        if disk_free_gb(REPO) < 10.0:
            return "disk free below 10 GB"
        dirty, lines = has_tracked_git_change()
        if dirty:
            return "tracked git changes detected: " + "; ".join(lines)
        snap = gpu_snapshot()
        if "error" not in snap:
            used = int(snap["memory_used_mib"])
            temp = int(snap["temperature_c"])
            self.max_gpu_used_mib = max(self.max_gpu_used_mib, used)
            self.max_temperature_c = max(self.max_temperature_c, temp)
            now_sec = time.time()
            if temp > 90:
                self.hot_since = self.hot_since or now_sec
                if now_sec - self.hot_since > 300:
                    return "GPU temperature > 90 C sustained > 5 min"
            else:
                self.hot_since = None
            if used > 31000:
                self.high_mem_since = self.high_mem_since or now_sec
                if now_sec - self.high_mem_since > 300:
                    return "GPU memory > 31000 MiB sustained > 5 min"
            else:
                self.high_mem_since = None
        return None


def shell_monitored(command: str, log_path: Path, guard: ResourceGuard) -> Tuple[int, str, int, Optional[str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak = gpu_snapshot().get("memory_used_mib", 0)
    stop_reason = None
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.Popen(["bash", "-lc", command], cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            snap = gpu_snapshot()
            if "error" not in snap:
                peak = max(int(peak), int(snap["memory_used_mib"]))
            stop_reason = guard.check()
            if stop_reason:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(1.0)
    final_snap = gpu_snapshot()
    if "error" not in final_snap:
        peak = max(int(peak), int(final_snap["memory_used_mib"]))
    return int(proc.returncode if proc.returncode is not None else -15), read_text(log_path), int(peak), stop_reason


def stratified_percentile_bootstrap(rows: List[Dict[str, Any]], diff_key: str, b: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(diff_key)
        if value in (None, ""):
            continue
        try:
            value_f = float(value)
        except Exception:
            continue
        if math.isfinite(value_f):
            grouped[str(row["dataset"])].append(value_f)
    arrays = {key: np.asarray(values, dtype=float) for key, values in grouped.items() if values}
    if not arrays:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(b, dtype=float)
    for i in range(b):
        samples = []
        for values in arrays.values():
            idx = rng.integers(0, len(values), len(values))
            samples.append(values[idx])
        means[i] = float(np.concatenate(samples).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.mean())


def sem(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.asarray(vals, dtype=float).mean())


def md_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def append_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip() + "\n")


def read_indices(path: Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece_brier(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict[str, Any]]]:
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(targets)
    p = probs[mask].reshape(-1)
    y = targets[mask].reshape(-1)
    if len(p) == 0:
        return float("nan"), float("nan"), []
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            bin_mask = (p >= left) & (p <= right)
        else:
            bin_mask = (p >= left) & (p < right)
        count = int(np.sum(bin_mask))
        if count == 0:
            bins.append({"bin": idx, "left": float(left), "right": float(right), "n": 0, "confidence": "", "accuracy": ""})
            continue
        conf = float(np.mean(p[bin_mask]))
        acc = float(np.mean(y[bin_mask]))
        ece += (count / len(p)) * abs(acc - conf)
        bins.append({"bin": idx, "left": float(left), "right": float(right), "n": count, "confidence": conf, "accuracy": acc})
    return float(ece), brier, bins
