#!/usr/bin/env python3
"""N5 rank sensitivity smoke and production orchestrator."""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
LOG_DIR = REPO / "logs" / "n5_rank_sensitivity"
STATE_PATH = REPO / "docs" / "stage3" / "n5_rank_sensitivity_runner_state.json"
FINALIZER = REPO / "scripts" / "n5_rank_sensitivity" / "n5_finalize.py"
TRANSFORMER_TRAIN = REPO / "scripts" / "n5_rank_sensitivity" / "train_peft_rank_variant.py"
DATASETS = ["freesolv", "esol", "lipo", "bace", "bbbp", "sider"]
SEEDS = [0, 1, 2]
RANKS = [4, 16]
METHODS = ["lora", "dora"]
BACKBONES = ["cmpnn", "chemberta2", "molformer_c3"]
ENV = {"cmpnn": "kapt-5090", "chemberta2": "kapt-chemberta2", "molformer_c3": "kapt-molformer-c3"}
DATASET_INFO = {
    "freesolv": {"path": "data/freesolv.csv", "dataset_type": "regression", "metric": "rmse"},
    "esol": {"path": "data/esol.csv", "dataset_type": "regression", "metric": "rmse"},
    "lipo": {"path": "data/lipo.csv", "dataset_type": "regression", "metric": "rmse"},
    "bace": {"path": "data/bace.csv", "dataset_type": "classification", "metric": "auc"},
    "bbbp": {"path": "data/bbbp.csv", "dataset_type": "classification", "metric": "auc"},
    "sider": {"path": "data/sider.csv", "dataset_type": "classification", "metric": "auc"},
}
INPUT_PATHS = [
    REPO / "docs" / "stage3" / "scaffold_split_extension_results.json",
    REPO / "docs" / "scaffold" / "scaffold_make_split.py",
    REPO / "docs" / "scaffold" / "scaffold_transformer_train.py",
    *[REPO / DATASET_INFO[name]["path"] for name in DATASETS],
]
FINAL_RE = re.compile(r"Final test (rmse|auc|mean_auc) = ([0-9.]+)")
EPOCH_RE = re.compile(r"Epoch ([0-9]+)/([0-9]+)")
TRAINABLE_RE = re.compile(r"Trainable params: ([0-9,]+)")


def quote(value: Any) -> str:
    return shlex.quote(str(value))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_sha() -> Dict[str, str]:
    return {rel(path): sha256(path) for path in INPUT_PATHS if path.exists()}


def read_text(path: Path, limit: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(text) > limit:
        return text[:limit] + f"\n... [truncated, total chars={len(text)}]\n"
    return text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def shell_capture(command: str) -> Dict[str, Any]:
    proc = subprocess.run(["bash", "-lc", command], cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": proc.returncode, "output": proc.stdout}


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(update: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    state.update(update)
    state["updated_at"] = now()
    write_json(STATE_PATH, state)
    return state


def ensure_initial_state() -> None:
    state = load_state()
    if "started_at" not in state:
        save_state(
            {
                "started_at": now(),
                "input_sha_pre": input_sha(),
                "env_pre": shell_capture("source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda env list"),
            }
        )


def gpu_used_mib() -> Optional[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(output.splitlines()[0].strip())
    except Exception:
        return None


def shell_monitored(command: str, log_path: Path) -> Tuple[int, str, Optional[int]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak = gpu_used_mib() or 0
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.Popen(["bash", "-lc", command], cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            value = gpu_used_mib()
            if value is not None:
                peak = max(peak, value)
            time.sleep(1.0)
        value = gpu_used_mib()
        if value is not None:
            peak = max(peak, value)
    return int(proc.returncode), read_text(log_path), peak


def conda_command(env: str, inner: str) -> str:
    return (
        "set -o pipefail; "
        "source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {quote(env)} && "
        f"cd {quote(REPO)} && "
        f"{inner}"
    )


def split_path(dataset: str, seed: int) -> Path:
    return REPO / "dumped" / "n5_rank_sensitivity_splits" / f"{dataset}_scaffold_seed{seed}.json"


def save_dir(backbone: str, dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Path:
    prefix = "smoke_n5_rank_sensitivity" if smoke else "n5_rank_sensitivity"
    return REPO / "dumped" / f"{prefix}_{dataset}_{method}_r{rank}_{backbone}_scaffold_seed{seed}"


def checkpoint_path(backbone: str, dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Path:
    base = save_dir(backbone, dataset, method, rank, seed, smoke=smoke)
    if backbone == "cmpnn":
        return base / "run_0" / "model_0" / "model.pt"
    return base / "model.pt"


def cmpnn_log_path(dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Path:
    prefix = "smoke_n5_rank_sensitivity" if smoke else "n5_rank_sensitivity"
    exp_name = f"{prefix}_{dataset}_{method}_r{rank}_cmpnn_scaffold"
    return REPO / "logs" / exp_name / f"seed{seed}" / "training.log"


def cmpnn_command(dataset: str, method: str, rank: int, seed: int, epochs: int, smoke: bool = False, batch_size: int = 50) -> str:
    info = DATASET_INFO[dataset]
    out_dir = save_dir("cmpnn", dataset, method, rank, seed, smoke=smoke)
    prefix = "smoke_n5_rank_sensitivity" if smoke else "n5_rank_sensitivity"
    exp_name = f"{prefix}_{dataset}_{method}_r{rank}_cmpnn_scaffold"
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
        str(epochs),
        "--batch_size",
        str(batch_size),
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
        rel(out_dir),
        "--exp_name",
        exp_name,
        "--exp_id",
        f"seed{seed}",
        "--peft_method",
        method,
        "--lora_rank",
        str(rank),
        "--lora_alpha",
        "16",
    ]
    return conda_command("kapt-5090", " ".join(quote(arg) for arg in args))


def transformer_command(backbone: str, dataset: str, method: str, rank: int, seed: int, run_index: int, epochs: int, smoke: bool = False, batch_size: int = 50) -> str:
    args = [
        "python",
        str(TRANSFORMER_TRAIN),
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
        "16",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--save_dir",
        str(save_dir(backbone, dataset, method, rank, seed, smoke=smoke)),
        "--lr",
        "0.001",
        "--patience",
        "10",
        "--run_index",
        str(run_index),
        "--total_runs",
        "216",
        "--split_indices_path",
        str(split_path(dataset, seed)),
    ]
    if smoke:
        args.append("--disable_early_stop")
    return conda_command(ENV[backbone], " ".join(quote(arg) for arg in args))


def parse_cmpnn(backbone: str, dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Dict[str, Any]:
    log_path = cmpnn_log_path(dataset, method, rank, seed, smoke=smoke)
    text = read_text(log_path)
    finals = [(m.group(1), float(m.group(2))) for m in FINAL_RE.finditer(text)]
    epochs = [(int(m.group(1)), int(m.group(2))) for m in EPOCH_RE.finditer(text)]
    trainable = None
    for match in TRAINABLE_RE.finditer(text):
        trainable = int(match.group(1).replace(",", ""))
    ckpt = checkpoint_path(backbone, dataset, method, rank, seed, smoke=smoke)
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


def parse_transformer(backbone: str, dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Dict[str, Any]:
    metrics_path = save_dir(backbone, dataset, method, rank, seed, smoke=smoke) / "metrics.json"
    ckpt = checkpoint_path(backbone, dataset, method, rank, seed, smoke=smoke)
    if not metrics_path.exists():
        return {"metric_type": DATASET_INFO[dataset]["metric"], "metric": None, "complete": False, "checkpoint": rel(ckpt), "checkpoint_exists": ckpt.exists()}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
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
        "training_log": rel(save_dir(backbone, dataset, method, rank, seed, smoke=smoke) / "training.log"),
    }


def parse_result(backbone: str, dataset: str, method: str, rank: int, seed: int, smoke: bool = False) -> Dict[str, Any]:
    if backbone == "cmpnn":
        return parse_cmpnn(backbone, dataset, method, rank, seed, smoke=smoke)
    return parse_transformer(backbone, dataset, method, rank, seed, smoke=smoke)


def contains_oom(text: str) -> bool:
    lower = text.lower()
    return "out of memory" in lower or "outofmemoryerror" in lower or "cuda error: out of memory" in lower


def smoke_specs() -> List[Dict[str, Any]]:
    return [
        {"index": 1, "backbone": "cmpnn", "dataset": "lipo", "method": "lora", "rank": 4, "seed": 0},
        {"index": 2, "backbone": "chemberta2", "dataset": "lipo", "method": "dora", "rank": 16, "seed": 0},
        {"index": 3, "backbone": "molformer_c3", "dataset": "lipo", "method": "dora", "rank": 4, "seed": 0},
    ]


def production_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    index = 1
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for method in METHODS:
                for rank in RANKS:
                    for seed in SEEDS:
                        specs.append({"index": index, "backbone": backbone, "dataset": dataset, "method": method, "rank": rank, "seed": seed})
                        index += 1
    return specs


def command_for(spec: Dict[str, Any], epochs: int, smoke: bool, batch_size: int = 50) -> str:
    if spec["backbone"] == "cmpnn":
        return cmpnn_command(spec["dataset"], spec["method"], int(spec["rank"]), int(spec["seed"]), epochs=epochs, smoke=smoke, batch_size=batch_size)
    return transformer_command(
        spec["backbone"],
        spec["dataset"],
        spec["method"],
        int(spec["rank"]),
        int(spec["seed"]),
        int(spec["index"]),
        epochs=epochs,
        smoke=smoke,
        batch_size=batch_size,
    )


def run_one(spec: Dict[str, Any], epochs: int, smoke: bool, max_attempts: int = 2) -> Dict[str, Any]:
    attempts = 0
    last_output = ""
    workaround = ""
    batch_size = 50
    start = time.time()
    while attempts < max_attempts:
        attempts += 1
        command = command_for(spec, epochs=epochs, smoke=smoke, batch_size=batch_size)
        prefix = "smoke" if smoke else "prod"
        log_path = LOG_DIR / f"{prefix}_{spec['index']:03d}_{spec['backbone']}_{spec['dataset']}_{spec['method']}_r{spec['rank']}_seed{spec['seed']}_attempt{attempts}.log"
        rc, output, peak = shell_monitored(command, log_path)
        last_output = output
        parsed = parse_result(spec["backbone"], spec["dataset"], spec["method"], int(spec["rank"]), int(spec["seed"]), smoke=smoke)
        if rc == 0 and parsed.get("complete"):
            return {
                **spec,
                "protocol": "scaffold",
                "status": "PASS",
                "attempts": attempts,
                "workaround": workaround,
                "returncode": rc,
                "elapsed_sec": time.time() - start,
                "peak_mib": parsed.get("peak_mib") or peak,
                "metric": parsed.get("metric"),
                "metric_type": parsed.get("metric_type"),
                "epochs_completed": parsed.get("epochs_completed"),
                "epochs_requested": parsed.get("epochs_requested"),
                "trainable_params": parsed.get("trainable_params"),
                "checkpoint": parsed.get("checkpoint"),
                "checkpoint_exists": parsed.get("checkpoint_exists"),
                "training_log": parsed.get("training_log"),
                "run_log": rel(log_path),
                "command": command,
            }
        if contains_oom(output) and attempts < max_attempts:
            batch_size = 25
            workaround = "OOM_BATCH25_RETRY"
        elif attempts < max_attempts:
            workaround = "SAME_COMMAND_RETRY"
    parsed = parse_result(spec["backbone"], spec["dataset"], spec["method"], int(spec["rank"]), int(spec["seed"]), smoke=smoke)
    return {
        **spec,
        "protocol": "scaffold",
        "status": "FAIL",
        "attempts": attempts,
        "workaround": workaround,
        "returncode": None,
        "elapsed_sec": time.time() - start,
        "peak_mib": parsed.get("peak_mib"),
        "metric": parsed.get("metric"),
        "metric_type": parsed.get("metric_type"),
        "epochs_completed": parsed.get("epochs_completed"),
        "epochs_requested": parsed.get("epochs_requested"),
        "trainable_params": parsed.get("trainable_params"),
        "checkpoint": parsed.get("checkpoint"),
        "checkpoint_exists": parsed.get("checkpoint_exists"),
        "training_log": parsed.get("training_log"),
        "run_log": rel(log_path),
        "last_error_excerpt": last_output[-4000:],
        "command": command_for(spec, epochs=epochs, smoke=smoke, batch_size=batch_size),
    }


def run_smoke() -> None:
    ensure_initial_state()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for spec in smoke_specs():
        result = run_one(spec, epochs=5, smoke=True, max_attempts=2)
        results.append(result)
        save_state({"phase": "smoke", "smoke_results": results})
        if result["status"] != "PASS":
            subprocess.run(["python", str(FINALIZER), "smoke"], cwd=str(REPO), check=False)
            raise SystemExit(1)
    save_state({"phase": "smoke_done", "smoke_results": results, "smoke_status": "PASS"})
    subprocess.run(["python", str(FINALIZER), "smoke"], cwd=str(REPO), check=False)


def run_production() -> None:
    ensure_initial_state()
    state = load_state()
    smoke = state.get("smoke_results", [])
    if len(smoke) < 3 or any(row.get("status") != "PASS" for row in smoke):
        save_state({"phase": "production_skipped", "production_skip_reason": "smoke not PASS"})
        subprocess.run(["python", str(FINALIZER), "production"], cwd=str(REPO), check=False)
        raise SystemExit(1)

    results: List[Dict[str, Any]] = state.get("production_results", [])
    existing = {(r["backbone"], r["dataset"], r["method"], int(r["rank"]), int(r["seed"])): r for r in results}
    for spec in production_specs():
        key = (spec["backbone"], spec["dataset"], spec["method"], int(spec["rank"]), int(spec["seed"]))
        if key in existing and existing[key].get("status") == "PASS":
            continue
        result = run_one(spec, epochs=50 if spec["backbone"] == "cmpnn" else 30, smoke=False, max_attempts=2)
        existing[key] = result
        ordered = [existing[(s["backbone"], s["dataset"], s["method"], int(s["rank"]), int(s["seed"]))] for s in production_specs() if (s["backbone"], s["dataset"], s["method"], int(s["rank"]), int(s["seed"])) in existing]
        save_state({"phase": "production", "production_results": ordered})
    save_state({"phase": "production_done", "production_results": [existing[(s["backbone"], s["dataset"], s["method"], int(s["rank"]), int(s["seed"]))] for s in production_specs()]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "production"])
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke()
    else:
        run_production()


if __name__ == "__main__":
    main()
