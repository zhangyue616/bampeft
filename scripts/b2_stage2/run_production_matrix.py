import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from b2_phase4_common import (
    BACKBONES,
    DATASET_ORDER,
    METHODS,
    PHASE4_BATCH_SIZE,
    PHASE4_EPOCHS,
    PHASE4_LR,
    PHASE4_PATIENCE,
    REPO_ROOT,
    SEEDS,
    TOTAL_PHASE4_RUNS,
    all_run_specs,
    gate_status,
    load_json,
    output_dir,
    write_json,
)


PROGRESS_JSON = REPO_ROOT / "docs/stage3/b2_phase4_progress.json"
PROGRESS_CSV = REPO_ROOT / "docs/stage3/b2_phase4_progress.csv"


def parse_csv_list(value: str, default):
    if value == "all":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B2 Phase 4 production matrix with v0.4 wall-time gate")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--backbones", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--epochs", type=int, default=PHASE4_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PHASE4_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PHASE4_PATIENCE)
    parser.add_argument("--lr", type=float, default=PHASE4_LR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def env_command(spec: Dict[str, object], run_index: int, args: argparse.Namespace) -> str:
    backbone = str(spec["backbone"])
    env_name = BACKBONES[backbone]["env_name"]
    save_dir = str(spec["save_dir"])
    log_path = f"{os.environ.get('BAM_TMP_DIR', 'outputs/tmp')}/b2_phase4_run_{run_index:03d}_{backbone}_{spec['method']}_{spec['dataset']}_seed{spec['seed']}.log"
    py_args = [
        "python",
        "scripts/b2_stage2/train_b2_phase4.py",
        "--backbone",
        backbone,
        "--method",
        str(spec["method"]),
        "--dataset",
        str(spec["dataset"]),
        "--seed",
        str(spec["seed"]),
        "--epochs",
        str(args.epochs),
        "--max_len",
        str(spec["max_len"]),
        "--batch_size",
        str(args.batch_size),
        "--save_dir",
        save_dir,
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--run_index",
        str(run_index),
        "--total_runs",
        str(TOTAL_PHASE4_RUNS),
    ]
    quoted = " ".join(shlex.quote(part) for part in py_args)
    return (
        "set -o pipefail; "
        "source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {shlex.quote(env_name)} && "
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{quoted} > {shlex.quote(log_path)} 2>&1"
    )


def completed_phase4_metrics(path: Path, expected_epochs: int) -> Dict[str, object]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = load_json(metrics_path)
    except Exception:
        return {}
    if (
        data.get("status") == "PASS"
        and data.get("phase") == "B2 Stage-2 Phase 4 production"
        and int(data.get("epochs_requested", -1)) == expected_epochs
    ):
        return data
    return {}


def append_projection(record: Dict[str, object], progress: List[Dict[str, object]]) -> Dict[str, object]:
    completed = [r for r in progress if r.get("status") == "PASS"]
    elapsed_sum = sum(float(r["wall_time_s"]) for r in completed)
    projected_days = elapsed_sum / len(completed) * TOTAL_PHASE4_RUNS / 86400.0 if completed else float("nan")
    record["cumulative_projected_total_days"] = projected_days
    record["gate_status"] = gate_status(projected_days)
    return record


def write_progress(progress: List[Dict[str, object]]) -> None:
    write_json(PROGRESS_JSON, progress)
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_index",
        "backbone",
        "method",
        "dataset",
        "seed",
        "status",
        "final_metric",
        "metric_type",
        "epoch_count_actual",
        "early_stop_epoch",
        "wall_time_s",
        "cumulative_projected_total_days",
        "gate_status",
        "save_dir",
    ]
    with open(PROGRESS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in progress:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def metric_record_from_metrics(run_index: int, spec: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_index": run_index,
        "backbone": spec["backbone"],
        "method": spec["method"],
        "dataset": spec["dataset"],
        "seed": spec["seed"],
        "status": metrics.get("status", "UNKNOWN"),
        "final_metric": metrics.get("final_metric"),
        "metric_type": metrics.get("metric_type"),
        "epoch_count_actual": metrics.get("epochs_completed"),
        "early_stop_epoch": metrics.get("early_stop_epoch"),
        "wall_time_s": metrics.get("elapsed_sec"),
        "peak_mib": metrics.get("nvidia_smi_peak_mib"),
        "save_dir": spec["save_dir"],
    }


def main() -> None:
    args = parse_args()
    datasets = parse_csv_list(args.datasets, DATASET_ORDER)
    methods = parse_csv_list(args.methods, METHODS)
    backbones = parse_csv_list(args.backbones, BACKBONES.keys())
    seeds = [int(x) for x in parse_csv_list(args.seeds, SEEDS)]
    specs = all_run_specs(datasets=datasets, methods=methods, backbones=backbones, seeds=seeds)
    progress: List[Dict[str, object]] = []
    executed_this_invocation = 0

    for idx, spec in enumerate(specs, start=1):
        save_dir = output_dir(str(spec["dataset"]), str(spec["method"]), str(spec["backbone"]), int(spec["seed"]))
        spec["save_dir"] = str(save_dir)
        if args.resume:
            existing = completed_phase4_metrics(save_dir, args.epochs)
            if existing:
                record = metric_record_from_metrics(idx, spec, existing)
                progress.append(record)
                append_projection(record, progress)
                print(
                    f"SKIP completed run {idx}/{len(specs)} {spec['backbone']} {spec['method']} "
                    f"{spec['dataset']} seed{spec['seed']} gate={record['gate_status']}",
                    flush=True,
                )
                write_progress(progress)
                continue

        if args.max_runs is not None and executed_this_invocation >= args.max_runs:
            print(f"MAX_RUNS_REACHED executed={executed_this_invocation}", flush=True)
            break

        command = env_command(spec, idx, args)
        print(
            f"RUN {idx}/{len(specs)} backbone={spec['backbone']} method={spec['method']} "
            f"dataset={spec['dataset']} seed={spec['seed']}",
            flush=True,
        )
        start = time.time()
        proc = subprocess.run(["bash", "-lc", command], cwd=str(REPO_ROOT), text=True)
        executed_this_invocation += 1
        if proc.returncode != 0:
            record = {
                "run_index": idx,
                "backbone": spec["backbone"],
                "method": spec["method"],
                "dataset": spec["dataset"],
                "seed": spec["seed"],
                "status": "FAIL",
                "wall_time_s": time.time() - start,
                "returncode": proc.returncode,
                "save_dir": spec["save_dir"],
            }
            progress.append(record)
            write_progress(progress)
            raise SystemExit(f"RUN_FAIL index={idx} returncode={proc.returncode}")

        metrics = load_json(save_dir / "metrics.json")
        record = metric_record_from_metrics(idx, spec, metrics)
        progress.append(record)
        append_projection(record, progress)
        metrics["cumulative_projected_total_days"] = record["cumulative_projected_total_days"]
        metrics["gate_status"] = record["gate_status"]
        write_json(save_dir / "metrics.json", metrics)
        write_progress(progress)
        print(
            f"RUN_DONE {idx}/{len(specs)} elapsed={record['wall_time_s']:.3f}s "
            f"projected_days={record['cumulative_projected_total_days']:.4f} gate={record['gate_status']}",
            flush=True,
        )
        if float(record["cumulative_projected_total_days"]) > 6.0:
            raise SystemExit(
                f"HARD_STOP_WALL_TIME_GATE run={idx} projected_days={record['cumulative_projected_total_days']:.4f}"
            )

    write_progress(progress)
    print(f"PHASE4_MATRIX_INVOCATION_DONE completed_records={len(progress)} executed={executed_this_invocation}", flush=True)


if __name__ == "__main__":
    main()
