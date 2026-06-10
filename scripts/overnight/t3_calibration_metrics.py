#!/usr/bin/env python3
"""Compute ECE/Brier calibration metrics for existing classification checkpoints."""

from __future__ import annotations
import os

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from overnight_common import (
    CLASSIFICATION_DATASETS,
    ENV_BY_BACKBONE,
    METHODS,
    T3_DIR,
    conda_command,
    ensure_output_dirs,
    md_table,
    read_json,
    write_csv,
    write_json,
)


def checkpoint_path(backbone: str, dataset: str, method: str, protocol: str, seed: int) -> Path:
    if backbone == "cmpnn":
        suffix = f"scaffold_seed{seed}" if protocol == "scaffold" else f"seed{seed}"
        return Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve() / "dumped" / f"baseline_{dataset}_{method}_{suffix}" / "run_0" / "model_0" / "model.pt"
    suffix = f"{backbone}_scaffold_seed{seed}" if protocol == "scaffold" else f"{backbone}_seed{seed}"
    return Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve() / "dumped" / f"baseline_{dataset}_{method}_{suffix}" / "model.pt"


def worker_command(row: Dict[str, Any], out_path: Path) -> str:
    args = [
        "python",
        "scripts/overnight/t3_eval_worker.py",
        "--backbone",
        row["backbone"],
        "--dataset",
        row["dataset"],
        "--method",
        row["method"],
        "--protocol",
        row["protocol"],
        "--seed",
        str(row["seed"]),
        "--checkpoint",
        str(row["checkpoint"]),
        "--out",
        str(out_path),
    ]
    env = "kapt" if row["backbone"] == "cmpnn" else ENV_BY_BACKBONE[row["backbone"]]
    import shlex

    return conda_command(env, " ".join(shlex.quote(str(a)) for a in args), hide_cuda=True)


def specs() -> List[Dict[str, Any]]:
    out = []
    for dataset in CLASSIFICATION_DATASETS:
        for backbone in ["cmpnn", "chemberta2", "molformer_c3"]:
            for method in METHODS:
                for protocol in ["scaffold", "random"]:
                    for seed in [0, 1, 2]:
                        ckpt = checkpoint_path(backbone, dataset, method, protocol, seed)
                        out.append(
                            {
                                "dataset": dataset,
                                "backbone": backbone,
                                "method": method,
                                "protocol": protocol,
                                "seed": seed,
                                "checkpoint": ckpt,
                            }
                        )
    return out


def plot_reliability(rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (T3_DIR / "t3_reliability_diagrams" / "PLOT_IMPORT_FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "PASS":
            by_dataset[row["dataset"]].append(row)

    for dataset, dataset_rows in by_dataset.items():
        plt.figure(figsize=(5.5, 5.0))
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        for method in METHODS:
            weighted: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "conf": 0.0, "acc": 0.0})
            for row in dataset_rows:
                if row["method"] != method:
                    continue
                for item in row.get("bins", []):
                    n = int(item.get("n") or 0)
                    if n <= 0 or item.get("confidence") == "" or item.get("accuracy") == "":
                        continue
                    bucket = weighted[int(item["bin"])]
                    bucket["n"] += n
                    bucket["conf"] += n * float(item["confidence"])
                    bucket["acc"] += n * float(item["accuracy"])
            xs, ys = [], []
            for bin_id in sorted(weighted):
                bucket = weighted[bin_id]
                if bucket["n"] > 0:
                    xs.append(bucket["conf"] / bucket["n"])
                    ys.append(bucket["acc"] / bucket["n"])
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical positive rate")
        plt.title(f"{dataset.upper()} reliability")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(T3_DIR / "t3_reliability_diagrams" / f"{dataset}_reliability.png", dpi=160)
        plt.close()


def main() -> None:
    ensure_output_dirs()
    start = time.time()
    worker_dir = T3_DIR / "worker_outputs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, spec in enumerate(specs(), start=1):
        out_path = worker_dir / f"{index:03d}_{spec['dataset']}_{spec['backbone']}_{spec['method']}_{spec['protocol']}_seed{spec['seed']}.json"
        if out_path.exists():
            cached = read_json(out_path)
            if cached.get("status") == "PASS":
                rows.append(cached)
                print(f"T3 {index}/{len(specs())} {spec['dataset']} {spec['backbone']} {spec['method']} {spec['protocol']} seed={spec['seed']} status=PASS cached", flush=True)
                continue
        if not spec["checkpoint"].exists():
            result = {
                **{k: v for k, v in spec.items() if k != "checkpoint"},
                "checkpoint": str(spec["checkpoint"]),
                "n_test": "",
                "ECE_10bin": "",
                "BrierScore": "",
                "status": "FAIL",
                "error": "checkpoint missing",
                "bins": [],
            }
            write_json(out_path, result)
        else:
            command = worker_command(spec, out_path)
            proc = subprocess.run(["bash", "-lc", command], cwd=os.environ.get("BAM_REPO_ROOT", str(Path(__file__).resolve().parents[2])), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if out_path.exists():
                result = read_json(out_path)
                result["worker_returncode"] = proc.returncode
            else:
                result = {
                    **{k: v for k, v in spec.items() if k != "checkpoint"},
                    "checkpoint": str(spec["checkpoint"]),
                    "n_test": "",
                    "ECE_10bin": "",
                    "BrierScore": "",
                    "status": "FAIL",
                    "error": "worker produced no output",
                    "worker_output": proc.stdout[-4000:],
                    "bins": [],
                    "worker_returncode": proc.returncode,
                }
        rows.append(result)
        if result.get("status") != "PASS":
            failures.append(result)
        print(f"T3 {index}/{len(specs())} {spec['dataset']} {spec['backbone']} {spec['method']} {spec['protocol']} seed={spec['seed']} status={result.get('status')}", flush=True)

    fields = ["dataset", "backbone", "method", "protocol", "seed", "n_test", "ECE_10bin", "BrierScore", "status"]
    write_csv(T3_DIR / "t3_ece_results.csv", rows, fields)
    write_json(T3_DIR / "t3_ece_results.json", rows)
    plot_reliability(rows)

    lines = ["# T3 Calibration Runtime", "", f"wall_clock_sec: {time.time() - start:.2f}", f"rows: {len(rows)}", f"failures: {len(failures)}", ""]
    if failures:
        lines.extend(md_table(failures[:50], ["dataset", "backbone", "method", "protocol", "seed", "status", "error"]))
    (T3_DIR / "t3_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T3 done rows={len(rows)} failures={len(failures)} wall_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
