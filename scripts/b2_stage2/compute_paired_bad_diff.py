import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from b2_phase4_common import BACKBONE_ORDER, DATASET_ORDER, REPO_ROOT, SEEDS, load_json, output_dir, write_json


OUT_JSON = REPO_ROOT / "docs/stage3/b2_phase4_paired_bad_diff_final.json"
OUT_CSV = REPO_ROOT / "docs/stage3/b2_phase4_paired_bad_diff_final.csv"
LEGACY_OUT_JSON = REPO_ROOT / "docs/stage3/b2_phase4_paired_bad_diff.json"
LEGACY_OUT_CSV = REPO_ROOT / "docs/stage3/b2_phase4_paired_bad_diff.csv"
REGRESSION_DATASETS = {"freesolv", "esol", "lipo"}


def metric_for(dataset: str, method: str, backbone: str, seed: int) -> Dict[str, object]:
    path = output_dir(dataset, method, backbone, seed) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_json(path)


def bad_diff(dataset: str, lora_metric: float, dora_metric: float) -> float:
    if dataset in REGRESSION_DATASETS:
        return dora_metric - lora_metric
    return lora_metric - dora_metric


def task_type_for(dataset: str) -> str:
    return "regression" if dataset in REGRESSION_DATASETS else "classification"


def summarize(values: List[float]) -> Dict[str, object]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for backbone in BACKBONE_ORDER:
        for dataset in DATASET_ORDER:
            for seed in SEEDS:
                lora = metric_for(dataset, "lora", backbone, seed)
                dora = metric_for(dataset, "dora", backbone, seed)
                diff = bad_diff(dataset, float(lora["final_metric"]), float(dora["final_metric"]))
                rows.append(
                    {
                        "backbone": backbone,
                        "dataset": dataset,
                        "seed": seed,
                        "task_type": task_type_for(dataset),
                        "metric_type": lora["metric_type"],
                        "lora_metric": lora["final_metric"],
                        "dora_metric": dora["final_metric"],
                        "paired_bad_diff": diff,
                    }
                )
                print(f"{backbone} {dataset} seed{seed}: bad_diff={diff:.8f}", flush=True)
    summary: Dict[str, Dict[str, object]] = {}
    by_dataset: Dict[str, Dict[str, Dict[str, object]]] = {}
    by_task_type: Dict[str, Dict[str, Dict[str, object]]] = {}
    for backbone in BACKBONE_ORDER:
        values = [float(row["paired_bad_diff"]) for row in rows if row["backbone"] == backbone]
        summary[backbone] = summarize(values)
        by_dataset[backbone] = {}
        for dataset in DATASET_ORDER:
            dataset_values = [
                float(row["paired_bad_diff"])
                for row in rows
                if row["backbone"] == backbone and row["dataset"] == dataset
            ]
            by_dataset[backbone][dataset] = summarize(dataset_values)
        by_task_type[backbone] = {}
        for task_type in ("regression", "classification"):
            task_values = [
                float(row["paired_bad_diff"])
                for row in rows
                if row["backbone"] == backbone and row["task_type"] == task_type
            ]
            by_task_type[backbone][task_type] = summarize(task_values)

    payload = {
        "direction_convention": {
            "classification": "AUC_LoRA - AUC_DoRA; negative = DoRA better",
            "regression": "RMSE_DoRA - RMSE_LoRA; negative = DoRA better",
        },
        "rows": rows,
        "summary": summary,
        "by_dataset": by_dataset,
        "by_task_type": by_task_type,
    }
    write_json(OUT_JSON, payload)
    write_json(LEGACY_OUT_JSON, payload)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backbone",
        "dataset",
        "seed",
        "task_type",
        "metric_type",
        "lora_metric",
        "dora_metric",
        "paired_bad_diff",
    ]
    for csv_path in (OUT_CSV, LEGACY_OUT_CSV):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print("PAIRED_BAD_DIFF_SUMMARY " + str(summary))
    print("PAIRED_BAD_DIFF_BY_DATASET " + str(by_dataset))
    print("PAIRED_BAD_DIFF_BY_TASK_TYPE " + str(by_task_type))


if __name__ == "__main__":
    main()
