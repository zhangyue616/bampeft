#!/usr/bin/env python3
"""Classical sklearn baselines on Morgan fingerprints."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from overnight_common import DATASET_INFO, DATASETS, N5_SPLIT_DIR, REPO, T6_DIR, ensure_output_dirs, read_indices, read_json, write_csv, write_json


MODELS = ["RandomForest", "GradientBoosting"]
SEEDS = [0, 1, 2]
PROTOCOLS = ["scaffold", "random"]


def read_dataset(dataset: str) -> Tuple[List[str], np.ndarray, List[str]]:
    path = REPO / DATASET_INFO[dataset]["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    smiles_col = fields[0]
    target_cols = fields[1:]
    smiles = [row[smiles_col] for row in rows]
    values = []
    for row in rows:
        vals = []
        for col in target_cols:
            value = row[col]
            vals.append(float(value) if value != "" else float("nan"))
        values.append(vals)
    return smiles, np.asarray(values, dtype=float), target_cols


def morgan(smiles: List[str]) -> np.ndarray:
    fps = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def split_indices(dataset: str, protocol: str, seed: int, n_items: int) -> Dict[str, List[int]]:
    if protocol == "scaffold":
        payload = read_json(N5_SPLIT_DIR / f"{dataset}_scaffold_seed{seed}.json")
        return {key: [int(i) for i in value] for key, value in payload["splits"].items()}
    split_dir = REPO / "dumped" / f"baseline_{dataset}_fullft_chemberta2_random_seed{seed}" / "split"
    if split_dir.exists():
        return {
            "train": read_indices(split_dir / "train_indices.txt"),
            "val": read_indices(split_dir / "val_indices.txt"),
            "test": read_indices(split_dir / "test_indices.txt"),
        }
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    train_size = int(0.8 * n_items)
    train_val_size = int(0.9 * n_items)
    return {"train": indices[:train_size].tolist(), "val": indices[train_size:train_val_size].tolist(), "test": indices[train_val_size:].tolist()}


def build_model(model_name: str, task: str, seed: int):
    if model_name == "RandomForest":
        if task == "regression":
            return RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    if task == "regression":
        return GradientBoostingRegressor(n_estimators=500, random_state=seed)
    return GradientBoostingClassifier(n_estimators=500, random_state=seed)


def eval_regression(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> float:
    mask_train = np.isfinite(y_train[:, 0])
    mask_test = np.isfinite(y_test[:, 0])
    model = build_model(model_name, "regression", seed)
    model.fit(x_train[mask_train], y_train[mask_train, 0])
    pred = model.predict(x_test[mask_test])
    return float(math.sqrt(mean_squared_error(y_test[mask_test, 0], pred)))


def eval_classification(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> Tuple[float, int]:
    aucs = []
    for task_idx in range(y_train.shape[1]):
        train_mask = np.isfinite(y_train[:, task_idx])
        test_mask = np.isfinite(y_test[:, task_idx])
        ytr = y_train[train_mask, task_idx].astype(int)
        yte = y_test[test_mask, task_idx].astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        model = build_model(model_name, "classification", seed)
        model.fit(x_train[train_mask], ytr)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test[test_mask])[:, 1]
        else:
            scores = model.decision_function(x_test[test_mask])
        aucs.append(float(roc_auc_score(yte, scores)))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def paper_best_peft() -> Dict[str, Dict[str, Any]]:
    inputs = [
        REPO / "docs/stage3/scaffold_split_extension_results.json",
        REPO / "docs/stage3/z1c_seed_extension_results_table.json",
        REPO / "docs/stage3/z1_seed_extension_results.json",
        REPO / "docs/stage3/n5_rank_sensitivity_results_table.json",
    ]
    best: Dict[str, Dict[str, Any]] = {}
    for path in inputs:
        if not path.exists():
            continue
        payload = read_json(path)
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if row.get("status") not in ("PASS", None, ""):
                continue
            method = row.get("method")
            if method not in {"lora", "dora"}:
                continue
            dataset = row.get("dataset")
            metric_type = row.get("metric_type") or DATASET_INFO.get(dataset, {}).get("metric")
            metric = row.get("final_metric", row.get("metric"))
            if dataset not in DATASET_INFO or metric in (None, ""):
                continue
            try:
                metric = float(metric)
            except Exception:
                continue
            current = best.get(dataset)
            better = current is None or (metric < current["metric"] if metric_type == "rmse" else metric > current["metric"])
            if better:
                best[dataset] = {"metric": metric, "metric_type": metric_type, "source": str(path.relative_to(REPO)), "method": method}
    return best


def main() -> None:
    ensure_output_dirs()
    start = time.time()
    best = paper_best_peft()
    rows = []
    failures = []
    conflict_rows = []
    features_cache: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    total = len(DATASETS) * len(MODELS) * len(SEEDS) * len(PROTOCOLS)
    index = 0
    for dataset in DATASETS:
        smiles, targets, target_cols = read_dataset(dataset)
        x = morgan(smiles)
        features_cache[dataset] = (x, targets, target_cols)
        task = DATASET_INFO[dataset]["dataset_type"]
        metric_type = DATASET_INFO[dataset]["metric"]
        for model_name in MODELS:
            for protocol in PROTOCOLS:
                for seed in SEEDS:
                    index += 1
                    row: Dict[str, Any] = {
                        "dataset": dataset,
                        "model": model_name,
                        "protocol": protocol,
                        "seed": seed,
                        "task_type": task,
                        "metric_type": metric_type,
                        "metric": "",
                        "n_tasks_evaluated": "",
                        "status": "FAIL",
                        "paper_best_peft_metric": best.get(dataset, {}).get("metric", ""),
                        "paper_best_peft_source": best.get(dataset, {}).get("source", ""),
                        "sklearn_outperforms_peft_by_gt_0_05": False,
                        "outperformance_margin": "",
                        "error": "",
                    }
                    try:
                        splits = split_indices(dataset, protocol, seed, len(smiles))
                        train_idx = np.asarray(splits["train"], dtype=int)
                        test_idx = np.asarray(splits["test"], dtype=int)
                        if task == "regression":
                            metric = eval_regression(model_name, x[train_idx], targets[train_idx], x[test_idx], targets[test_idx], seed)
                            n_tasks = 1
                        else:
                            metric, n_tasks = eval_classification(model_name, x[train_idx], targets[train_idx], x[test_idx], targets[test_idx], seed)
                        row["metric"] = metric
                        row["n_tasks_evaluated"] = n_tasks
                        row["status"] = "PASS" if math.isfinite(metric) else "FAIL"
                        if row["status"] != "PASS":
                            row["error"] = "non-finite metric"
                        if row["status"] == "PASS" and dataset in best:
                            peft = float(best[dataset]["metric"])
                            if metric_type == "rmse":
                                margin = peft - metric
                            else:
                                margin = metric - peft
                            row["outperformance_margin"] = margin
                            if margin > 0.05:
                                row["sklearn_outperforms_peft_by_gt_0_05"] = True
                                conflict_rows.append(row.copy())
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    if row["status"] != "PASS":
                        failures.append(row)
                    print(f"T6 {index}/{total} {dataset} {model_name} {protocol} seed={seed} status={row['status']} metric={row['metric']}", flush=True)
    fields = [
        "dataset",
        "model",
        "protocol",
        "seed",
        "task_type",
        "metric_type",
        "metric",
        "n_tasks_evaluated",
        "status",
        "paper_best_peft_metric",
        "paper_best_peft_source",
        "sklearn_outperforms_peft_by_gt_0_05",
        "outperformance_margin",
        "error",
    ]
    write_csv(T6_DIR / "t6_sklearn_baselines_results.csv", rows, fields)
    write_json(T6_DIR / "t6_sklearn_baselines_results.json", rows)
    lines = [
        "# T6 Failures And Conflict Notes",
        "",
        "Dataset scope uses code-provenance paper benchmark datasets: FreeSolv, ESOL, Lipo, BACE, BBBP, SIDER.",
        "The commission text mentioned ClinTox, but N5/stage3 provenance uses BACE; no ClinTox paper PEFT baseline was inferred.",
        "",
        f"Failures: {len(failures)}",
        f"Sklearn > best PEFT by >0.05 findings: {len(conflict_rows)}",
        f"Wall-clock seconds: {time.time() - start:.2f}",
        "",
    ]
    if conflict_rows:
        lines.append("## P1 Findings Against Inline Claim Block D")
        lines.append("")
        lines.append("| dataset | model | protocol | seed | metric | paper_best_peft_metric | margin | recommendation |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in conflict_rows:
            lines.append(
                f"| {row['dataset']} | {row['model']} | {row['protocol']} | {row['seed']} | {row['metric']} | "
                f"{row['paper_best_peft_metric']} | {row['outperformance_margin']} | review BEFORE Round 55 ChatGPT milestone review |"
            )
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append("| dataset | model | protocol | seed | error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in failures[:100]:
            lines.append(f"| {row['dataset']} | {row['model']} | {row['protocol']} | {row['seed']} | {row['error']} |")
    (T6_DIR / "t6_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T6 done rows={len(rows)} failures={len(failures)} conflicts={len(conflict_rows)} wall_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
