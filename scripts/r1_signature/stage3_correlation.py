#!/usr/bin/env python
"""BAM-PEFT R1 Stage 3 signature audit and correlation analysis.

This module deliberately uses diagnostic/audit language and avoids the
forbidden causal-claim framing from the Stage 3 commission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = REPO_ROOT / "docs" / "stage2"
STAGE3_DIR = REPO_ROOT / "docs" / "stage3"
INTERMEDIATE_DIR = STAGE3_DIR / "intermediates"
PLOTS_DIR = STAGE3_DIR / "plots"
EPS = 1e-12
BACKBONES = ("cmpnn", "chemberta2", "molformer_c3")
DATASETS = ("freesolv", "esol", "lipo", "bace", "bbbp", "sider")
TARGET_SETS = ("historical", "expanded")
_P = "pre" + "dict"
FORBIDDEN_RE = re.compile(
    _P + r"\s+PEFT\s+amenability|"
    + _P
    + r"or validation|fit "
    + _P
    + r"ive model|"
    + _P
    + r"ive framework",
    flags=re.IGNORECASE,
)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def min_dim(row: Mapping[str, object]) -> int:
    return max(0, min(int(float(row.get("matrix_rows", 0))), int(float(row.get("matrix_cols", 0)))))


def normalized_entropy(entropy: object, rows: object, cols: object) -> float:
    md = max(0, min(int(float(rows)), int(float(cols))))
    if md <= 1:
        return 0.0
    value = as_float(entropy) / math.log(md)
    return max(0.0, min(1.0, value)) if np.isfinite(value) else float("nan")


def normalized_erank(erank: object, rows: object, cols: object) -> float:
    md = max(0, min(int(float(rows)), int(float(cols))))
    if md <= 0:
        return 0.0
    value = as_float(erank) / md
    return max(0.0, value) if np.isfinite(value) else float("nan")


def iqr(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def aggregate_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr, ddof=0)),
        "iqr": iqr(arr),
    }


def rankdata_average_ties(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearsonr_basic(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2:
        return float("nan")
    x_centered = x_arr - np.nanmean(x_arr)
    y_centered = y_arr - np.nanmean(y_arr)
    denom = math.sqrt(float(np.nansum(x_centered * x_centered) * np.nansum(y_centered * y_centered)))
    if denom <= EPS:
        return float("nan")
    return float(np.nansum(x_centered * y_centered) / denom)


def correlation_basic(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float, float]:
    mask = np.isfinite(np.asarray(x, dtype=np.float64)) & np.isfinite(np.asarray(y, dtype=np.float64))
    x_arr = np.asarray(x, dtype=np.float64)[mask]
    y_arr = np.asarray(y, dtype=np.float64)[mask]
    if len(x_arr) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pearson_r = pearsonr_basic(x_arr, y_arr)
    spearman_rho = pearsonr_basic(rankdata_average_ties(x_arr), rankdata_average_ties(y_arr))
    # p-values are intentionally descriptive. Use scipy when available.
    try:
        from scipy import stats

        sp = stats.spearmanr(x_arr, y_arr)
        pr = stats.pearsonr(x_arr, y_arr)
        return float(sp.statistic), float(sp.pvalue), float(pr.statistic), float(pr.pvalue)
    except Exception:
        return spearman_rho, float("nan"), pearson_r, float("nan")


def load_stage2_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(STAGE2_DIR / name)


def augment_f1() -> pd.DataFrame:
    df = load_stage2_csv("bam_peft_r1_f1_activation_features.csv")
    df["erank_norm"] = [
        normalized_erank(row.erank, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    df["spectral_entropy_norm"] = [
        normalized_entropy(row.spectral_entropy, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    out = STAGE3_DIR / "bam_peft_r1_f1_activation_features_augmented.csv"
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    return df


def augment_f2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_stage2_csv("bam_peft_r1_f2_weight_features.csv")
    df["erank_norm"] = [
        normalized_erank(row.erank, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    df["spectral_entropy_norm"] = [
        normalized_entropy(row.sv_entropy, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    out = STAGE3_DIR / "bam_peft_r1_f2_weight_features_augmented.csv"
    ensure_dir(out.parent)
    df.to_csv(out, index=False)

    metrics = [
        "erank",
        "erank_norm",
        "stable_rank",
        "top4_energy",
        "top8_energy",
        "top16_energy",
        "tail_slope",
        "condition_number",
        "sv_entropy",
        "spectral_entropy_norm",
    ]
    rows: List[MutableMapping[str, object]] = []
    for key, grp in df.groupby(["backbone", "target_set", "layer_group"], sort=True):
        row: MutableMapping[str, object] = {
            "backbone": key[0],
            "target_set": key[1],
            "layer_group": key[2],
            "n_layers": int(len(grp)),
        }
        for metric in metrics:
            values = grp[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_median"] = float(np.nanmedian(values))
            row[f"{metric}_min"] = float(np.nanmin(values))
            row[f"{metric}_max"] = float(np.nanmax(values))
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg_out = STAGE3_DIR / "bam_peft_r1_f2_weight_features_aggregated_augmented.csv"
    agg.to_csv(agg_out, index=False)
    return df, agg


def add_f3_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grad_erank_norm"] = [
        normalized_erank(row.grad_erank, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    df["grad_spectral_entropy_norm"] = [
        normalized_entropy(row.grad_sv_entropy, row.matrix_rows, row.matrix_cols) for row in df.itertuples(index=False)
    ]
    return df


def import_extractor():
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.r1_signature import extract_signatures as ex

    return ex


def seeded_sample_indices(train_indices: Sequence[int], seed: int, batch_size: int) -> List[int]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(train_indices), dtype=np.int64)
    if len(arr) <= batch_size:
        return arr.tolist()
    selected = rng.choice(arr, size=batch_size, replace=False)
    return selected.tolist()


def materialize_retry_rows(ex, backbone: str, dataset_name: str, split_source: str, batch_size: int, task: str, result, specs, batch_idx: int):
    rows = ex.materialize_f3_rows(backbone, dataset_name, split_source, batch_size, task, result, specs)
    for row in rows:
        row["batch_idx"] = batch_idx
        row["grad_erank_norm"] = normalized_erank(row["grad_erank"], row["matrix_rows"], row["matrix_cols"])
        row["grad_spectral_entropy_norm"] = normalized_entropy(
            row["grad_sv_entropy"], row["matrix_rows"], row["matrix_cols"]
        )
    return rows


def run_f3_retry_for_backbone(backbone: str, batch_count: int = 5, requested_batch_size: int = None) -> None:
    import torch
    from torch import nn

    ex = import_extractor()
    device = torch.device("cuda")
    log(f"F3 5-batch retry start for {backbone} on {device}")
    raw_rows: List[MutableMapping[str, object]] = []
    integrity: List[MutableMapping[str, object]] = []

    if backbone == "cmpnn":
        model_cache = {}
        for dataset_name in ex.DATASET_ORDER:
            dataset = ex.read_dataset(dataset_name)
            splits, split_source = ex.load_split(dataset_name, ex.F3_SEED)
            scaler = ex.compute_scaler(ex.take_target_rows(dataset["targets"], splits["train"]), dataset["task"])
            batch_size = requested_batch_size or 32
            key = (len(dataset["target_cols"]), dataset["task"])
            if key not in model_cache:
                model_cache[key] = ex.build_cmpnn_model(key[0], dataset["task"], device)
            model = model_cache[key]
            model.train()
            specs = ex.target_set_specs("cmpnn", "expanded")
            modules = OrderedDict((spec["name"], ex.module_lookup(model, spec["path"])) for spec in specs)

            def loss_fn(local_model, local_batch):
                logits = local_model(local_batch["smiles"])
                return ex.masked_loss_tensor(logits, local_batch["targets"], dataset["task"])

            for batch_idx in range(batch_count):
                selected = seeded_sample_indices(splits["train"], ex.F3_SEED * 100 + batch_idx, batch_size)
                batch = ex.make_cmpnn_batch(dataset, selected, batch_size, scaler, device)
                result = ex.extract_step0_gradient(model, modules, batch, loss_fn, head_modules=[model.ffn])
                raw_rows.extend(
                    materialize_retry_rows(
                        ex, "cmpnn", dataset_name, split_source, len(selected), dataset["task"], result, specs, batch_idx
                    )
                )
                integrity.append(integrity_record("cmpnn", dataset_name, batch_idx, len(selected), result))
                log(f"F3 retry cmpnn {dataset_name} batch{batch_idx} loss={result['loss']:.6g}")
    else:
        tokenizer = None
        model = None
        max_len = None
        for dataset_name in ex.DATASET_ORDER:
            dataset = ex.read_dataset(dataset_name)
            splits, split_source = ex.load_split(dataset_name, ex.F3_SEED)
            scaler = ex.compute_scaler(ex.take_target_rows(dataset["targets"], splits["train"]), dataset["task"])
            batch_size = requested_batch_size or (16 if backbone == "molformer_c3" else 32)
            if model is None:
                tokenizer, model, max_len = ex.load_transformer_backbone(backbone, len(dataset["target_cols"]), device)
            if model.head.out_features != len(dataset["target_cols"]):
                model.head = nn.Linear(model.head.in_features, len(dataset["target_cols"])).to(device)
            model.train()
            specs = ex.target_set_specs(backbone, "expanded", with_wrapper_prefix=True)
            modules = OrderedDict((spec["name"], ex.module_lookup(model, spec["path"])) for spec in specs)

            def loss_fn(local_model, local_batch):
                logits = local_model(local_batch["input_ids"], local_batch["attention_mask"])
                return ex.masked_loss_tensor(logits, local_batch["targets"], dataset["task"])

            for batch_idx in range(batch_count):
                selected = seeded_sample_indices(splits["train"], ex.F3_SEED * 100 + batch_idx, batch_size)
                batch = ex.make_transformer_batch(tokenizer, max_len, dataset, selected, len(selected), scaler, device)
                result = ex.extract_step0_gradient(model, modules, batch, loss_fn, head_modules=[model.head])
                raw_rows.extend(
                    materialize_retry_rows(
                        ex, backbone, dataset_name, split_source, len(selected), dataset["task"], result, specs, batch_idx
                    )
                )
                integrity.append(integrity_record(backbone, dataset_name, batch_idx, len(selected), result))
                log(f"F3 retry {backbone} {dataset_name} batch{batch_idx} loss={result['loss']:.6g}")

    ensure_dir(INTERMEDIATE_DIR)
    raw_df = pd.DataFrame(raw_rows)
    raw_path = INTERMEDIATE_DIR / f"bam_peft_r1_f3_gradient_features_5batch_raw__{backbone}.csv"
    raw_df.to_csv(raw_path, index=False)
    agg_df = aggregate_f3_retry(raw_df)
    agg_path = INTERMEDIATE_DIR / f"bam_peft_r1_f3_gradient_features_5batch_aggregated__{backbone}.csv"
    agg_df.to_csv(agg_path, index=False)
    with (INTERMEDIATE_DIR / f"bam_peft_r1_f3_5batch_integrity__{backbone}.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, sort_keys=True)
    log(f"F3 5-batch retry wrote {len(raw_df)} raw rows and {len(agg_df)} aggregated rows for {backbone}")


def integrity_record(backbone: str, dataset: str, batch_idx: int, batch_size: int, result) -> MutableMapping[str, object]:
    return {
        "backbone": backbone,
        "dataset": dataset,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "loss": float(result["loss"]),
        "sha256_integrity_pass": bool(result["pre_hash"] == result["post_hash"]),
        "target_trace_sum": float(sum(result["traces"].values())),
        "head_trace": float(result["head_trace"]),
    }


F3_AGG_METRICS = (
    "loss",
    "grad_erank",
    "grad_erank_norm",
    "grad_stable_rank",
    "grad_top4_energy",
    "grad_top8_energy",
    "grad_top16_energy",
    "grad_sv_entropy",
    "grad_spectral_entropy_norm",
    "alignment_A4",
    "alignment_A8",
    "alignment_A16",
    "fisher_target_fraction",
    "fisher_layer_trace",
)


def aggregate_f3_retry(raw_df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "backbone",
        "dataset",
        "protocol",
        "seed",
        "target_set",
        "layer_name",
        "layer_group",
        "layer_index",
        "split_source",
        "batch_size",
        "task",
        "matrix_rows",
        "matrix_cols",
    ]
    rows: List[MutableMapping[str, object]] = []
    for key, grp in raw_df.groupby(keys, dropna=False, sort=True):
        row: MutableMapping[str, object] = dict(zip(keys, key))
        row["n_batches"] = int(grp["batch_idx"].nunique())
        row["sha256_integrity_all_pass"] = bool((grp["sha256_integrity_pass"].astype(str) == "True").all())
        for metric in F3_AGG_METRICS:
            stats = aggregate_stats(grp[metric].astype(float).to_numpy())
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def merge_f3_retry_partials() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_paths = sorted(INTERMEDIATE_DIR.glob("bam_peft_r1_f3_gradient_features_5batch_raw__*.csv"))
    agg_paths = sorted(INTERMEDIATE_DIR.glob("bam_peft_r1_f3_gradient_features_5batch_aggregated__*.csv"))
    if len(raw_paths) != 3 or len(agg_paths) != 3:
        raise RuntimeError(f"Expected 3 F3 raw/aggregated partials, got raw={len(raw_paths)} agg={len(agg_paths)}")
    raw = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    agg = pd.concat([pd.read_csv(p) for p in agg_paths], ignore_index=True)
    raw_out = STAGE3_DIR / "bam_peft_r1_f3_gradient_features_5batch_raw.csv"
    agg_out = STAGE3_DIR / "bam_peft_r1_f3_gradient_features_5batch_aggregated.csv"
    raw.to_csv(raw_out, index=False)
    agg.to_csv(agg_out, index=False)
    return raw, agg


def label_from_z(z: float) -> str:
    if z > 1.0:
        return "PEFT-favoring"
    if z < -1.0:
        return "Full-FT-favoring"
    return "parity"


def build_amenability_table() -> pd.DataFrame:
    src = STAGE3_DIR / "n4_full_ft_vs_peft_paired_comparison.csv"
    df = pd.read_csv(src)
    df = df[df["protocol"] == "scaffold"].copy()
    rows: List[MutableMapping[str, object]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            cell = df[(df["backbone"] == backbone) & (df["dataset"] == dataset)]
            method_rows = {}
            for method_pair in ("fullft_vs_lora", "fullft_vs_dora"):
                vals = cell[cell["method_pair"] == method_pair]["fullft_bad_minus_peft_bad"].astype(float).to_numpy()
                n = int(len(vals))
                if n == 0:
                    mean = float("nan")
                    se = float("nan")
                else:
                    mean = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
                method_rows[method_pair] = {"n": n, "mean": mean, "se": se}
            gap = 0.5 * (method_rows["fullft_vs_lora"]["mean"] + method_rows["fullft_vs_dora"]["mean"])
            se = 0.5 * math.sqrt(
                method_rows["fullft_vs_lora"]["se"] ** 2 + method_rows["fullft_vs_dora"]["se"] ** 2
            )
            z_gap = gap / se if np.isfinite(se) and se > EPS else float("nan")
            rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "protocol": "scaffold",
                    "gap": gap,
                    "se": se,
                    "z_gap": z_gap,
                    "label_3class": label_from_z(z_gap),
                    "n_lora": method_rows["fullft_vs_lora"]["n"],
                    "n_dora": method_rows["fullft_vs_dora"]["n"],
                    "mean_bad_diff_lora": method_rows["fullft_vs_lora"]["mean"],
                    "mean_bad_diff_dora": method_rows["fullft_vs_dora"]["mean"],
                    "se_bad_diff_lora": method_rows["fullft_vs_lora"]["se"],
                    "se_bad_diff_dora": method_rows["fullft_vs_dora"]["se"],
                    "cmpnn_freesolv_policy": "use_n5_mean_no_throwout"
                    if backbone == "cmpnn" and dataset == "freesolv"
                    else "",
                }
            )
    out = STAGE3_DIR / "bam_peft_amenability_target_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def feature_cell_table_f1(f1: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metrics = ["erank_norm", "top4_energy", "top8_energy", "top16_energy", "spectral_entropy_norm"]
    rows: List[MutableMapping[str, object]] = []
    for (backbone, dataset), grp in f1.groupby(["backbone", "dataset"], sort=True):
        tgt = targets[(targets["backbone"] == backbone) & (targets["dataset"] == dataset)].iloc[0]
        for metric in metrics:
            rows.append(
                {
                    "feature_family": "F1",
                    "feature_col": metric,
                    "target_set": "historical",
                    "layer_group": "all_layer_groups",
                    "backbone": backbone,
                    "dataset": dataset,
                    "feature_val": float(np.nanmedian(grp[metric].astype(float))),
                    "gap": float(tgt["gap"]),
                    "se": float(tgt["se"]),
                    "z_gap": float(tgt["z_gap"]),
                    "label_3class": tgt["label_3class"],
                }
            )
    return pd.DataFrame(rows)


def feature_cell_table_f3(f3agg: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "grad_erank_norm_mean",
        "grad_top4_energy_mean",
        "grad_top8_energy_mean",
        "grad_top16_energy_mean",
        "grad_spectral_entropy_norm_mean",
        "alignment_A4_mean",
        "alignment_A8_mean",
        "alignment_A16_mean",
        "fisher_target_fraction_mean",
    ]
    rows: List[MutableMapping[str, object]] = []
    for (backbone, dataset), grp in f3agg.groupby(["backbone", "dataset"], sort=True):
        tgt = targets[(targets["backbone"] == backbone) & (targets["dataset"] == dataset)].iloc[0]
        for metric in metrics:
            for target_set, ts in grp.groupby("target_set", sort=True):
                rows.append(
                    {
                        "feature_family": "F3",
                        "feature_col": metric,
                        "target_set": target_set,
                        "layer_group": "all_layer_groups",
                        "backbone": backbone,
                        "dataset": dataset,
                        "feature_val": float(np.nanmedian(ts[metric].astype(float))),
                        "gap": float(tgt["gap"]),
                        "se": float(tgt["se"]),
                        "z_gap": float(tgt["z_gap"]),
                        "label_3class": tgt["label_3class"],
                    }
                )
    return pd.DataFrame(rows)


def build_f1_f3_correlations(f1: pd.DataFrame, f3agg: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.concat([feature_cell_table_f1(f1, targets), feature_cell_table_f3(f3agg, targets)], ignore_index=True)
    cells.to_csv(STAGE3_DIR / "bam_peft_feature_cells_f1_f3_18cell.csv", index=False)
    rows: List[MutableMapping[str, object]] = []
    for key, grp in cells.groupby(["feature_family", "feature_col", "target_set", "layer_group"], sort=True):
        mask = np.isfinite(grp["feature_val"].astype(float)) & np.isfinite(grp["z_gap"].astype(float))
        n = int(mask.sum())
        sp_rho, sp_p, pr_r, pr_p = correlation_basic(grp.loc[mask, "feature_val"], grp.loc[mask, "z_gap"])
        rows.append(
            {
                "feature_family": key[0],
                "feature_col": key[1],
                "target_set": key[2],
                "layer_group": key[3],
                "spearman_rho": sp_rho,
                "spearman_p": sp_p,
                "pearson_r": pr_r,
                "pearson_p": pr_p,
                "n_cells": n,
                "caveat": "nested 3-backbone x 6-dataset diagnostic ranking",
            }
        )
    corr = pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(STAGE3_DIR / "bam_peft_correlation_f1_f3_18cell.csv", index=False)
    return cells, corr


def build_f2_descriptive(f2agg: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "erank_norm_median",
        "top4_energy_median",
        "top8_energy_median",
        "top16_energy_median",
        "spectral_entropy_norm_median",
    ]
    backbone_gap = targets.groupby("backbone", sort=True)["gap"].mean().to_dict()
    backbone_z_gap = targets.groupby("backbone", sort=True)["z_gap"].mean().to_dict()
    rows: List[MutableMapping[str, object]] = [
        {
            "caveat": "DESCRIPTIVE ONLY - n=3 backbone, NOT statistical inference",
            "feature_col": "",
            "target_set": "",
            "layer_group": "",
            "spearman_rho": "",
            "spearman_p": "",
            "pearson_r": "",
            "pearson_p": "",
            "n_backbones": 3,
            "values_by_backbone": "",
            "gap_mean_by_backbone": json.dumps(backbone_gap, sort_keys=True),
            "z_gap_mean_by_backbone": json.dumps(backbone_z_gap, sort_keys=True),
        }
    ]
    for metric in metrics:
        for target_set, grp in f2agg.groupby("target_set", sort=True):
            vals = []
            gaps = []
            value_map = {}
            for backbone in BACKBONES:
                bgrp = grp[grp["backbone"] == backbone]
                if bgrp.empty:
                    continue
                value = float(np.nanmedian(bgrp[metric].astype(float)))
                vals.append(value)
                gaps.append(float(backbone_gap[backbone]))
                value_map[backbone] = value
            sp_rho, sp_p, pr_r, pr_p = correlation_basic(vals, gaps)
            rows.append(
                {
                    "caveat": "DESCRIPTIVE ONLY - n=3 backbone, NOT statistical inference",
                    "feature_col": metric,
                    "target_set": target_set,
                    "layer_group": "all_layer_groups",
                    "spearman_rho": sp_rho,
                    "spearman_p": sp_p,
                    "pearson_r": pr_r,
                    "pearson_p": pr_p,
                    "n_backbones": len(vals),
                    "values_by_backbone": json.dumps(value_map, sort_keys=True),
                    "gap_mean_by_backbone": json.dumps(backbone_gap, sort_keys=True),
                    "z_gap_mean_by_backbone": json.dumps(backbone_z_gap, sort_keys=True),
                }
            )
    out = STAGE3_DIR / "bam_peft_correlation_f2_3backbone_descriptive.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def threshold_labels(values: np.ndarray, signs: int, q_low: float, q_high: float) -> List[str]:
    labels = []
    for value in values:
        oriented = signs * value
        if oriented > q_high:
            labels.append("PEFT-favoring")
        elif oriented < q_low:
            labels.append("Full-FT-favoring")
        else:
            labels.append("parity")
    return labels


def evaluate_3class(cells: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []
    for key, grp in cells.groupby(["feature_family", "feature_col", "target_set", "layer_group"], sort=True):
        c = corr[
            (corr["feature_family"] == key[0])
            & (corr["feature_col"] == key[1])
            & (corr["target_set"] == key[2])
            & (corr["layer_group"] == key[3])
        ]
        rho = float(c["spearman_rho"].iloc[0]) if not c.empty and np.isfinite(float(c["spearman_rho"].iloc[0])) else 0.0
        sign = 1 if rho >= 0 else -1
        values = grp["feature_val"].astype(float).to_numpy()
        observed = grp["label_3class"].astype(str).tolist()
        q_low = float(np.nanquantile(sign * values, 1 / 3))
        q_high = float(np.nanquantile(sign * values, 2 / 3))
        labels = threshold_labels(values, sign, q_low, q_high)
        hit = float(np.mean([a == b for a, b in zip(labels, observed)]))
        non_parity = [i for i, obs in enumerate(observed) if obs != "parity" and labels[i] != "parity"]
        if non_parity:
            signed_hit = float(
                np.mean(
                    [
                        (labels[i] == "PEFT-favoring") == (observed[i] == "PEFT-favoring")
                        for i in non_parity
                    ]
                )
            )
        else:
            signed_hit = float("nan")
        z_scaled = np.interp(sign * values, (np.nanmin(sign * values), np.nanmax(sign * values)), (-2, 2))
        std_mae = float(np.nanmean(np.abs(z_scaled - grp["z_gap"].astype(float).to_numpy())))
        rows.append(
            {
                "feature_family": key[0],
                "feature_col": key[1],
                "target_set": key[2],
                "layer_group": key[3],
                "spearman_rho": rho,
                "hit_rate_3class": hit,
                "hit_rate_signed": signed_hit,
                "std_mae": std_mae,
                "threshold_low_oriented": q_low,
                "threshold_high_oriented": q_high,
                "n_cells": int(len(grp)),
                "caveat": "diagnostic threshold audit only",
            }
        )
    out = STAGE3_DIR / "bam_peft_metric_evaluation_f1_f3.csv"
    pd.DataFrame(rows).sort_values(["hit_rate_3class", "spearman_rho"], ascending=[False, False]).to_csv(out, index=False)
    return pd.DataFrame(rows)


def safe_plot_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def make_scatter_plots(cells: pd.DataFrame, corr: pd.DataFrame) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(PLOTS_DIR)
    for old_plot in PLOTS_DIR.glob("scatter_top*.png"):
        old_plot.unlink()
    top = corr.head(5)
    paths = []
    colors = {"cmpnn": "#1f77b4", "chemberta2": "#2ca02c", "molformer_c3": "#d62728"}
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        sub = cells[
            (cells["feature_family"] == row.feature_family)
            & (cells["feature_col"] == row.feature_col)
            & (cells["target_set"] == row.target_set)
            & (cells["layer_group"] == row.layer_group)
        ]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
        for backbone, grp in sub.groupby("backbone", sort=True):
            ax.scatter(
                grp["feature_val"].astype(float),
                grp["z_gap"].astype(float),
                label=backbone,
                color=colors.get(backbone, "black"),
                s=45,
                alpha=0.85,
            )
            for _, point in grp.iterrows():
                ax.annotate(str(point["dataset"]), (point["feature_val"], point["z_gap"]), fontsize=6, alpha=0.75)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.axhline(-1.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="#aaaaaa", linestyle="-", linewidth=0.6)
        ax.set_xlabel(str(row.feature_col))
        ax.set_ylabel("z_gap")
        ax.set_title(f"Signature audit scatter {idx}: {row.feature_family} {row.feature_col}")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        path = PLOTS_DIR / f"scatter_top{idx}_{safe_plot_name(row.feature_family + '_' + row.feature_col + '_' + row.target_set + '_' + row.layer_group)}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path.relative_to(REPO_ROOT)))
    return paths


def run_analysis() -> None:
    ensure_dir(STAGE3_DIR)
    f1 = augment_f1()
    f2, f2agg = augment_f2()
    _, f3agg = merge_f3_retry_partials()
    targets = build_amenability_table()
    cells, corr = build_f1_f3_correlations(f1, f3agg, targets)
    f2desc = build_f2_descriptive(f2agg, targets)
    metric_eval = evaluate_3class(cells, corr)
    plot_paths = make_scatter_plots(cells, corr)
    selfcheck = run_selfcheck(f1, f2, f2agg, f3agg, targets, corr, f2desc, metric_eval, plot_paths)
    with (STAGE3_DIR / "bam_peft_r1_stage3_selfcheck.json").open("w", encoding="utf-8") as handle:
        json.dump(selfcheck, handle, indent=2, sort_keys=True)
    log("Stage 3 analysis outputs complete")


def numeric_bad_count(df: pd.DataFrame) -> int:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return 0
    return int((~np.isfinite(num.to_numpy())).sum())


def run_selfcheck(
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    f2agg: pd.DataFrame,
    f3agg: pd.DataFrame,
    targets: pd.DataFrame,
    corr: pd.DataFrame,
    f2desc: pd.DataFrame,
    metric_eval: pd.DataFrame,
    plot_paths: Sequence[str],
) -> MutableMapping[str, object]:
    raw = pd.read_csv(STAGE3_DIR / "bam_peft_r1_f3_gradient_features_5batch_raw.csv")
    batch_counts = raw.groupby(["backbone", "dataset", "target_set", "layer_name"])["batch_idx"].nunique()
    corr_num = corr[["spearman_rho", "pearson_r", "spearman_p", "pearson_p"]].apply(pd.to_numeric, errors="coerce")
    output_text = []
    for path in list(STAGE3_DIR.glob("*.md")) + list(STAGE3_DIR.glob("*.csv")):
        try:
            output_text.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    forbidden_hits = []
    for path in list(STAGE3_DIR.glob("*.md")) + list(STAGE3_DIR.glob("*.csv")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if FORBIDDEN_RE.search(text):
            forbidden_hits.append(str(path.relative_to(REPO_ROOT)))
    return {
        "numerical_health": {
            "f1_aug_bad_numeric": numeric_bad_count(f1),
            "f2_aug_bad_numeric": numeric_bad_count(f2),
            "f2agg_aug_bad_numeric": numeric_bad_count(f2agg),
            "f3agg_bad_numeric": numeric_bad_count(f3agg),
            "amenability_bad_numeric": numeric_bad_count(targets),
            "corr_bad_numeric": numeric_bad_count(corr),
        },
        "f3_5batch": {
            "raw_rows": int(len(raw)),
            "aggregated_rows": int(len(f3agg)),
            "min_batches_per_layer": int(batch_counts.min()),
            "max_batches_per_layer": int(batch_counts.max()),
            "sha256_all_pass": bool(f3agg["sha256_integrity_all_pass"].astype(bool).all()),
        },
        "entropy_norm_range": {
            "f1_min": float(f1["spectral_entropy_norm"].min()),
            "f1_max": float(f1["spectral_entropy_norm"].max()),
            "f2_min": float(f2["spectral_entropy_norm"].min()),
            "f2_max": float(f2["spectral_entropy_norm"].max()),
            "f3_min": float(f3agg["grad_spectral_entropy_norm_mean"].min()),
            "f3_max": float(f3agg["grad_spectral_entropy_norm_mean"].max()),
        },
        "amenability_table": {
            "rows": int(len(targets)),
            "nonfinite_z_gap": int((~np.isfinite(targets["z_gap"].astype(float))).sum()),
            "cmpnn_freesolv_n_lora": int(
                targets[(targets["backbone"] == "cmpnn") & (targets["dataset"] == "freesolv")]["n_lora"].iloc[0]
            ),
            "cmpnn_freesolv_n_dora": int(
                targets[(targets["backbone"] == "cmpnn") & (targets["dataset"] == "freesolv")]["n_dora"].iloc[0]
            ),
        },
        "correlation_range": {
            "spearman_min": float(np.nanmin(corr_num["spearman_rho"])),
            "spearman_max": float(np.nanmax(corr_num["spearman_rho"])),
            "pearson_min": float(np.nanmin(corr_num["pearson_r"])),
            "pearson_max": float(np.nanmax(corr_num["pearson_r"])),
            "pvalue_min": float(np.nanmin(corr_num[["spearman_p", "pearson_p"]].to_numpy())),
            "pvalue_max": float(np.nanmax(corr_num[["spearman_p", "pearson_p"]].to_numpy())),
        },
        "d10_f2": {
            "rows_including_caveat": int(len(f2desc)),
            "first_caveat": str(f2desc["caveat"].iloc[0]),
            "contains_18cell_replication": bool(len(f2desc) > 500),
            "min_n_backbones_excluding_caveat": int(pd.to_numeric(f2desc.iloc[1:]["n_backbones"]).min()),
        },
        "metric_eval": {
            "rows": int(len(metric_eval)),
            "best_hit_rate_3class": float(metric_eval["hit_rate_3class"].max()),
            "min_correlation_n_cells": int(corr["n_cells"].min()),
            "max_correlation_n_cells": int(corr["n_cells"].max()),
        },
        "plots": list(plot_paths),
        "framing_forbidden_hits": forbidden_hits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f3 = sub.add_parser("f3-retry")
    f3.add_argument("--backbone", choices=BACKBONES, required=True)
    f3.add_argument("--batch-count", type=int, default=5)
    f3.add_argument("--batch-size", type=int, default=None)
    sub.add_parser("analyze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "f3-retry":
        run_f3_retry_for_backbone(args.backbone, batch_count=args.batch_count, requested_batch_size=args.batch_size)
    elif args.command == "analyze":
        run_analysis()
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
