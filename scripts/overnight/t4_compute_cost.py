#!/usr/bin/env python3
"""Aggregate wall-clock and peak VRAM from existing result files and logs."""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from overnight_common import REPO, T4_DIR, ensure_output_dirs, read_json, write_csv, write_json


RESULT_JSONS = [
    REPO / "docs/stage3/scaffold_split_extension_results.json",
    REPO / "docs/stage3/z1c_seed_extension_results_table.json",
    REPO / "docs/stage3/z1_seed_extension_results.json",
    REPO / "docs/stage3/n4_full_ft_results_table.json",
    REPO / "docs/stage3/n5_rank_sensitivity_results_table.json",
]


def infer_from_path(path: Path) -> Dict[str, str]:
    text = str(path)
    out = {"dataset": "", "backbone": "", "method": "", "protocol": "", "seed": ""}
    name = path.parent.name
    m = re.search(r"baseline_([a-z0-9]+)_([a-z0-9_]+?)(?:_(chemberta2|molformer_c3|cmpnn))?_(scaffold|random)?_?seed([0-9]+)", name)
    if m:
        out["dataset"] = m.group(1)
        out["method"] = m.group(2)
        out["backbone"] = m.group(3) or "cmpnn"
        out["protocol"] = m.group(4) or ("scaffold" if "_scaffold_" in name else "random")
        out["seed"] = m.group(5)
    return out


def normalize_row(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    elapsed = row.get("elapsed_sec", row.get("wall_clock_sec", ""))
    peak = row.get("peak_mib", row.get("nvidia_smi_peak_mib", row.get("torch_peak_allocated_mib", "")))
    protocol = row.get("protocol") or row.get("split_type") or ""
    if protocol == "scaffold_balanced":
        protocol = "scaffold"
    return {
        "source": source,
        "backbone": row.get("backbone", ""),
        "dataset": row.get("dataset", ""),
        "method": row.get("method", row.get("peft_method", "")),
        "protocol": protocol,
        "seed": row.get("seed", ""),
        "status": row.get("status", ""),
        "elapsed_sec": elapsed,
        "peak_vram_mib": peak,
    }


def load_stage3_rows() -> List[Dict[str, Any]]:
    out = []
    for path in RESULT_JSONS:
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if rows is None and isinstance(payload, dict):
            rows = payload.get("production_results")
        if not isinstance(rows, list):
            continue
        for row in rows:
            norm = normalize_row(row, f"stage3:{path.name}")
            if norm["elapsed_sec"] != "" or norm["peak_vram_mib"] != "":
                out.append(norm)
    return out


def load_dumped_metrics() -> List[Dict[str, Any]]:
    out = []
    for path in (REPO / "dumped").glob("**/metrics.json"):
        try:
            row = read_json(path)
        except Exception:
            continue
        norm = normalize_row(row, f"dumped:{path.relative_to(REPO)}")
        inferred = infer_from_path(path)
        for key, value in inferred.items():
            if not norm.get(key) and value:
                norm[key] = value
        if norm["elapsed_sec"] != "" or norm["peak_vram_mib"] != "":
            out.append(norm)
    return out


def to_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") not in ("", "PASS"):
            continue
        key = (row.get("backbone", ""), row.get("method", ""), row.get("protocol", ""))
        groups[key].append(row)
    out = []
    for (backbone, method, protocol), subset in sorted(groups.items()):
        elapsed = np.asarray([to_float(row.get("elapsed_sec")) for row in subset], dtype=float)
        peak = np.asarray([to_float(row.get("peak_vram_mib")) for row in subset], dtype=float)
        elapsed = elapsed[np.isfinite(elapsed)]
        peak = peak[np.isfinite(peak)]
        out.append(
            {
                "backbone": backbone,
                "method": method,
                "protocol": protocol,
                "n_runs": len(subset),
                "wall_clock_mean_sec": float(elapsed.mean()) if len(elapsed) else "",
                "wall_clock_std_sec": float(elapsed.std(ddof=1)) if len(elapsed) > 1 else "",
                "peak_vram_mean_mib": float(peak.mean()) if len(peak) else "",
                "peak_vram_max_mib": float(peak.max()) if len(peak) else "",
            }
        )
    return out


def main() -> None:
    ensure_output_dirs()
    start = time.time()
    rows = load_stage3_rows() + load_dumped_metrics()
    # De-duplicate by source path and key to avoid obvious double counts from mirrored summary files.
    dedup = {}
    for row in rows:
        key = (row.get("source"), row.get("backbone"), row.get("dataset"), row.get("method"), row.get("protocol"), row.get("seed"))
        dedup[key] = row
    rows = list(dedup.values())
    agg = aggregate(rows)
    per_fields = ["source", "backbone", "dataset", "method", "protocol", "seed", "status", "elapsed_sec", "peak_vram_mib"]
    agg_fields = ["backbone", "method", "protocol", "n_runs", "wall_clock_mean_sec", "wall_clock_std_sec", "peak_vram_mean_mib", "peak_vram_max_mib"]
    write_csv(T4_DIR / "t4_compute_cost_per_run.csv", rows, per_fields)
    write_csv(T4_DIR / "t4_compute_cost_aggregate.csv", agg, agg_fields)
    write_json(T4_DIR / "t4_compute_cost_per_run.json", rows)
    write_json(T4_DIR / "t4_compute_cost_aggregate.json", agg)
    (T4_DIR / "t4_summary.json").write_text(
        json.dumps({"rows": len(rows), "groups": len(agg), "wall_clock_sec": time.time() - start}, indent=2),
        encoding="utf-8",
    )
    print(f"T4 done rows={len(rows)} groups={len(agg)} wall_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
