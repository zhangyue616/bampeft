#!/usr/bin/env python3
"""Compute paired bad_diff bootstrap confidence intervals for MolDoRA tables."""

from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
B_PRODUCTION = 10000
B_SMOKE = 100
BOOTSTRAP_SEED = 42
SANITY_EPSILON = 0.0005
BACKBONES = ["cmpnn", "chemberta2", "molformer_c3"]
PROTOCOLS = ["scaffold", "random"]
METHOD_COMPARISONS = ["lora_vs_dora", "fullft_vs_lora", "fullft_vs_dora"]

INPUTS = {
    "lora_vs_dora_scaffold_stage2": REPO / "docs/stage3/scaffold_split_paired_bad_diff.json",
    "lora_vs_dora_scaffold_z1c": REPO / "docs/stage3/z1c_seed_extension_paired_bad_diff.json",
    "lora_vs_dora_random": REPO / "docs/stage3/z1_seed_extension_paired_bad_diff_extended.json",
    "fullft_vs_peft": REPO / "docs/stage3/n4_full_ft_vs_peft_paired_comparison.json",
    "fullft_raw_results": REPO / "docs/stage3/n4_full_ft_results_table.json",
}

OUT_RESULTS_JSON = REPO / "docs/stage3/bootstrap_ci_results_table.json"
OUT_RESULTS_CSV = REPO / "docs/stage3/bootstrap_ci_results_table.csv"
OUT_SUMMARY_JSON = REPO / "docs/stage3/bootstrap_ci_per_backbone_summary.json"
OUT_SUMMARY_CSV = REPO / "docs/stage3/bootstrap_ci_per_backbone_summary.csv"
OUT_REPORT = REPO / "docs/stage3/PATH_A_BOOTSTRAP_CI_REPORT_2026-05-18.md"
SMOKE_SUMMARY = REPO / "logs/bootstrap_ci/bootstrap_ci_smoke_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def shell_capture(command: str) -> Dict[str, Any]:
    proc = subprocess.run(["bash", "-lc", command], cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": proc.returncode, "output": proc.stdout}


def sem(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def percentile_bootstrap_ci(rows: List[Dict[str, Any]], b: int, seed: int) -> Tuple[float, float, float]:
    """Dataset-stratified bootstrap; unit within each stratum is the dataset x seed pair."""
    grouped: Dict[str, np.ndarray] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(float(row["bad_diff"]))
    grouped = {key: np.asarray(value, dtype=float) for key, value in grouped.items()}
    rng = np.random.default_rng(seed)
    means = np.empty(b, dtype=float)
    for i in range(b):
        samples = []
        for values in grouped.values():
            idx = rng.integers(0, len(values), len(values))
            samples.append(values[idx])
        means[i] = float(np.concatenate(samples).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.mean())


def normalize_lora_dora_row(row: Dict[str, Any], protocol: str, source: str) -> Dict[str, Any]:
    return {
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "protocol": protocol,
        "method_comparison": "lora_vs_dora",
        "bad_diff": float(row["paired_bad_diff"]),
        "metric_type": row.get("metric_type"),
        "task_type": row.get("task_type"),
        "direction": row.get("direction", "regression: dora_rmse - lora_rmse; classification: lora_auc - dora_auc; negative = DoRA better"),
        "source": source,
    }


def normalize_fullft_row(row: Dict[str, Any]) -> Dict[str, Any]:
    method_pair = row["method_pair"]
    return {
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "protocol": row["protocol"],
        "method_comparison": method_pair,
        "bad_diff": float(row["fullft_bad_minus_peft_bad"]),
        "metric_type": row.get("metric_type"),
        "task_type": row.get("task_type"),
        "direction": row.get("direction", "regression: fullft_rmse - peft_rmse; classification: peft_auc - fullft_auc; negative = Full FT better"),
        "source": "docs/stage3/n4_full_ft_vs_peft_paired_comparison.json",
    }


def load_paired_rows() -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    stage2_rows = read_json(INPUTS["lora_vs_dora_scaffold_stage2"])["rows"]
    z1c_rows = read_json(INPUTS["lora_vs_dora_scaffold_z1c"])["rows"]
    rows.extend(normalize_lora_dora_row(row, "scaffold", "docs/stage3/scaffold_split_paired_bad_diff.json") for row in stage2_rows)
    rows.extend(normalize_lora_dora_row(row, "scaffold", "docs/stage3/z1c_seed_extension_paired_bad_diff.json") for row in z1c_rows)
    notes.append(
        "Scope calibration: commission listed z1c_seed_extension_paired_bad_diff.json as scaffold n=36 input, "
        "but file contains 54 rows = 18 pairs/backbone. Computation combines scaffold_split_paired_bad_diff.json "
        "(Stage 2 seeds 0/1/2) + z1c_seed_extension_paired_bad_diff.json (seeds 10/100/1000) to obtain scaffold n=36/backbone."
    )

    random_rows = read_json(INPUTS["lora_vs_dora_random"])["rows"]
    rows.extend(normalize_lora_dora_row(row, "random", "docs/stage3/z1_seed_extension_paired_bad_diff_extended.json") for row in random_rows)

    fullft_rows = read_json(INPUTS["fullft_vs_peft"])["pairs"]
    rows.extend(normalize_fullft_row(row) for row in fullft_rows if row.get("method_pair") in {"fullft_vs_lora", "fullft_vs_dora"})
    return rows, notes


def expected_n(backbone: str, comparison: str, protocol: str) -> int:
    if backbone == "cmpnn" and protocol == "scaffold" and comparison in {"fullft_vs_lora", "fullft_vs_dora"}:
        return 35
    return 36


def compute_records(b: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    paired_rows, notes = load_paired_rows()
    results: List[Dict[str, Any]] = []
    idx = 1
    for backbone in BACKBONES:
        for comparison in METHOD_COMPARISONS:
            for protocol in PROTOCOLS:
                subset = [
                    row
                    for row in paired_rows
                    if row["backbone"] == backbone and row["method_comparison"] == comparison and row["protocol"] == protocol
                ]
                subset = sorted(subset, key=lambda row: (row["dataset"], int(row["seed"])))
                values = np.asarray([row["bad_diff"] for row in subset], dtype=float)
                if len(values) == 0:
                    raise RuntimeError(f"No paired rows for {backbone}/{comparison}/{protocol}")
                direct_mean = float(values.mean())
                lower, upper, boot_mean = percentile_bootstrap_ci(subset, b=b, seed=BOOTSTRAP_SEED)
                sanity_delta = abs(direct_mean - boot_mean)
                input_paths = sorted({row["source"] for row in subset})
                n_expected = expected_n(backbone, comparison, protocol)
                n_pass = len(values) == n_expected
                sanity_pass = bool(sanity_delta <= SANITY_EPSILON and n_pass)
                direction = subset[0]["direction"]
                if comparison == "lora_vs_dora":
                    direction = "bad-space differences: regression dora_rmse - lora_rmse; classification lora_auc - dora_auc; negative = DoRA better"
                else:
                    direction = "bad-space differences: regression fullft_rmse - peft_rmse; classification peft_auc - fullft_auc; negative = Full FT better"
                results.append(
                    {
                        "index": idx,
                        "backbone": backbone,
                        "method_comparison": comparison,
                        "protocol": protocol,
                        "n_pairs": int(len(values)),
                        "expected_n_pairs": int(n_expected),
                        "direction": direction,
                        "mean_paired_bad_diff": direct_mean,
                        "sem": sem(values),
                        "ci_percentile_lower_025": lower,
                        "ci_percentile_upper_975": upper,
                        "ci_bca_lower_025": "",
                        "ci_bca_upper_975": "",
                        "b_bootstrap_samples": int(b),
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "input_data_path": ";".join(input_paths),
                        "sanity_check_pass": sanity_pass,
                        "sanity_check_delta": sanity_delta,
                        "n_check_pass": bool(n_pass),
                        "bootstrap_unit": "dataset_seed_pair_stratified_by_dataset",
                    }
                )
                idx += 1
    return results, notes


def build_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_key[(row["backbone"], row["method_comparison"])][row["protocol"]] = row
    for backbone in BACKBONES:
        for comparison in METHOD_COMPARISONS:
            scaffold = by_key[(backbone, comparison)].get("scaffold")
            random = by_key[(backbone, comparison)].get("random")
            out.append(
                {
                    "backbone": backbone,
                    "method_comparison": comparison,
                    "scaffold_n_pairs": scaffold.get("n_pairs") if scaffold else "",
                    "scaffold_mean": scaffold.get("mean_paired_bad_diff") if scaffold else "",
                    "scaffold_sem": scaffold.get("sem") if scaffold else "",
                    "scaffold_ci_percentile_lower_025": scaffold.get("ci_percentile_lower_025") if scaffold else "",
                    "scaffold_ci_percentile_upper_975": scaffold.get("ci_percentile_upper_975") if scaffold else "",
                    "random_n_pairs": random.get("n_pairs") if random else "",
                    "random_mean": random.get("mean_paired_bad_diff") if random else "",
                    "random_sem": random.get("sem") if random else "",
                    "random_ci_percentile_lower_025": random.get("ci_percentile_lower_025") if random else "",
                    "random_ci_percentile_upper_975": random.get("ci_percentile_upper_975") if random else "",
                    "direction": (scaffold or random or {}).get("direction", ""),
                }
            )
    return out


def md_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.8f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    status: str,
    records: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    notes: List[str],
    input_sha_pre: Dict[str, str],
    input_sha_post: Dict[str, str],
    env_pre: Dict[str, Any],
    env_post: Dict[str, Any],
    helper_sha: Dict[str, str],
    elapsed_min: float,
    smoke_status: str,
) -> None:
    S = chr(167)
    pass_count = sum(1 for row in records if row.get("sanity_check_pass"))
    sha_unchanged = input_sha_pre == input_sha_post
    lines: List[str] = []
    lines.append("# PATH_A_BOOTSTRAP_CI_REPORT_2026-05-18")
    lines.append("")
    lines.append(f"## {S}1 Status")
    lines.append(f"- Status: {status}")
    lines.append(f"- Compute time: {elapsed_min:.2f} min")
    lines.append("- Git boundary: no git add/commit/push by Codex")
    lines.append(f"- Phase 1 smoke status: {smoke_status}")
    lines.append(f"- 18 CI values compute status: {pass_count}/18 PASS")
    lines.append("")
    lines.append(f"## {S}2 Scope lock")
    lines.append("- Input files:")
    for key, path in INPUTS.items():
        lines.append(f"  - {key}: `{path.relative_to(REPO)}`")
    lines.append("- Output scope: 18 paired bootstrap CI records, 3 backbones x 3 method comparisons x 2 protocols.")
    lines.append("- Bootstrap B: 10000 for production; smoke B: 100.")
    lines.append("- Bootstrap unit: dataset x seed pair; implementation preserves dataset strata and resamples observed seed-pairs within each dataset.")
    lines.append("- CI method: percentile 2.5% / 97.5%.")
    lines.append("- BCa: not computed; scipy.stats.bootstrap BCa was not used because the primary method is custom dataset-stratified bootstrap.")
    lines.append("")
    lines.append(f"## {S}3 18 paired bootstrap CI values table")
    lines.extend(
        md_table(
            records,
            [
                "index",
                "backbone",
                "method_comparison",
                "protocol",
                "n_pairs",
                "mean_paired_bad_diff",
                "sem",
                "ci_percentile_lower_025",
                "ci_percentile_upper_975",
                "sanity_check_pass",
            ],
        )
    )
    lines.append("")
    lines.append(f"## {S}4 Methodology")
    lines.append("- Stratified bootstrap on dataset x seed pairs.")
    lines.append("- B = 10000 bootstrap samples.")
    lines.append("- Percentile method: 2.5% and 97.5% percentiles from bootstrap mean distribution.")
    lines.append("- Random seed = 42.")
    lines.append("- Direct mean and SEM are computed from observed paired bad_diff values.")
    lines.append("")
    lines.append(f"## {S}5 Sanity checks")
    lines.append(f"- Bootstrap mean sanity: {pass_count}/18 within epsilon = {SANITY_EPSILON}.")
    lines.append(f"- Input data SHA256 unchanged: {sha_unchanged}.")
    lines.append("Input SHA256 pre:")
    lines.append("```json")
    lines.append(json.dumps(input_sha_pre, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("Input SHA256 post:")
    lines.append("```json")
    lines.append(json.dumps(input_sha_post, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append(f"## {S}6 Pre/post env confirmation")
    for label, payload in [("pre", env_pre), ("post", env_post)]:
        lines.append(f"### conda env list {label}")
        lines.append("```bash")
        lines.append(payload["command"])
        lines.append("```")
        lines.append("```text")
        lines.append(payload["output"])
        lines.append("```")
    lines.append("")
    lines.append(f"## {S}7 Codex helper scripts persistent path declaration")
    for path, digest in helper_sha.items():
        lines.append(f"- `{path}` sha256={digest}")
    lines.append("- No helper script was intentionally kept only in an agent-internal ephemeral workspace.")
    lines.append("")
    lines.append(f"## {S}8 Git boundary statement")
    lines.append("- No git add / commit / push executed by Codex.")
    lines.append("- chemprop/ + docs/scaffold/ + conda envs: no intentional modification.")
    lines.append("- Input data files: read-only; SHA256 unchanged.")
    lines.append("- New files: `docs/stage3/bootstrap_ci_results_table.{json,csv}`, `docs/stage3/bootstrap_ci_per_backbone_summary.{json,csv}`, `docs/stage3/PATH_A_BOOTSTRAP_CI_REPORT_2026-05-18.md`, `scripts/bootstrap_ci/*`.")
    lines.append("- Generated run logs: `logs/bootstrap_ci/smoke.log`, `logs/bootstrap_ci/production.log`, `logs/bootstrap_ci/bootstrap_ci_smoke_summary.json`.")
    lines.append("")
    lines.append(f"## {S}9 Unverified items + residual risks")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append(f"## {S}10 STOP for Claude review")
    lines.append("STOP for Claude review. Zhang Yue is sole git commit executor.")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(mode: str) -> None:
    start = time.time()
    b = B_SMOKE if mode == "smoke" else B_PRODUCTION
    env_pre = shell_capture("source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda env list")
    input_sha_pre = {str(path.relative_to(REPO)): sha256(path) for path in INPUTS.values()}
    records, notes = compute_records(b=b)
    pass_count = sum(1 for row in records if row["sanity_check_pass"])

    if mode == "smoke":
        compute_pass_count = sum(1 for row in records if row["n_check_pass"] and row["n_pairs"] > 0)
        payload = {
            "mode": "smoke",
            "b_bootstrap_samples": b,
            "pass_count": compute_pass_count,
            "strict_sanity_pass_count": pass_count,
            "total": len(records),
            "status": "PASS" if compute_pass_count == 18 else "FAIL",
            "records": records,
        }
        write_json(SMOKE_SUMMARY, payload)
        print(json.dumps({k: payload[k] for k in ["mode", "b_bootstrap_samples", "pass_count", "total", "status"]}, sort_keys=True))
        if compute_pass_count != 18:
            raise SystemExit(1)
        return

    summary = build_summary(records)
    result_fields = [
        "index",
        "backbone",
        "method_comparison",
        "protocol",
        "n_pairs",
        "expected_n_pairs",
        "direction",
        "mean_paired_bad_diff",
        "sem",
        "ci_percentile_lower_025",
        "ci_percentile_upper_975",
        "ci_bca_lower_025",
        "ci_bca_upper_975",
        "b_bootstrap_samples",
        "bootstrap_seed",
        "input_data_path",
        "sanity_check_pass",
        "sanity_check_delta",
        "n_check_pass",
        "bootstrap_unit",
    ]
    summary_fields = [
        "backbone",
        "method_comparison",
        "scaffold_n_pairs",
        "scaffold_mean",
        "scaffold_sem",
        "scaffold_ci_percentile_lower_025",
        "scaffold_ci_percentile_upper_975",
        "random_n_pairs",
        "random_mean",
        "random_sem",
        "random_ci_percentile_lower_025",
        "random_ci_percentile_upper_975",
        "direction",
    ]
    write_json(OUT_RESULTS_JSON, {"rows": records, "summary": {"n_rows": len(records), "pass_count": pass_count}})
    write_csv(OUT_RESULTS_CSV, records, result_fields)
    write_json(OUT_SUMMARY_JSON, {"rows": summary, "summary": {"n_rows": len(summary)}})
    write_csv(OUT_SUMMARY_CSV, summary, summary_fields)
    input_sha_post = {str(path.relative_to(REPO)): sha256(path) for path in INPUTS.values()}
    env_post = shell_capture("source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda env list")
    helper_paths = [
        REPO / "scripts/bootstrap_ci/bootstrap_ci_compute.py",
        REPO / "scripts/bootstrap_ci/run_bootstrap_ci.sh",
    ]
    helper_sha = {str(path.relative_to(REPO)): sha256(path) for path in helper_paths}
    smoke_status = "NOT_RUN"
    if SMOKE_SUMMARY.exists():
        smoke_payload = read_json(SMOKE_SUMMARY)
        smoke_status = f"{smoke_payload.get('status')} ({smoke_payload.get('pass_count')}/{smoke_payload.get('total')} PASS, B={smoke_payload.get('b_bootstrap_samples')})"
    status = "PASS" if pass_count == 18 and input_sha_pre == input_sha_post else "PARTIAL"
    write_report(
        status=status,
        records=records,
        summary=summary,
        notes=notes,
        input_sha_pre=input_sha_pre,
        input_sha_post=input_sha_post,
        env_pre=env_pre,
        env_post=env_post,
        helper_sha=helper_sha,
        elapsed_min=(time.time() - start) / 60.0,
        smoke_status=smoke_status,
    )
    print(json.dumps({"mode": "production", "status": status, "pass_count": pass_count, "total": len(records)}, sort_keys=True))
    if pass_count != 18:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "production"])
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
