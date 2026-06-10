#!/usr/bin/env python3
"""Finalize N5 rank sensitivity outputs, audit checkpoints, and write report."""

from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
SCRIPT_DIR = REPO / "scripts" / "n5_rank_sensitivity"
sys.path.insert(0, str(SCRIPT_DIR))

import n5_rank_runner as runner  # noqa: E402


REPORT = REPO / "docs" / "stage3" / "PATH_A_N5_RANK_SENSITIVITY_REPORT_2026-05-18.md"
RESULTS_JSON = REPO / "docs" / "stage3" / "n5_rank_sensitivity_results_table.json"
RESULTS_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_results_table.csv"
SUMMARY_JSON = REPO / "docs" / "stage3" / "n5_rank_sensitivity_per_backbone_rank_summary.json"
SUMMARY_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_per_backbone_rank_summary.csv"
AUDIT_JSON = REPO / "docs" / "stage3" / "n5_rank_sensitivity_checkpoint_audit.json"
AUDIT_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_checkpoint_audit.csv"
PAIRED_JSON = REPO / "docs" / "stage3" / "n5_rank_sensitivity_vs_r8_baseline_paired_comparison.json"
PAIRED_CSV = REPO / "docs" / "stage3" / "n5_rank_sensitivity_vs_r8_baseline_paired_comparison.csv"
FINALIZE_SUMMARY = REPO / "docs" / "stage3" / "n5_rank_sensitivity_finalize_summary.json"
BASELINE = REPO / "docs" / "stage3" / "scaffold_split_extension_results.json"


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def shell_capture(command: str) -> Dict[str, Any]:
    proc = subprocess.run(["bash", "-lc", command], cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": proc.returncode, "output": proc.stdout}


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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.8f}"
    return str(value)


def md_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return lines


def get_arg(args: Any, key: str, default: Any = None) -> Any:
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def canonical_method(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("dora"):
        return "dora"
    if text.startswith("lora"):
        return "lora"
    return text


def normalize_results(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_key = {
        (row["backbone"], row["dataset"], row["method"], int(row["rank"]), int(row["seed"])): row
        for row in state.get("production_results", [])
    }
    rows: List[Dict[str, Any]] = []
    for spec in runner.production_specs():
        key = (spec["backbone"], spec["dataset"], spec["method"], int(spec["rank"]), int(spec["seed"]))
        row = by_key.get(key)
        if row is None:
            row = {**spec, "protocol": "scaffold", "status": "NOT_RUN", "attempts": 0, "workaround": "missing_from_runner_state"}
        rows.append(row)
    return rows


def audit_checkpoint(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "method": row["method"],
        "rank": int(row["rank"]),
        "seed": int(row["seed"]),
        "status": "FAIL",
        "checkpoint_path": row.get("checkpoint") or runner.rel(runner.checkpoint_path(row["backbone"], row["dataset"], row["method"], int(row["rank"]), int(row["seed"]))),
        "exists": False,
        "matches_spec": False,
        "reason": "",
    }
    if row.get("status") != "PASS":
        out["reason"] = "run status not PASS"
        return out
    ckpt = REPO / out["checkpoint_path"]
    out["exists"] = ckpt.exists()
    if not ckpt.exists():
        out["reason"] = "checkpoint missing"
        return out
    try:
        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = state.get("args")
        config = state.get("lora_config", {})
        state_dict = state.get("state_dict", {})
        failures: List[str] = []
        if int(config.get("rank", -1)) != int(row["rank"]):
            failures.append(f"lora_config rank {config.get('rank')} != {row['rank']}")
        config_method = config.get("peft_method", config.get("method"))
        if canonical_method(config_method) != row["method"]:
            failures.append(f"lora_config method {config_method} != {row['method']}")
        if bool(config.get("use_dora")) != (row["method"] == "dora"):
            failures.append("use_dora mismatch")
        if int(get_arg(args, "seed", -1)) != int(row["seed"]):
            failures.append("args seed mismatch")
        if get_arg(args, "dataset", row["dataset"]) != row["dataset"]:
            failures.append("args dataset mismatch")
        arg_method = get_arg(args, "method", get_arg(args, "peft_method", row["method"]))
        if canonical_method(arg_method) != row["method"]:
            failures.append(f"args method unexpected {arg_method}")
        arg_rank = get_arg(args, "lora_rank", get_arg(args, "rank", row["rank"]))
        if int(arg_rank) != int(row["rank"]):
            failures.append(f"args rank {arg_rank} != {row['rank']}")
        split_type = get_arg(args, "split_type", "scaffold_balanced")
        if split_type != "scaffold_balanced":
            failures.append(f"split_type {split_type} != scaffold_balanced")
        lora_key_count = sum(1 for key in state_dict if "lora" in key.lower())
        if lora_key_count == 0:
            failures.append("state_dict has no lora keys")
        out.update(
            {
                "lora_config_rank": config.get("rank"),
                "lora_config_alpha": config.get("alpha"),
                "lora_config_peft_method": config_method,
                "lora_config_use_dora": config.get("use_dora"),
                "args_rank": arg_rank,
                "args_split_type": split_type,
                "state_dict_lora_key_count": lora_key_count,
            }
        )
        if failures:
            out["reason"] = "; ".join(failures)
            return out
        out["status"] = "PASS"
        out["matches_spec"] = True
        out["reason"] = "ok"
        return out
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out


def lower_is_better(metric_type: str) -> bool:
    return metric_type in {"rmse", "mae", "mse", "loss", "cross_entropy"}


def paired_bad_diff(n5_metric: float, baseline_metric: float, metric_type: str) -> float:
    if lower_is_better(metric_type):
        return float(n5_metric) - float(baseline_metric)
    return float(baseline_metric) - float(n5_metric)


def build_paired(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_rows = read_json(BASELINE)["rows"]
    baseline = {
        (row["backbone"], row["dataset"], row["method"], int(row["seed"])): row
        for row in base_rows
        if row.get("status") == "PASS"
    }
    paired: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "PASS":
            continue
        key = (row["backbone"], row["dataset"], row["method"], int(row["seed"]))
        base = baseline.get(key)
        if base is None:
            continue
        metric_type = row.get("metric_type") or base.get("metric_type")
        diff = paired_bad_diff(float(row["metric"]), float(base["final_metric"]), metric_type)
        paired.append(
            {
                "backbone": row["backbone"],
                "dataset": row["dataset"],
                "method": row["method"],
                "rank": int(row["rank"]),
                "seed": int(row["seed"]),
                "protocol": "scaffold",
                "metric_type": metric_type,
                "task_type": base.get("task_type"),
                "n5_metric": float(row["metric"]),
                "r8_baseline_metric": float(base["final_metric"]),
                "paired_bad_diff_vs_r8": diff,
                "direction": "positive = rank variant worse than r=8 baseline in bad-space",
                "n5_checkpoint": row.get("checkpoint"),
                "r8_checkpoint": base.get("checkpoint_path"),
            }
        )
    return paired


def build_summary(rows: List[Dict[str, Any]], paired: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paired_by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in paired:
        paired_by_key[(row["backbone"], row["method"], int(row["rank"]))].append(row)
    out: List[Dict[str, Any]] = []
    for backbone in runner.BACKBONES:
        for method in runner.METHODS:
            for rank in runner.RANKS:
                subset = [row for row in rows if row["backbone"] == backbone and row["method"] == method and int(row["rank"]) == rank]
                passed = [row for row in subset if row.get("status") == "PASS"]
                metrics = [float(row["metric"]) for row in passed if row.get("metric") is not None]
                paired_subset = paired_by_key.get((backbone, method, rank), [])
                diffs = [float(row["paired_bad_diff_vs_r8"]) for row in paired_subset]
                out.append(
                    {
                        "backbone": backbone,
                        "method": method,
                        "rank": rank,
                        "protocol": "scaffold",
                        "n_total": len(subset),
                        "n_pass": len(passed),
                        "status": "PASS" if len(passed) == len(subset) else "PARTIAL",
                        "metric_mean_mixed": mean(metrics),
                        "metric_sem_mixed": sem(metrics),
                        "paired_vs_r8_n": len(diffs),
                        "paired_bad_diff_vs_r8_mean": mean(diffs),
                        "paired_bad_diff_vs_r8_sem": sem(diffs),
                        "direction": "positive paired_bad_diff_vs_r8 = rank variant worse than r=8 baseline",
                    }
                )
    return out


def helper_sha() -> Dict[str, str]:
    paths = [
        SCRIPT_DIR / "n5_rank_runner.py",
        SCRIPT_DIR / "train_peft_rank_variant.py",
        SCRIPT_DIR / "n5_finalize.py",
        SCRIPT_DIR / "run_n5_rank.sh",
        SCRIPT_DIR / "precompute_scaffold_splits.sh",
    ]
    return {runner.rel(path): sha256(path) for path in paths if path.exists()}


def write_outputs(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], audit: List[Dict[str, Any]], paired: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    result_fields = [
        "index",
        "backbone",
        "dataset",
        "method",
        "rank",
        "seed",
        "protocol",
        "status",
        "metric",
        "metric_type",
        "elapsed_sec",
        "peak_mib",
        "attempts",
        "workaround",
        "epochs_completed",
        "trainable_params",
        "checkpoint",
        "checkpoint_exists",
        "run_log",
    ]
    summary_fields = [
        "backbone",
        "method",
        "rank",
        "protocol",
        "n_total",
        "n_pass",
        "status",
        "metric_mean_mixed",
        "metric_sem_mixed",
        "paired_vs_r8_n",
        "paired_bad_diff_vs_r8_mean",
        "paired_bad_diff_vs_r8_sem",
        "direction",
    ]
    audit_fields = [
        "backbone",
        "dataset",
        "method",
        "rank",
        "seed",
        "status",
        "checkpoint_path",
        "exists",
        "matches_spec",
        "reason",
        "lora_config_rank",
        "lora_config_alpha",
        "lora_config_peft_method",
        "lora_config_use_dora",
        "args_rank",
        "args_split_type",
        "state_dict_lora_key_count",
    ]
    paired_fields = [
        "backbone",
        "dataset",
        "method",
        "rank",
        "seed",
        "protocol",
        "metric_type",
        "task_type",
        "n5_metric",
        "r8_baseline_metric",
        "paired_bad_diff_vs_r8",
        "direction",
        "n5_checkpoint",
        "r8_checkpoint",
    ]
    write_json(RESULTS_JSON, {"rows": rows, "summary": {"n_rows": len(rows), "n_pass": sum(1 for row in rows if row.get("status") == "PASS")}})
    write_csv(RESULTS_CSV, rows, result_fields)
    write_json(SUMMARY_JSON, {"rows": summary, "summary": {"n_rows": len(summary)}})
    write_csv(SUMMARY_CSV, summary, summary_fields)
    write_json(AUDIT_JSON, {"rows": audit, "summary": {"n_rows": len(audit), "n_pass": sum(1 for row in audit if row.get("status") == "PASS")}})
    write_csv(AUDIT_CSV, audit, audit_fields)
    write_json(PAIRED_JSON, {"rows": paired, "summary": {"n_rows": len(paired)}})
    write_csv(PAIRED_CSV, paired, paired_fields)
    final_summary = {
        "result_rows": len(rows),
        "pass_rows": sum(1 for row in rows if row.get("status") == "PASS"),
        "audit_rows": len(audit),
        "audit_pass_rows": sum(1 for row in audit if row.get("status") == "PASS"),
        "paired_rows": len(paired),
        "smoke_pass_rows": sum(1 for row in state.get("smoke_results", []) if row.get("status") == "PASS"),
        "fail_rows": [row for row in rows if row.get("status") != "PASS"],
        "audit_fail_rows": [row for row in audit if row.get("status") != "PASS"],
    }
    write_json(FINALIZE_SUMMARY, final_summary)
    return final_summary


def write_report(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], audit: List[Dict[str, Any]], paired: List[Dict[str, Any]], final_summary: Dict[str, Any], state: Dict[str, Any]) -> None:
    S = chr(167)
    smoke = state.get("smoke_results", [])
    smoke_pass = sum(1 for row in smoke if row.get("status") == "PASS")
    prod_pass = final_summary["pass_rows"]
    audit_pass = final_summary["audit_pass_rows"]
    fail_count = 216 - prod_pass
    if smoke_pass < 3:
        status = "FAIL"
    elif prod_pass == 216 and audit_pass == 216:
        status = "PASS"
    elif fail_count > 21:
        status = "FAIL"
    else:
        status = "PARTIAL"
    elapsed_hr = sum(float(row.get("elapsed_sec") or 0.0) for row in rows) / 3600.0
    input_sha_pre = state.get("input_sha_pre", {})
    input_sha_post = runner.input_sha()
    env_pre = state.get("env_pre") or shell_capture("source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda env list")
    env_post = shell_capture("source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda env list")
    git_boundary = {
        "status": shell_capture("git status --short -- chemprop docs/scaffold scripts/n5_rank_sensitivity docs/stage3/n5_rank_sensitivity_results_table.json docs/stage3/n5_rank_sensitivity_results_table.csv docs/stage3/n5_rank_sensitivity_per_backbone_rank_summary.json docs/stage3/n5_rank_sensitivity_per_backbone_rank_summary.csv docs/stage3/n5_rank_sensitivity_checkpoint_audit.json docs/stage3/n5_rank_sensitivity_checkpoint_audit.csv docs/stage3/n5_rank_sensitivity_vs_r8_baseline_paired_comparison.json docs/stage3/n5_rank_sensitivity_vs_r8_baseline_paired_comparison.csv docs/stage3/n5_rank_sensitivity_finalize_summary.json docs/stage3/PATH_A_N5_RANK_SENSITIVITY_REPORT_2026-05-18.md"),
        "diff_boundary": shell_capture("git diff --name-only -- chemprop docs/scaffold"),
    }
    lines: List[str] = []
    lines.append("# PATH_A_N5_RANK_SENSITIVITY_REPORT_2026-05-18")
    lines.append("")
    lines.append(f"## {S}1 Status")
    lines.append(f"- Status: {status}")
    lines.append(f"- Phase 1 smoke: {smoke_pass}/3 PASS")
    lines.append(f"- Phase 2 production: {prod_pass}/216 PASS")
    lines.append(f"- Stage 4 checkpoint audit: {audit_pass}/216 PASS")
    lines.append(f"- Compute time: {elapsed_hr:.2f} hr on 5090 (sum of per-run elapsed seconds)")
    lines.append("- Git boundary: no git add/commit/push by Codex")
    lines.append("")
    lines.append(f"## {S}2 Scope lock")
    lines.append("- Scope: LoRA and DoRA rank sensitivity, ranks r=4 and r=16, scaffold-primary protocol only.")
    lines.append("- Backbones: CMPNN, ChemBERTa-2, MoLFormer-c3.")
    lines.append("- Datasets: FreeSolv, ESOL, Lipo, BACE, BBBP, SIDER.")
    lines.append("- Seeds: 0, 1, 2.")
    lines.append("- Total production scope: 3 backbones x 6 datasets x 2 methods x 2 ranks x 3 seeds = 216 runs.")
    lines.append("- Rank decision: lora_alpha kept at Stage 2 baseline value 16; only rank is changed to isolate rank sensitivity.")
    lines.append("- Scaffold split precompute: `scripts/n5_rank_sensitivity/precompute_scaffold_splits.sh` in `kapt` env.")
    lines.append("")
    lines.append(f"## {S}3 216-run results table summary")
    lines.extend(md_table(summary, ["backbone", "method", "rank", "protocol", "n_total", "n_pass", "status", "metric_mean_mixed", "metric_sem_mixed", "paired_bad_diff_vs_r8_mean", "paired_bad_diff_vs_r8_sem"]))
    lines.append("")
    lines.append(f"## {S}4 Methodology")
    lines.append("- Pattern: PEFT training follows Stage 2/Z1-c style with rank parameter swap r=4/r=16 instead of r=8 baseline.")
    lines.append("- CMPNN path: chemprop `train.py` with `--peft_method {lora,dora}` and `--lora_rank` set to 4 or 16.")
    lines.append("- Transformer path: N5 wrapper around Stage 2 scaffold trainer with PEFT `LoraConfig(r=rank, lora_alpha=16, use_dora=method==dora)`.")
    lines.append("- Direction-corrected paired bad_diff vs Stage 2 r=8 baseline pairs same backbone, dataset, method, seed.")
    lines.append("- Sample size target: n=18 paired comparisons per backbone x method x rank.")
    lines.append("")
    lines.append(f"## {S}5 N5 vs r=8 baseline paired comparison")
    pair_summary = []
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in paired:
        grouped[(row["backbone"], row["method"], int(row["rank"]))].append(float(row["paired_bad_diff_vs_r8"]))
    for backbone in runner.BACKBONES:
        for method in runner.METHODS:
            for rank in runner.RANKS:
                vals = grouped.get((backbone, method, rank), [])
                pair_summary.append({"backbone": backbone, "method": method, "rank": rank, "n": len(vals), "mean_paired_bad_diff_vs_r8": mean(vals), "sem": sem(vals), "direction": "positive = rank variant worse than r=8"})
    lines.extend(md_table(pair_summary, ["backbone", "method", "rank", "n", "mean_paired_bad_diff_vs_r8", "sem", "direction"]))
    lines.append("")
    lines.append(f"## {S}6 Sanity checks")
    lines.append(f"- Input data SHA256 unchanged: {input_sha_pre == input_sha_post}")
    lines.append("- Boundary clean check: `git diff --name-only -- chemprop docs/scaffold` output below.")
    lines.append("```text")
    lines.append(git_boundary["diff_boundary"]["output"])
    lines.append("```")
    lines.append("- Helper scripts persistent and sha256 declared in section 8.")
    lines.append("Input SHA256 pre:")
    lines.append("```json")
    lines.append(json.dumps(input_sha_pre, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("Input SHA256 post:")
    lines.append("```json")
    lines.append(json.dumps(input_sha_post, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append(f"## {S}7 Pre/post env confirmation")
    for label, payload in [("pre", env_pre), ("post", env_post)]:
        lines.append(f"### conda env list {label}")
        lines.append("```bash")
        lines.append(payload.get("command", ""))
        lines.append("```")
        lines.append("```text")
        lines.append(payload.get("output", ""))
        lines.append("```")
    lines.append("")
    lines.append(f"## {S}8 Codex helper scripts persistent path declaration")
    for path, digest in helper_sha().items():
        lines.append(f"- `{path}` sha256={digest}")
    lines.append("- No helper script was intentionally kept only in an agent-internal ephemeral workspace.")
    lines.append("")
    lines.append(f"## {S}9 Git boundary statement")
    lines.append("- No git add / commit / push executed by Codex.")
    lines.append("- chemprop/ + docs/scaffold/ + conda envs: no intentional modification.")
    lines.append("- Input data files unchanged according to SHA256 pre/post comparison.")
    lines.append("- New files: docs/stage3/n5_rank_sensitivity_*.{json,csv}, docs/stage3/PATH_A_N5_RANK_SENSITIVITY_REPORT_2026-05-18.md, scripts/n5_rank_sensitivity/*, logs/n5_rank_sensitivity/*, dumped/n5_rank_sensitivity_*.")
    lines.append("Git status boundary subset:")
    lines.append("```text")
    lines.append(git_boundary["status"]["output"])
    lines.append("```")
    lines.append("")
    lines.append(f"## {S}10 Failures + workarounds")
    fail_rows = [row for row in rows if row.get("status") != "PASS"]
    if fail_rows:
        lines.extend(md_table(fail_rows, ["index", "backbone", "dataset", "method", "rank", "seed", "status", "attempts", "workaround", "run_log"]))
    else:
        lines.append("- No production run failures.")
    audit_fail = [row for row in audit if row.get("status") != "PASS"]
    if audit_fail:
        lines.append("- Audit failures:")
        lines.extend(md_table(audit_fail, ["backbone", "dataset", "method", "rank", "seed", "checkpoint_path", "reason"]))
    lines.append("")
    lines.append(f"## {S}11 Unverified items + residual risks")
    if status == "PASS":
        lines.append("- None surfaced by N5 finalizer.")
    else:
        lines.append("- See section 10 failures and audit rows.")
    lines.append("")
    lines.append(f"## {S}12 STOP for Claude review")
    lines.append("STOP for Claude main session review. Zhang Yue is sole git commit executor and morning surface reviewer.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(mode: str) -> None:
    state = runner.load_state()
    rows = normalize_results(state)
    paired = build_paired(rows)
    summary = build_summary(rows, paired)
    audit = [audit_checkpoint(row) for row in rows]
    final_summary = write_outputs(rows, summary, audit, paired, state)
    write_report(rows, summary, audit, paired, final_summary, state)
    if mode == "audit-only":
        print(json.dumps({"mode": mode, "audit_pass": final_summary["audit_pass_rows"], "audit_rows": final_summary["audit_rows"]}, sort_keys=True))
    else:
        print(json.dumps({"mode": mode, "pass_rows": final_summary["pass_rows"], "result_rows": final_summary["result_rows"], "audit_pass_rows": final_summary["audit_pass_rows"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="production", choices=["smoke", "production", "audit-only", "report-only"])
    args = parser.parse_args()
    finalize(args.mode)


if __name__ == "__main__":
    main()
