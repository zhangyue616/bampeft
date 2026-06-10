import os
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = REPO / "docs" / "stage3"
DATASETS = ["freesolv", "esol", "lipo", "bace", "bbbp", "sider"]
SEEDS = [0, 1, 2, 10, 100, 1000]
BACKBONES = ["cmpnn", "chemberta2", "molformer_c3"]
PROTOCOLS = ["random", "scaffold"]
DATASET_INFO = {
    "freesolv": {"task_type": "regression", "metric_type": "rmse"},
    "esol": {"task_type": "regression", "metric_type": "rmse"},
    "lipo": {"task_type": "regression", "metric_type": "rmse"},
    "bace": {"task_type": "classification", "metric_type": "auc"},
    "bbbp": {"task_type": "classification", "metric_type": "auc"},
    "sider": {"task_type": "classification", "metric_type": "auc"},
}
FINAL_RE = re.compile(r"Final test (rmse|auc) = ([0-9.]+)")
EPOCH_RE = re.compile(r"Epoch ([0-9]+)/([0-9]+)")
TRAINABLE_RE = re.compile(r"Trainable params: ([0-9,]+)")


def load_vanilla_audit():
    path = REPO / "chemprop" / "audit" / "vanilla_checkpoint_audit.py"
    spec = importlib.util.spec_from_file_location("vanilla_checkpoint_audit", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


VANILLA_AUDIT = load_vanilla_audit()
_TORCH_LOAD = torch.load


def torch_load_trusted_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


VANILLA_AUDIT.torch.load = torch_load_trusted_checkpoint


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def save_dir(backbone: str, dataset: str, protocol: str, seed: int) -> Path:
    return REPO / "dumped" / f"baseline_{dataset}_fullft_{backbone}_{protocol}_seed{seed}"


def checkpoint_path(backbone: str, dataset: str, protocol: str, seed: int) -> Path:
    if backbone == "cmpnn":
        return save_dir(backbone, dataset, protocol, seed) / "run_0" / "model_0" / "model.pt"
    return save_dir(backbone, dataset, protocol, seed) / "model.pt"


def cmpnn_log_path(dataset: str, protocol: str, seed: int) -> Path:
    return REPO / "logs" / f"baseline_{dataset}_fullft_cmpnn_{protocol}" / f"seed{seed}" / "training.log"


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_cmpnn(dataset: str, protocol: str, seed: int) -> Dict[str, Any]:
    log_path = cmpnn_log_path(dataset, protocol, seed)
    lines = read_lines(log_path)
    finals: List[Tuple[str, float]] = []
    epochs: List[Tuple[int, int]] = []
    trainable = None
    done = False
    for line in lines:
        match = FINAL_RE.search(line)
        if match:
            finals.append((match.group(1), float(match.group(2))))
        epoch = EPOCH_RE.search(line)
        if epoch:
            epochs.append((int(epoch.group(1)), int(epoch.group(2))))
        tmatch = TRAINABLE_RE.search(line)
        if tmatch:
            trainable = int(tmatch.group(1).replace(",", ""))
        if "Training completed successfully!" in line:
            done = True
    metric_type, metric = finals[-1] if finals else (DATASET_INFO[dataset]["metric_type"], None)
    ckpt = checkpoint_path("cmpnn", dataset, protocol, seed)
    return {
        "metric_type": metric_type,
        "final_metric": metric,
        "status": "PASS" if metric is not None and done and ckpt.exists() else "FAIL",
        "epochs_completed": epochs[-1][0] if epochs else None,
        "epochs_requested": epochs[-1][1] if epochs else 50,
        "batch_size": 50,
        "lr": 0.001,
        "weight_decay": 0.0,
        "trainable_params": trainable,
        "elapsed_sec": None,
        "peak_mib": None,
        "checkpoint_path": rel(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "training_log": rel(log_path),
    }


def parse_transformer(backbone: str, dataset: str, protocol: str, seed: int) -> Dict[str, Any]:
    metrics_path = save_dir(backbone, dataset, protocol, seed) / "metrics.json"
    ckpt = checkpoint_path(backbone, dataset, protocol, seed)
    if not metrics_path.exists():
        return {
            "metric_type": DATASET_INFO[dataset]["metric_type"],
            "final_metric": None,
            "status": "FAIL",
            "epochs_completed": None,
            "epochs_requested": 50,
            "batch_size": 50,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "trainable_params": None,
            "elapsed_sec": None,
            "peak_mib": None,
            "checkpoint_path": rel(ckpt),
            "checkpoint_exists": ckpt.exists(),
            "training_log": rel(save_dir(backbone, dataset, protocol, seed) / "training.log"),
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "metric_type": metrics.get("metric_type", DATASET_INFO[dataset]["metric_type"]),
        "final_metric": metrics.get("final_metric"),
        "status": "PASS" if metrics.get("status") == "PASS" and ckpt.exists() else "FAIL",
        "epochs_completed": metrics.get("epochs_completed"),
        "epochs_requested": metrics.get("epochs_requested"),
        "batch_size": metrics.get("batch_size"),
        "lr": metrics.get("lr"),
        "weight_decay": metrics.get("weight_decay"),
        "trainable_params": metrics.get("trainable_params"),
        "elapsed_sec": metrics.get("elapsed_sec"),
        "peak_mib": metrics.get("nvidia_smi_peak_mib"),
        "checkpoint_path": rel(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "training_log": rel(save_dir(backbone, dataset, protocol, seed) / "training.log"),
    }


def metric_row(backbone: str, dataset: str, protocol: str, seed: int) -> Dict[str, Any]:
    parsed = parse_cmpnn(dataset, protocol, seed) if backbone == "cmpnn" else parse_transformer(backbone, dataset, protocol, seed)
    return {
        "backbone": backbone,
        "dataset": dataset,
        "method": "fullft",
        "seed": seed,
        "protocol": protocol,
        "task_type": DATASET_INFO[dataset]["task_type"],
        **parsed,
    }


def stats(values: List[float]) -> Dict[str, Any]:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"n": 0, "mean": None, "std": None, "sem": None}
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"n": len(values), "mean": mean, "std": std, "sem": std / math.sqrt(len(values))}


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cmpnn_peft_log_path(dataset: str, method: str, seed: int) -> Path:
    return REPO / "logs" / f"baseline_{dataset}_{method}" / f"seed{seed}" / "training.log"


def parse_existing_cmpnn_random_peft() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for method in ["lora", "dora"]:
            for seed in [0, 1, 2]:
                text = "\n".join(read_lines(cmpnn_peft_log_path(dataset, method, seed)))
                finals = [(match.group(1), float(match.group(2))) for match in FINAL_RE.finditer(text)]
                if not finals:
                    continue
                metric_type, metric = finals[-1]
                rows.append(
                    {
                        "backbone": "cmpnn",
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "protocol": "random",
                        "final_metric": metric,
                        "task_type": DATASET_INFO[dataset]["task_type"],
                        "metric_type": metric_type,
                    }
                )
    return rows


def parse_existing_transformer_random_peft() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in load_csv_rows(OUT / "b2_phase4_progress.csv"):
        method = row.get("method", "")
        backbone = row.get("backbone", "")
        dataset = row.get("dataset", "")
        if method not in {"lora", "dora"} or backbone not in {"chemberta2", "molformer_c3"}:
            continue
        seed = int(row.get("seed", 0))
        if seed not in [0, 1, 2] or row.get("status") != "PASS":
            continue
        rows.append(
            {
                "backbone": backbone,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "protocol": "random",
                "final_metric": float(row["final_metric"]),
                "task_type": DATASET_INFO[dataset]["task_type"],
                "metric_type": row.get("metric_type") or DATASET_INFO[dataset]["metric_type"],
            }
        )
    return rows


def load_peft_rows() -> List[Dict[str, Any]]:
    sources = [
        OUT / "z1_seed_extension_results.csv",
        OUT / "scaffold_split_extension_results.csv",
        OUT / "z1c_seed_extension_results_table.csv",
    ]
    rows: List[Dict[str, Any]] = []
    rows.extend(parse_existing_cmpnn_random_peft())
    rows.extend(parse_existing_transformer_random_peft())
    for path in sources:
        for row in load_csv_rows(path):
            backbone = row.get("backbone", "")
            method = row.get("method", "")
            dataset = row.get("dataset", "")
            seed = int(row.get("seed", 0))
            protocol = row.get("protocol") or row.get("split_type") or "random"
            if protocol == "scaffold_balanced":
                protocol = "scaffold"
            if protocol == "":
                protocol = "random"
            if method not in {"lora", "dora"} or backbone not in BACKBONES:
                continue
            rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "protocol": protocol,
                    "final_metric": float(row["final_metric"]),
                    "task_type": row.get("task_type") or DATASET_INFO[dataset]["task_type"],
                    "metric_type": row.get("metric_type") or DATASET_INFO[dataset]["metric_type"],
                }
            )
    return rows


def peft_index(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str, int, str], float]:
    index: Dict[Tuple[str, str, str, int, str], float] = {}
    for row in rows:
        key = (row["backbone"], row["dataset"], row["method"], int(row["seed"]), row["protocol"])
        index[key] = float(row["final_metric"])
    return index


def bad_diff(dataset: str, fullft: float, peft: float) -> float:
    if DATASET_INFO[dataset]["task_type"] == "regression":
        return fullft - peft
    return peft - fullft


def fullft_vs_peft(result_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    peft = peft_index(load_peft_rows())
    pairs: List[Dict[str, Any]] = []
    for row in result_rows:
        if row["status"] != "PASS" or row["final_metric"] is None:
            continue
        for method in ["lora", "dora"]:
            key = (row["backbone"], row["dataset"], method, int(row["seed"]), row["protocol"])
            if key not in peft:
                continue
            diff = bad_diff(row["dataset"], float(row["final_metric"]), peft[key])
            pairs.append(
                {
                    "backbone": row["backbone"],
                    "protocol": row["protocol"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "method_pair": f"fullft_vs_{method}",
                    "task_type": row["task_type"],
                    "metric_type": row["metric_type"],
                    "fullft_metric": row["final_metric"],
                    "peft_metric": peft[key],
                    "fullft_bad_minus_peft_bad": diff,
                    "direction": "regression: fullft_rmse - peft_rmse; classification: peft_auc - fullft_auc; negative = Full FT better",
                }
            )
    summary_rows: List[Dict[str, Any]] = []
    for backbone in BACKBONES:
        for protocol in PROTOCOLS:
            full_values = [
                float(row["final_metric"])
                for row in result_rows
                if row["status"] == "PASS" and row["backbone"] == backbone and row["protocol"] == protocol and row["final_metric"] is not None
            ]
            lora_values = [
                float(value)
                for key, value in peft.items()
                if key[0] == backbone and key[2] == "lora" and key[4] == protocol
            ]
            dora_values = [
                float(value)
                for key, value in peft.items()
                if key[0] == backbone and key[2] == "dora" and key[4] == protocol
            ]
            full_stats = stats(full_values)
            lora_stats = stats(lora_values)
            dora_stats = stats(dora_values)
            lora_diffs = [
                float(pair["fullft_bad_minus_peft_bad"])
                for pair in pairs
                if pair["backbone"] == backbone and pair["protocol"] == protocol and pair["method_pair"] == "fullft_vs_lora"
            ]
            dora_diffs = [
                float(pair["fullft_bad_minus_peft_bad"])
                for pair in pairs
                if pair["backbone"] == backbone and pair["protocol"] == protocol and pair["method_pair"] == "fullft_vs_dora"
            ]
            summary_rows.append(
                {
                    "backbone": backbone,
                    "protocol": protocol,
                    "fullft_n": full_stats["n"],
                    "fullft_mean": full_stats["mean"],
                    "fullft_sem": full_stats["sem"],
                    "lora_n": lora_stats["n"],
                    "lora_mean": lora_stats["mean"],
                    "lora_sem": lora_stats["sem"],
                    "dora_n": dora_stats["n"],
                    "dora_mean": dora_stats["mean"],
                    "dora_sem": dora_stats["sem"],
                    "fullft_minus_lora_bad_mean": stats(lora_diffs)["mean"],
                    "fullft_minus_dora_bad_mean": stats(dora_diffs)["mean"],
                    "direction": "bad-space differences: negative = Full FT better",
                }
            )
    return pairs, summary_rows


def audit_cmpnn(row: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = Path(REPO / row["checkpoint_path"])
    audit: Dict[str, Any] = {
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "protocol": row["protocol"],
        "checkpoint_path": row["checkpoint_path"],
        "exists": ckpt.exists(),
        "expected_effective_method": "full_ft",
    }
    if not ckpt.exists():
        audit.update({"matches_spec": False, "reason": "missing checkpoint", "fail_count": 1, "audit_row_count": 0})
        return audit
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    args = state.get("args")
    state_dict = state.get("state_dict", {})
    dealiased = {key: value for key, value in VANILLA_AUDIT._dealiased_items(state_dict)}
    lora_key_count = sum(1 for key in dealiased if "lora_" in key)
    adapter_key_count = sum(1 for key in dealiased if ".adapter." in key)
    contract_numel = sum(int(value.numel()) for value in dealiased.values() if hasattr(value, "numel"))
    expected_split_type = "scaffold_balanced" if row["protocol"] == "scaffold" else "random"
    checks = [
        {"args", "state_dict", "data_scaler", "features_scaler"}.issubset(set(state.keys())),
        "lora_config" not in state or state.get("lora_config") is None,
        "adapter_config" not in state or state.get("adapter_config") is None,
        VANILLA_AUDIT._arg_get(args, "peft_method") == "none",
        bool(VANILLA_AUDIT._arg_get(args, "head_only", False)) is False,
        bool(VANILLA_AUDIT._arg_get(args, "use_dora", False)) is False,
        VANILLA_AUDIT._arg_get(args, "effective_method") == "full_ft",
        VANILLA_AUDIT._arg_get(args, "split_type") == expected_split_type,
        int(VANILLA_AUDIT._arg_get(args, "seed", -1)) == int(row["seed"]),
        int(VANILLA_AUDIT._arg_get(args, "batch_size", -1)) == 50,
        int(VANILLA_AUDIT._arg_get(args, "epochs", -1)) == 50,
        float(VANILLA_AUDIT._arg_get(args, "weight_decay", -1)) == 0.0,
        lora_key_count == 0,
        adapter_key_count == 0,
        contract_numel > 0,
    ]
    failures = [check for check in checks if not check]
    audit.update(
        {
            "matches_spec": all(checks),
            "reason": "pass" if not failures else "metadata/content mismatch",
            "fail_count": len(failures),
            "audit_row_count": len(checks),
            "args_split_type": VANILLA_AUDIT._arg_get(args, "split_type"),
            "args_seed": VANILLA_AUDIT._arg_get(args, "seed"),
            "args_batch_size": VANILLA_AUDIT._arg_get(args, "batch_size"),
            "args_epochs": VANILLA_AUDIT._arg_get(args, "epochs"),
            "args_weight_decay": VANILLA_AUDIT._arg_get(args, "weight_decay"),
            "contract_trainable_numel": contract_numel,
            "lora_key_count": lora_key_count,
            "adapter_key_count": adapter_key_count,
        }
    )
    return audit


def count_keys(state_dict: Dict[str, Any], needle: str) -> int:
    return sum(1 for key in state_dict if needle in key)


def audit_transformer(row: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = Path(REPO / row["checkpoint_path"])
    audit: Dict[str, Any] = {
        "backbone": row["backbone"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "protocol": row["protocol"],
        "checkpoint_path": row["checkpoint_path"],
        "exists": ckpt.exists(),
        "expected_effective_method": "full_ft",
    }
    if not ckpt.exists():
        audit.update({"matches_spec": False, "reason": "missing checkpoint"})
        return audit
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", {})
    args = state.get("args", {})
    full_ft_config = state.get("full_ft_config", {})
    checks = [
        {"args", "state_dict", "data_scaler", "features_scaler", "full_ft_config", "provenance"}.issubset(set(state.keys())),
        "lora_config" not in state,
        full_ft_config.get("peft_method") == "fullft",
        full_ft_config.get("trainability") == "all_parameters_trainable",
        args.get("split_type") in {"random", "scaffold_balanced"},
        int(args.get("seed", -1)) == int(row["seed"]),
        int(args.get("batch_size", -1)) == 50,
        int(args.get("epochs", -1)) == 50,
        float(args.get("lr", -1)) == 2e-5,
        float(args.get("weight_decay", -1)) == 0.01,
        count_keys(state_dict, "lora_A") == 0,
        count_keys(state_dict, "lora_B") == 0,
        count_keys(state_dict, "lora_m") == 0,
    ]
    audit.update(
        {
            "matches_spec": all(checks),
            "reason": "pass" if all(checks) else "metadata/content mismatch",
            "has_lora_config": "lora_config" in state,
            "has_full_ft_config": "full_ft_config" in state,
            "lora_A_count": count_keys(state_dict, "lora_A"),
            "lora_B_count": count_keys(state_dict, "lora_B"),
            "lora_m_count": count_keys(state_dict, "lora_m"),
            "args_split_type": args.get("split_type"),
            "args_seed": args.get("seed"),
            "args_batch_size": args.get("batch_size"),
            "args_epochs": args.get("epochs"),
            "args_lr": args.get("lr"),
            "args_weight_decay": args.get("weight_decay"),
        }
    )
    return audit


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_rows = [
        metric_row(backbone, dataset, protocol, seed)
        for protocol in PROTOCOLS
        for backbone in BACKBONES
        for dataset in DATASETS
        for seed in SEEDS
    ]
    result_fields = [
        "backbone",
        "dataset",
        "method",
        "seed",
        "protocol",
        "task_type",
        "metric_type",
        "final_metric",
        "status",
        "epochs_completed",
        "epochs_requested",
        "batch_size",
        "lr",
        "weight_decay",
        "trainable_params",
        "elapsed_sec",
        "peak_mib",
        "checkpoint_path",
        "checkpoint_exists",
        "training_log",
    ]
    write_json(OUT / "n4_full_ft_results_table.json", {"rows": result_rows, "summary": {"n_rows": len(result_rows)}})
    write_csv(OUT / "n4_full_ft_results_table.csv", result_rows, result_fields)

    summary_rows = []
    for backbone in BACKBONES:
        for protocol in PROTOCOLS:
            values = [
                float(row["final_metric"])
                for row in result_rows
                if row["backbone"] == backbone and row["protocol"] == protocol and row["status"] == "PASS" and row["final_metric"] is not None
            ]
            row_stats = stats(values)
            summary_rows.append(
                {
                    "backbone": backbone,
                    "protocol": protocol,
                    "n": row_stats["n"],
                    "mean": row_stats["mean"],
                    "std": row_stats["std"],
                    "sem": row_stats["sem"],
                    "metric_note": "raw metric aggregate across regression RMSE and classification AUC; paired bad-space table is direction-aware",
                }
            )
    write_json(OUT / "n4_full_ft_per_backbone_summary.json", {"rows": summary_rows})
    write_csv(OUT / "n4_full_ft_per_backbone_summary.csv", summary_rows, ["backbone", "protocol", "n", "mean", "std", "sem", "metric_note"])

    pair_rows, comparison_summary = fullft_vs_peft(result_rows)
    write_json(
        OUT / "n4_full_ft_vs_peft_paired_comparison.json",
        {"pairs": pair_rows, "summary": comparison_summary},
    )
    write_csv(
        OUT / "n4_full_ft_vs_peft_paired_comparison.csv",
        pair_rows,
        [
            "backbone",
            "protocol",
            "dataset",
            "seed",
            "method_pair",
            "task_type",
            "metric_type",
            "fullft_metric",
            "peft_metric",
            "fullft_bad_minus_peft_bad",
            "direction",
        ],
    )
    write_csv(
        OUT / "n4_full_ft_vs_peft_summary.csv",
        comparison_summary,
        [
            "backbone",
            "protocol",
            "fullft_n",
            "fullft_mean",
            "fullft_sem",
            "lora_n",
            "lora_mean",
            "lora_sem",
            "dora_n",
            "dora_mean",
            "dora_sem",
            "fullft_minus_lora_bad_mean",
            "fullft_minus_dora_bad_mean",
            "direction",
        ],
    )

    audit_rows = []
    for row in result_rows:
        if row["status"] != "PASS":
            audit_rows.append(
                {
                    "backbone": row["backbone"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "protocol": row["protocol"],
                    "checkpoint_path": row["checkpoint_path"],
                    "exists": row["checkpoint_exists"],
                    "expected_effective_method": "full_ft",
                    "matches_spec": False,
                    "reason": "run status not PASS",
                }
            )
        elif row["backbone"] == "cmpnn":
            audit_rows.append(audit_cmpnn(row))
        else:
            audit_rows.append(audit_transformer(row))
    audit_fields = sorted({key for row in audit_rows for key in row.keys()})
    write_json(
        OUT / "n4_full_ft_checkpoint_audit.json",
        {
            "rows": audit_rows,
            "summary": {
                "n_rows": len(audit_rows),
                "matches_spec": sum(1 for row in audit_rows if row.get("matches_spec") is True),
            },
        },
    )
    write_csv(OUT / "n4_full_ft_checkpoint_audit.csv", audit_rows, audit_fields)

    fail_rows = [row for row in result_rows if row["status"] != "PASS"]
    audit_fail_rows = [row for row in audit_rows if row.get("matches_spec") is not True]
    write_json(
        OUT / "n4_full_ft_finalize_summary.json",
        {
            "result_rows": len(result_rows),
            "pass_rows": len(result_rows) - len(fail_rows),
            "fail_rows": fail_rows,
            "audit_rows": len(audit_rows),
            "audit_pass_rows": len(audit_rows) - len(audit_fail_rows),
            "audit_fail_rows": audit_fail_rows,
            "pair_rows": len(pair_rows),
            "comparison_summary": comparison_summary,
        },
    )
    if fail_rows or audit_fail_rows:
        raise RuntimeError({"fail_rows": fail_rows, "audit_fail_rows": audit_fail_rows})


if __name__ == "__main__":
    main()
