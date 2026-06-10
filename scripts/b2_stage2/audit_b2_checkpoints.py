import csv
from pathlib import Path
from typing import Dict, List

import torch

from b2_phase4_common import BACKBONES, BACKBONE_ORDER, DATASET_ORDER, METHODS, REPO_ROOT, SEEDS, output_dir, write_json


OUT_JSON = REPO_ROOT / "docs/stage3/b2_phase4_checkpoint_audit.json"
OUT_CSV = REPO_ROOT / "docs/stage3/b2_phase4_checkpoint_audit.csv"


EXPECTED_M = {"chemberta2": 9, "molformer_c3": 36}


def count_keys(state_dict: Dict[str, object], needle: str) -> int:
    return sum(1 for key in state_dict if needle in key)


def audit_one(dataset: str, method: str, backbone: str, seed: int) -> Dict[str, object]:
    ckpt_path = output_dir(dataset, method, backbone, seed) / "model.pt"
    row: Dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "method": method,
        "backbone": backbone,
        "seed": seed,
        "exists": ckpt_path.exists(),
    }
    if not ckpt_path.exists():
        row["matches_spec"] = False
        row["reason"] = "missing checkpoint"
        return row
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    top_keys = sorted(state.keys())
    state_dict = state.get("state_dict", {})
    lora_config = state.get("lora_config", {})
    provenance = state.get("provenance", {})
    lora_a = count_keys(state_dict, "lora_A")
    lora_b = count_keys(state_dict, "lora_B")
    lora_m = count_keys(state_dict, "lora_magnitude_vector")
    expected_use_dora = method == "dora"
    expected_targets = [] if method == "head" else ["query", "key", "value"]
    expected_m = EXPECTED_M[backbone] if method == "dora" else 0
    row.update(
        {
            "top_keys": ",".join(top_keys),
            "use_dora": lora_config.get("use_dora"),
            "target_modules": ",".join(lora_config.get("target_modules", [])),
            "lora_A_count": lora_a,
            "lora_B_count": lora_b,
            "lora_m_count": lora_m,
            "DeepChem_rev": provenance.get("hf_model_revision"),
            "IBM_rev": provenance.get("remote_code_revision"),
            "transformers": provenance.get("transformers"),
            "peft": provenance.get("peft"),
            "torch": provenance.get("torch"),
            "env": provenance.get("conda_env"),
        }
    )
    checks = [
        {"args", "state_dict", "data_scaler", "features_scaler", "lora_config", "provenance"}.issubset(set(top_keys)),
        lora_config.get("peft_method") == method,
        lora_config.get("use_dora") == expected_use_dora,
        lora_config.get("target_modules") == expected_targets,
        provenance.get("hf_model_revision") == BACKBONES[backbone]["model_revision"],
        provenance.get("remote_code_revision") == BACKBONES[backbone]["remote_code_revision"],
    ]
    if method == "head":
        checks.append(lora_a == 0 and lora_b == 0 and lora_m == 0)
    elif method == "lora":
        checks.append(lora_a > 0 and lora_b > 0 and lora_m == 0)
    else:
        checks.append(lora_a > 0 and lora_b > 0 and lora_m == expected_m)
    row["matches_spec"] = all(checks)
    row["reason"] = "pass" if row["matches_spec"] else "schema/content mismatch"
    return row


def main() -> None:
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for seed in SEEDS:
            for backbone in BACKBONE_ORDER:
                for method in METHODS:
                    row = audit_one(dataset, method, backbone, seed)
                    rows.append(row)
                    print(
                        f"{dataset} seed{seed} {backbone} {method}: "
                        f"matches={row['matches_spec']} A={row.get('lora_A_count')} "
                        f"B={row.get('lora_B_count')} M={row.get('lora_m_count')}",
                        flush=True,
                    )
    pass_count = sum(1 for row in rows if row["matches_spec"])
    write_json(OUT_JSON, {"rows": rows, "pass_count": pass_count, "total": len(rows)})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "checkpoint",
            "method",
            "backbone",
            "dataset",
            "seed",
            "use_dora",
            "target_modules",
            "lora_A_count",
            "lora_B_count",
            "lora_m_count",
            "DeepChem_rev",
            "IBM_rev",
            "transformers",
            "peft",
            "torch",
            "env",
            "matches_spec",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    if pass_count != len(rows):
        raise SystemExit(f"CHECKPOINT_AUDIT_FAIL {pass_count}/{len(rows)}")
    print(f"CHECKPOINT_AUDIT_PASS {pass_count}/{len(rows)}")


if __name__ == "__main__":
    main()
