import csv
from pathlib import Path
from typing import Dict, List

from b2_phase4_common import BACKBONE_ORDER, DATASET_ORDER, METHODS, REPO_ROOT, SEEDS, output_dir, sha256_file, write_json


OUT_JSON = REPO_ROOT / "docs/stage3/b2_phase4_split_audit.json"
OUT_CSV = REPO_ROOT / "docs/stage3/b2_phase4_split_audit.csv"


def main() -> None:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for dataset in DATASET_ORDER:
        for seed in SEEDS:
            split_records: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
            paths_by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
            for backbone in BACKBONE_ORDER:
                for method in METHODS:
                    split_dir = output_dir(dataset, method, backbone, seed) / "split"
                    for split in ("train", "val", "test"):
                        path = split_dir / f"{split}_smiles.txt"
                        if not path.exists():
                            missing.append(str(path))
                            continue
                        split_records[split].append(sha256_file(path))
                        paths_by_split[split].append(str(path))
            row = {
                "dataset": dataset,
                "seed": seed,
                "train_sha256_unique": sorted(set(split_records["train"])),
                "val_sha256_unique": sorted(set(split_records["val"])),
                "test_sha256_unique": sorted(set(split_records["test"])),
                "train_file_count": len(split_records["train"]),
                "val_file_count": len(split_records["val"]),
                "test_file_count": len(split_records["test"]),
            }
            row["byte_identical"] = (
                row["train_file_count"] == 6
                and row["val_file_count"] == 6
                and row["test_file_count"] == 6
                and len(row["train_sha256_unique"]) == 1
                and len(row["val_sha256_unique"]) == 1
                and len(row["test_sha256_unique"]) == 1
            )
            rows.append(row)
            print(f"{dataset} seed{seed}: byte_identical={row['byte_identical']}", flush=True)
    payload = {"rows": rows, "missing": missing, "pass_count": sum(1 for r in rows if r["byte_identical"])}
    write_json(OUT_JSON, payload)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "seed",
            "byte_identical",
            "train_file_count",
            "val_file_count",
            "test_file_count",
            "train_sha256_unique",
            "val_sha256_unique",
            "test_sha256_unique",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    if missing:
        raise SystemExit(f"SPLIT_AUDIT_FAIL missing={len(missing)}")
    if payload["pass_count"] != len(rows):
        raise SystemExit(f"SPLIT_AUDIT_FAIL byte_identical={payload['pass_count']}/{len(rows)}")
    print(f"SPLIT_AUDIT_PASS {payload['pass_count']}/{len(rows)}")


if __name__ == "__main__":
    main()
