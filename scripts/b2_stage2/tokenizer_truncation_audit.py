import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

from b2_phase4_common import BACKBONES, DATASET_ORDER, REPO_ROOT, read_dataset, write_json


OUT_JSON = REPO_ROOT / "docs/stage3/b2_phase4_tokenizer_truncation_audit.json"
OUT_CSV = REPO_ROOT / "docs/stage3/b2_phase4_tokenizer_truncation_audit.csv"


def encode_lengths(tokenizer, smiles: List[str], chunk_size: int = 128) -> Dict[str, object]:
    lengths: List[int] = []
    unk_examples: List[str] = []
    unk_count = 0
    unk_id = tokenizer.unk_token_id
    for start in range(0, len(smiles), chunk_size):
        chunk = smiles[start : start + chunk_size]
        encoded = tokenizer(chunk, padding=False, truncation=False, return_tensors=None)
        for smi, ids in zip(chunk, encoded["input_ids"]):
            lengths.append(len(ids))
            if unk_id is not None:
                count = ids.count(unk_id)
                unk_count += count
                if count and len(unk_examples) < 5:
                    unk_examples.append(smi)
    arr = np.array(lengths, dtype=np.int64)
    return {
        "lengths": lengths,
        "min": int(arr.min()),
        "median": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "unk_token_count": int(unk_count),
        "unk_examples": unk_examples,
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for backbone, spec in BACKBONES.items():
        tokenizer = AutoTokenizer.from_pretrained(spec["hf_path"], trust_remote_code=spec["trust_remote_code"])
        native_max_len = int(spec["native_max_len"])
        for dataset_name in DATASET_ORDER:
            dataset = read_dataset(dataset_name)
            stats = encode_lengths(tokenizer, dataset.smiles)
            over_limit_count = sum(1 for value in stats["lengths"] if value > native_max_len)
            row = {
                "backbone": backbone,
                "dataset": dataset_name,
                "n_samples": len(dataset.smiles),
                "native_max_len": native_max_len,
                "token_len_min": stats["min"],
                "token_len_median": stats["median"],
                "token_len_p95": stats["p95"],
                "token_len_p99": stats["p99"],
                "token_len_max": stats["max"],
                "over_limit_count": over_limit_count,
                "over_limit_rate": over_limit_count / len(dataset.smiles),
                "unk_token_count": stats["unk_token_count"],
                "unk_rate": stats["unk_token_count"] / len(dataset.smiles),
                "unk_examples": stats["unk_examples"],
                "chunked_execution": True,
            }
            print(
                f"{backbone} {dataset_name}: p99={row['token_len_p99']} max={row['token_len_max']} "
                f"over={row['over_limit_count']} unk={row['unk_token_count']}",
                flush=True,
            )
            rows.append(row)
    write_json(OUT_JSON, rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "backbone",
            "dataset",
            "n_samples",
            "native_max_len",
            "token_len_p95",
            "token_len_p99",
            "token_len_max",
            "over_limit_count",
            "over_limit_rate",
            "unk_token_count",
            "unk_rate",
            "chunked_execution",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"WROTE {OUT_JSON}")


if __name__ == "__main__":
    main()
