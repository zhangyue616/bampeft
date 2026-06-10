import os
import argparse, csv, json, os, sys
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
if not hasattr(np, 'float'):
    np.float = float

REPO = Path(os.environ.get("BAM_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO))

from chemprop.data.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.utils import filter_invalid_smiles
from chemprop.data.scaffold import scaffold_split

DATASET_PATHS = {
    'freesolv': 'data/freesolv.csv',
    'esol': 'data/esol.csv',
    'lipo': 'data/lipo.csv',
    'bace': 'data/bace.csv',
    'bbbp': 'data/bbbp.csv',
    'sider': 'data/sider.csv',
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=sorted(DATASET_PATHS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-root', default=str(DATA_ROOT))
    p.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    return p.parse_args()

def load_dataset(dataset: str) -> tuple:
    path = Path(DATA_ROOT) / DATASET_PATHS[dataset]
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    points: List[MoleculeDatapoint] = []
    for idx, row in enumerate(rows):
        point = MoleculeDatapoint(line=row, args=Namespace(features_generator=None), use_compound_names=False)
        point.original_index = idx
        points.append(point)
    raw = MoleculeDataset(points)
    filtered = filter_invalid_smiles(raw)
    return header, rows, filtered

def indices(ds: MoleculeDataset) -> List[int]:
    return [int(point.original_index) for point in ds.data]

def main() -> None:
    args = parse_args()
    global DATA_ROOT
    DATA_ROOT = Path(args.data_root).resolve()
    header, rows, data = load_dataset(args.dataset)
    train, val, test = scaffold_split(data, sizes=tuple(args.split_sizes), balanced=True, seed=args.seed, logger=None)
    payload = {
        'dataset': args.dataset,
        'seed': args.seed,
        'split_type': 'scaffold_balanced',
        'split_sizes': list(args.split_sizes),
        'n_rows_raw': len(rows),
        'n_rows_used': len(data),
        'splits': {
            'train': indices(train),
            'val': indices(val),
            'test': indices(test),
        },
    }
    payload['counts'] = {name: len(vals) for name, vals in payload['splits'].items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

if __name__ == '__main__':
    main()
