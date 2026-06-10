# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.train import make_predictions


if __name__ == '__main__':
    repo_root = Path(os.environ.get('BAM_REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
    data_root = Path(os.environ.get('DATA_ROOT', repo_root / 'data')).resolve()
    args = parse_train_args()
    args.checkpoint_path = os.environ.get(
        'BAM_CMPNN_PRETRAINED',
        str(repo_root / 'pretrained' / 'original_CMPN_0623_1350_14000th_epoch.pkl'),
    )
    args.num_tasks = 1
    args.dataset_type = 'classification'
    modify_train_args(args)

    data = pd.read_csv(data_root / 'bbbp.csv')
    pred, smiles = make_predictions(args, data.smiles.tolist())

    df = pd.DataFrame({'smiles': smiles})
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
    df.to_csv(repo_root / 'predict.csv', index=False)