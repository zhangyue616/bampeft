# Data Source Instructions

BAM-PEFT does not redistribute dataset CSV files. The experiments used the KANO-provided processed CSV files for six MoleculeNet/DeepChem datasets.

## Source

1. Download the processed data files from the KANO repository: https://github.com/HICAI-ZJU/KANO
2. The original public sources are MoleculeNet and DeepChem.
3. Place the six CSV files in a local directory and set:

```bash
export DATA_ROOT=/path/to/kano_processed_data
```

The scripts expect these file names under `DATA_ROOT`:

| Dataset | File name | Task type | Metric | Exact CSV header |
| --- | --- | --- | --- | --- |
| BACE | `bace.csv` | classification | AUC | `smiles,Class` |
| BBBP | `bbbp.csv` | classification | AUC | `smiles,p_np` |
| ESOL | `esol.csv` | regression | RMSE | `smiles,logSolubility` |
| FreeSolv | `freesolv.csv` | regression | RMSE | `smiles,freesolv` |
| Lipo | `lipo.csv` | regression | RMSE | `smiles,lipo` |
| SIDER | `sider.csv` | multi-label classification | AUC | `smiles` plus the 27 SIDER side-effect target columns from the KANO processed file |

The original moldora working copy stored these files under `data/`; in this public package use `DATA_ROOT` instead.

## Scaffold Splits

The `splits/` directory contains 36 scaffold split JSON files:

```text
{bace,bbbp,esol,freesolv,lipo,sider}_scaffold_seed{0,1,2,10,100,1000}.json
```

Each JSON contains `splits.train`, `splits.val`, and `splits.test` arrays of row indices into the corresponding processed KANO CSV after invalid SMILES filtering. The split sizes are 0.8/0.1/0.1 and the split type is `scaffold_balanced`.

To regenerate splits, use:

```bash
PYTHONPATH="$BAM_REPO_ROOT/src" python "$BAM_REPO_ROOT/scripts/splits/scaffold_make_split.py" \
  --dataset freesolv \
  --seed 0 \
  --data-root "$DATA_ROOT" \
  --out "$BAM_REPO_ROOT/splits/freesolv_scaffold_seed0.json" \
  --split_sizes 0.8 0.1 0.1
```