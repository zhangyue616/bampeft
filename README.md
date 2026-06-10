# BAM-PEFT

BAM-PEFT is a benchmark and audit package for parameter-efficient fine-tuning of molecular foundation and graph neural network backbones. The released code supports the controlled LoRA/DoRA/full fine-tuning comparisons, scaffold split reuse, bootstrap confidence intervals, and F1/F2/F3 spectral audit source-data reproduction used in the associated manuscript.

## Attribution

This project is based on a fork of KANO (HICAI-ZJU/KANO; Fang et al., Nature Machine Intelligence 2023; MIT license). BAM-PEFT adds LoRA/DoRA fine-tuning, benchmark orchestration, held-out audit summaries, and spectral audit utilities on top of the KANO-derived codebase. The packaged CMPNN pretrained checkpoint in `pretrained/original_CMPN_0623_1350_14000th_epoch.pkl` is from KANO and is included for reproducibility. The processed molecular datasets used by the experiments are the KANO-provided processed versions of MoleculeNet/DeepChem datasets; the CSV files are not redistributed in this package.

## Layout

- `src/`: cleaned chemprop/KANO-derived training code and BAM-PEFT PEFT additions.
- `scripts/`: cleaned experiment, split, bootstrap, and spectral-audit scripts.
- `splits/`: 36 scaffold split JSON files for six datasets and seeds 0, 1, 2, 10, 100, and 1000.
- `data_source/README.md`: data acquisition, required schema, and placement instructions.
- `results/`: minimal locked CSV/JSON source results for manuscript figures and tables.
- `envs/`: environment requirement snapshots for the five backbone environments.
- `pretrained/`: KANO CMPNN pretrained checkpoint used by the CMPNN fine-tuning scripts.
- `THIRD_PARTY_NOTICES.md`: third-party attribution and license notes.
- `MANIFEST.sha256`: SHA256 manifest for all packaged files except the manifest itself.

## Environments

Install the requirement file matching the backbone to reproduce:

- CMPNN: `envs/requirements_kapt-5090.txt`
- ChemBERTa-2: `envs/requirements_kapt-chemberta2.txt`
- MoLFormer-c3: `envs/requirements_kapt-molformer-c3.txt`
- GraphMVP-GIN held-out summaries: `envs/requirements_kapt_graphmvp.txt`
- Uni-Mol2-84M held-out summaries: `envs/requirements_kapt_unimol2.txt`

Set these paths before running scripts:

```bash
export BAM_REPO_ROOT=/path/to/BAM
export DATA_ROOT=/path/to/kano_processed_data
export BAM_CMPNN_PRETRAINED="$BAM_REPO_ROOT/pretrained/original_CMPN_0623_1350_14000th_epoch.pkl"
export BAM_TMP_DIR="$BAM_REPO_ROOT/outputs/tmp"
```

For Uni-Mol2-specific reruns not included in this lightweight script set, use `UNIMOL2_CHECKPOINT` to point to the external Uni-Mol2 checkpoint. The locked held-out result tables are already included under `results/spec_a_r3_production/` and `results/spec_b_v3/`.

## Data

This package does not redistribute the six processed CSV files. Download the KANO processed data from the KANO repository and place the files as described in `data_source/README.md`. The original dataset sources are MoleculeNet and DeepChem.

## Reproduction Sketch

1. Prepare an environment from `envs/` and activate it.
2. Set `BAM_REPO_ROOT`, `DATA_ROOT`, `BAM_CMPNN_PRETRAINED`, and optionally `BAM_TMP_DIR`.
3. Reuse the scaffold splits in `splits/` for scaffold-primary runs.
4. Use `scripts/b2_stage2/` for transformer LoRA/DoRA runs.
5. Use `scripts/n4_full_ft/` for Full FT baselines.
6. Use `scripts/n5_rank_sensitivity/` for rank sensitivity.
7. Use `scripts/bootstrap_ci/` to recompute paired bootstrap summaries from locked paired-difference JSON files.
8. Use `scripts/r1_signature/` for F1/F2/F3 spectral audit extraction and correlation summaries.

The `results/` directory contains the locked source tables used to render manuscript figures and tables. Figure-rendering scripts are intentionally not included because they are coupled to the manuscript build tree.

## Citation

Please cite the BAM-PEFT manuscript when available. Also cite KANO for the base fork, checkpoint, and processed-data source:

```bibtex
@article{fang2023knowledge,
  title={Knowledge graph-enhanced molecular contrastive learning with functional prompt},
  author={Fang, Yin and Zhang, Qiang and Zhang, Ningyu and Chen, Zhuo and Zhuang, Xiang and Shao, Xin and Fan, Xiaohui and Chen, Huajun},
  journal={Nature Machine Intelligence},
  year={2023}
}
```

## License

BAM-PEFT additions are released under the MIT license. Third-party components retain their original licenses and attribution requirements; see `THIRD_PARTY_NOTICES.md` and `src/chemprop/torchlight/LICENSE`.