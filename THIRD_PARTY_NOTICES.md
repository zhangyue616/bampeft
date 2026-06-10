# Third-Party Notices

This file summarizes third-party software, models, datasets, and research artifacts referenced or used by BAM-PEFT.

## KANO

- License: MIT
- Repository: https://github.com/HICAI-ZJU/KANO
- Paper: Fang et al., Nature Machine Intelligence 2023, `@article{fang2023knowledge}`
- Use in this package: BAM-PEFT is based on a KANO fork. The CMPNN pretrained checkpoint and the processed six-dataset CSV files used by the experiments come from KANO.

## chemprop

- License: MIT
- Repository: https://github.com/chemprop/chemprop
- Use in this package: molecular graph learning framework inherited through KANO and adapted by BAM-PEFT.

## torchlight

- License: MIT
- Repository: https://github.com/RamonYeung/torchlight
- Copyright: Copyright (c) 2019 Yang Haihong
- Use in this package: helper infrastructure used by the KANO-derived training code. The original license file is preserved at `src/chemprop/torchlight/LICENSE`.

## KCL

- Repository: https://github.com/ZJU-Fangyin/KCL
- Paper: Fang et al., AAAI 2022
- Use in this package: related work and upstream context for KANO-style molecular representation learning.

## CMPNN

- Paper: Song et al. 2020
- Use in this package: molecular graph encoder architecture used by the KANO-derived CMPNN branch.

## RDKit

- Repository: https://github.com/rdkit/rdkit
- Use in this package: cheminformatics utilities, molecule parsing, scaffold construction, and molecular fingerprints.

## Hugging Face Transformers and Model Checkpoints

- Library: https://github.com/huggingface/transformers
- Models used by scripts: DeepChem/ChemBERTa-77M-MLM and DeepChem/MoLFormer-c3-1.1B
- Use in this package: transformer backbone loading for ChemBERTa-2 and MoLFormer-c3 experiments.

## MoleculeNet and DeepChem

- MoleculeNet paper: Wu et al. 2018
- DeepChem repository: https://github.com/deepchem/deepchem
- Use in this package: original public data sources for BACE, BBBP, ESOL, FreeSolv, Lipo, and SIDER. BAM-PEFT uses the processed KANO versions and does not redistribute the CSV files.