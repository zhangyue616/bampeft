import hashlib
import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, MACCSkeys


class FGFeatureExtractor:
    """RDKit functional-group descriptor extractor with train-only normalization."""

    def __init__(self, descriptor_config_path: str, descriptor_type: str = 'fg'):
        self.descriptor_config_path = descriptor_config_path
        with open(descriptor_config_path, 'rb') as f:
            config_bytes = f.read()
        self.config_hash = hashlib.sha256(config_bytes).hexdigest()
        self.config = json.loads(config_bytes.decode('utf-8'))
        self.descriptor_type = descriptor_type

        self.frag_names = list(self.config['fragments'])
        missing_fragments = [name for name in self.frag_names if not hasattr(Fragments, name)]
        if missing_fragments:
            raise ValueError(f'Unknown RDKit fragment descriptors in FG config: {missing_fragments}')
        self.maccs_indices = [int(i) for i in self.config['maccs_bits']]
        self.smarts = list(self.config['smarts'])
        self.smarts_patterns = [(entry['name'], Chem.MolFromSmarts(entry['pattern'])) for entry in self.smarts]
        if any(pattern is None for _, pattern in self.smarts_patterns):
            bad = [name for name, pattern in self.smarts_patterns if pattern is None]
            raise ValueError(f'Invalid SMARTS patterns in FG descriptor config: {bad}')

        self.normalization_stats: Optional[Dict[str, np.ndarray]] = None
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._shuffle_cache: Dict[str, np.ndarray] = {}

    @property
    def n_features(self) -> int:
        if self.descriptor_type == 'size':
            return 3
        return len(self.frag_names) + len(self.maccs_indices) + len(self.smarts_patterns)

    def _mol_from_smiles(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES for FG descriptor: {smiles!r}')
        return mol

    def _compute_raw(self, mol: Chem.Mol) -> np.ndarray:
        if self.descriptor_type == 'size':
            return np.array([
                Descriptors.MolWt(mol),
                float(mol.GetNumAtoms()),
                float(mol.GetNumHeavyAtoms()),
            ], dtype=np.float32)

        frags = [float(getattr(Fragments, name)(mol)) for name in self.frag_names]
        maccs_all = MACCSkeys.GenMACCSKeys(mol)
        maccs = [float(maccs_all.GetBit(i)) for i in self.maccs_indices]
        smarts_counts = [float(len(mol.GetSubstructMatches(pattern))) for _, pattern in self.smarts_patterns]
        return np.array(frags + maccs + smarts_counts, dtype=np.float32)

    def compute_raw_from_smiles(self, smiles: str) -> np.ndarray:
        cache_key = (smiles, self.config_hash)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_raw(self._mol_from_smiles(smiles))
        return self._cache[cache_key]

    def fit_normalization(self, train_smiles_list: Iterable[str]) -> None:
        """Fit z-score stats on the training split only, then freeze them."""
        rows = [self.compute_raw_from_smiles(smiles) for smiles in train_smiles_list]
        if not rows:
            raise ValueError('Cannot fit FG normalization on an empty training split')
        matrix = np.stack(rows).astype(np.float32)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.normalization_stats = {
            'mean': matrix.mean(axis=0).astype(np.float32),
            'std': std.astype(np.float32),
        }

    def set_normalization(self, mean: List[float], std: List[float]) -> None:
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        if len(mean_arr) != self.n_features or len(std_arr) != self.n_features:
            raise ValueError(
                f'FG normalization length mismatch: expected {self.n_features}, '
                f'got mean={len(mean_arr)} std={len(std_arr)}'
            )
        self.normalization_stats = {'mean': mean_arr, 'std': np.where(std_arr < 1e-8, 1.0, std_arr)}

    def fit_shuffle(self, train_smiles_list: Iterable[str], seed: int) -> None:
        smiles = list(train_smiles_list)
        rows = [self.compute_raw_from_smiles(s).copy() for s in smiles]
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(rows))
        self._shuffle_cache = {smiles[i]: rows[order[i]] for i in range(len(smiles))}

    def compute_from_smiles(self, smiles: str, use_shuffle: bool = False) -> np.ndarray:
        raw = self._shuffle_cache[smiles] if use_shuffle and smiles in self._shuffle_cache else self.compute_raw_from_smiles(smiles)
        if self.normalization_stats is None:
            return raw.astype(np.float32)
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        return ((raw - mean) / (std + 1e-8)).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        mean = []
        std = []
        if self.normalization_stats is not None:
            mean = self.normalization_stats['mean'].astype(float).tolist()
            std = self.normalization_stats['std'].astype(float).tolist()
        return {
            'fg_descriptor_config_hash': self.config_hash,
            'n_fg': self.n_features,
            'fg_normalization_mean': mean,
            'fg_normalization_std': std,
            'descriptor_type': self.descriptor_type,
            'descriptor_config_path': self.descriptor_config_path,
        }


def default_fg_config_path() -> str:
    import os
    return os.path.join(os.path.dirname(__file__), 'fg_descriptor_config.json')
