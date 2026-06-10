#!/usr/bin/env python
"""BAM-PEFT R1 signature extraction.

This script is intentionally self-contained because CMPNN, ChemBERTa-2, and
MoLFormer-c3 require different local environments in this workspace.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ORDER = ("freesolv", "esol", "lipo", "bace", "bbbp", "sider")
F1_SEEDS = (0, 1, 2)
F3_SEED = 0
PROTOCOL = "scaffold"
F1_MAX_ROWS = 5000
EPS = 1e-12
CORRELATION_POLICY = {
    "cmpnn_freesolv_n4_110": "use_n5_mean_no_throwout",
}

DATASETS = {
    "freesolv": {
        "path": "data/freesolv.csv",
        "smiles_col": "smiles",
        "task": "regression",
    },
    "esol": {
        "path": "data/esol.csv",
        "smiles_col": "smiles",
        "task": "regression",
    },
    "lipo": {
        "path": "data/lipo.csv",
        "smiles_col": "smiles",
        "task": "regression",
    },
    "bace": {
        "path": "data/bace.csv",
        "smiles_col": "smiles",
        "task": "classification",
    },
    "bbbp": {
        "path": "data/bbbp.csv",
        "smiles_col": "smiles",
        "task": "classification",
    },
    "sider": {
        "path": "data/sider.csv",
        "smiles_col": "smiles",
        "task": "classification",
    },
}

TRANSFORMER_BACKBONES = {
    "chemberta2": {
        "hf_path": "DeepChem/ChemBERTa-77M-MLM",
        "model_revision": "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "trust_remote_code": False,
        "native_max_len": 512,
    },
    "molformer_c3": {
        "hf_path": "DeepChem/MoLFormer-c3-1.1B",
        "model_revision": "9f1b9ea3590833bd0ea1a70e789c5d3da11ba7ed",
        "remote_code_revision": "7b12d946c181a37f6012b9dc3b002275de070314",
        "remote_code_source": "ibm/MoLFormer-XL-both-10pct",
        "trust_remote_code": True,
        "native_max_len": 202,
    },
}

CMPNN_TARGETS = (
    {"name": "W_i_atom", "path": "encoder.cmpn.encoder.W_i_atom", "group": "cmpnn_W_i_atom", "index": -1},
    {"name": "W_i_bond", "path": "encoder.cmpn.encoder.W_i_bond", "group": "cmpnn_W_i_bond", "index": -1},
    {"name": "W_h_0", "path": "encoder.cmpn.encoder.W_h_0", "group": "cmpnn_W_h_0", "index": -1},
    {"name": "W_h_1", "path": "encoder.cmpn.encoder.W_h_1", "group": "cmpnn_W_h_1", "index": -1},
    {"name": "W_o", "path": "encoder.cmpn.encoder.W_o", "group": "cmpnn_W_o", "index": -1},
    {"name": "lr", "path": "encoder.cmpn.encoder.lr", "group": "cmpnn_lr", "index": -1},
)


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message):
    print("%s %s" % (timestamp(), message), flush=True)


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("cmpnn", "chemberta2", "molformer_c3"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "docs" / "stage2"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--f1-batch-size", type=int, default=64)
    parser.add_argument("--f3-batch-size", type=int, default=None)
    parser.add_argument("--skip-f1", action="store_true")
    parser.add_argument("--skip-f2", action="store_true")
    parser.add_argument("--skip-f3", action="store_true")
    return parser.parse_args()


def read_csv_rows(path):
    with open(str(path), "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    ensure_dir(path.parent)
    with open(str(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_or_write_csv(path, rows, fieldnames):
    ensure_dir(path.parent)
    exists = path.exists()
    with open(str(path), "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_dataset(dataset_name):
    spec = DATASETS[dataset_name]
    path = REPO_ROOT / spec["path"]
    with open(str(path), "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    smiles_col = spec["smiles_col"]
    target_cols = [col for col in fieldnames if col != smiles_col]
    smiles = [row[smiles_col] for row in rows]
    targets = []
    for row in rows:
        target_row = []
        for col in target_cols:
            value = row[col]
            target_row.append(float(value) if value != "" else float("nan"))
        targets.append(target_row)
    return {
        "smiles": smiles,
        "targets": np.asarray(targets, dtype=np.float32),
        "target_cols": target_cols,
        "task": spec["task"],
    }


def load_split(dataset_name, seed):
    candidates = [
        REPO_ROOT / "dumped" / "n4_full_ft_splits" / ("%s_scaffold_seed%d.json" % (dataset_name, seed)),
        REPO_ROOT / "dumped" / "n5_rank_sensitivity_splits" / ("%s_scaffold_seed%d.json" % (dataset_name, seed)),
    ]
    for path in candidates:
        if path.exists():
            with open(str(path), "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data["splits"], str(path.relative_to(REPO_ROOT))
    raise FileNotFoundError("No scaffold split JSON found for %s seed %d" % (dataset_name, seed))


def take_indices(values, indices):
    return [values[i] for i in indices]


def take_target_rows(targets, indices):
    return targets[np.asarray(indices, dtype=np.int64)]


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def compute_scaler(train_targets, task):
    if task != "regression":
        return None
    means = np.nanmean(train_targets, axis=0)
    stds = np.nanstd(train_targets, axis=0)
    stds = np.where(stds < EPS, 1.0, stds)
    return {"means": means.astype(np.float32), "stds": stds.astype(np.float32)}


def apply_scaler(targets, scaler):
    if scaler is None:
        return targets
    return (targets - scaler["means"]) / scaler["stds"]


def as_float(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return "%.10g" % value


def svd_values(matrix, center=False):
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    if center:
        array = array - np.nanmean(array, axis=0, keepdims=True)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.svd(array, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(array + 1e-8 * np.random.default_rng(0).standard_normal(array.shape), compute_uv=False)


def energy_fractions(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    energy = s * s
    total = float(energy.sum())
    if total <= EPS:
        return 0.0, 0.0, 0.0
    return (
        float(energy[:4].sum() / total),
        float(energy[:8].sum() / total),
        float(energy[:16].sum() / total),
    )


def entropy_rank(singular_values, squared=False):
    s = np.asarray(singular_values, dtype=np.float64)
    if squared:
        values = s * s
    else:
        values = s
    total = float(values.sum())
    if total <= EPS:
        return 0.0, 0.0
    probs = values / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, 1.0))))
    return float(math.exp(entropy)), entropy


def participation_ratio(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    energy = s * s
    denom = float(np.sum(energy * energy))
    if denom <= EPS:
        return 0.0
    return float((energy.sum() ** 2) / denom)


def stable_rank(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size == 0 or float(s[0] * s[0]) <= EPS:
        return 0.0
    return float(np.sum(s * s) / (s[0] * s[0]))


def tail_slope(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size < 4:
        return 0.0
    start = max(1, s.size // 2)
    idx = np.arange(start + 1, s.size + 1, dtype=np.float64)
    tail = np.clip(s[start:], EPS, None)
    if tail.size < 2:
        return 0.0
    slope = np.polyfit(np.log(idx), np.log(tail), 1)[0]
    return float(slope)


def condition_number(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size == 0:
        return 0.0
    denom = max(float(s[-1]), EPS)
    value = float(s[0] / denom)
    return min(value, 1e12)


def activation_feature_row(base, matrix, n_seen=None):
    arr = np.asarray(matrix, dtype=np.float64)
    s = svd_values(arr, center=True)
    top4, top8, top16 = energy_fractions(s)
    erank, entropy = entropy_rank(s, squared=False)
    pr = participation_ratio(s)
    if s.size == 0 or float(np.mean(s * s)) <= EPS:
        anisotropy = 0.0
    else:
        anisotropy = float((s[0] * s[0]) / np.mean(s * s))
    row = OrderedDict(base)
    row.update(
        {
            "n_samples_seen": int(n_seen if n_seen is not None else arr.shape[0]),
            "matrix_rows": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "matrix_cols": int(arr.shape[1]) if arr.ndim >= 2 else 0,
            "erank": as_float(erank),
            "participation_ratio": as_float(pr),
            "top4_energy": as_float(top4),
            "top8_energy": as_float(top8),
            "top16_energy": as_float(top16),
            "anisotropy": as_float(anisotropy),
            "spectral_entropy": as_float(entropy),
        }
    )
    return row, s.astype(np.float32)


def weight_feature_row(base, weight):
    arr = np.asarray(weight, dtype=np.float64)
    s = svd_values(arr, center=False)
    top4, top8, top16 = energy_fractions(s)
    erank, entropy = entropy_rank(s, squared=False)
    row = OrderedDict(base)
    row.update(
        {
            "matrix_rows": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "matrix_cols": int(arr.shape[1]) if arr.ndim >= 2 else 0,
            "erank": as_float(erank),
            "stable_rank": as_float(stable_rank(s)),
            "top4_energy": as_float(top4),
            "top8_energy": as_float(top8),
            "top16_energy": as_float(top16),
            "tail_slope": as_float(tail_slope(s)),
            "condition_number": as_float(condition_number(s)),
            "sv_entropy": as_float(entropy),
        }
    )
    return row, s.astype(np.float32)


def gradient_feature_row(base, grad, weight, target_trace, denominator_trace):
    grad_arr = np.asarray(grad, dtype=np.float64)
    weight_arr = np.asarray(weight, dtype=np.float64)
    s = svd_values(grad_arr, center=False)
    top4, top8, top16 = energy_fractions(s)
    erank, entropy = entropy_rank(s, squared=False)
    align4 = alignment_fraction(grad_arr, weight_arr, 4)
    align8 = alignment_fraction(grad_arr, weight_arr, 8)
    align16 = alignment_fraction(grad_arr, weight_arr, 16)
    if denominator_trace <= EPS:
        fisher_fraction = 0.0
    else:
        fisher_fraction = float(target_trace / denominator_trace)
    row = OrderedDict(base)
    row.update(
        {
            "matrix_rows": int(grad_arr.shape[0]) if grad_arr.ndim >= 1 else 0,
            "matrix_cols": int(grad_arr.shape[1]) if grad_arr.ndim >= 2 else 0,
            "grad_erank": as_float(erank),
            "grad_stable_rank": as_float(stable_rank(s)),
            "grad_top4_energy": as_float(top4),
            "grad_top8_energy": as_float(top8),
            "grad_top16_energy": as_float(top16),
            "grad_sv_entropy": as_float(entropy),
            "alignment_A4": as_float(align4),
            "alignment_A8": as_float(align8),
            "alignment_A16": as_float(align16),
            "fisher_target_fraction": as_float(fisher_fraction),
            "fisher_layer_trace": as_float(target_trace),
        }
    )
    return row, s.astype(np.float32)


def alignment_fraction(grad, weight, rank):
    grad_arr = np.asarray(grad, dtype=np.float64)
    weight_arr = np.asarray(weight, dtype=np.float64)
    denom = float(np.sum(grad_arr * grad_arr))
    if denom <= EPS:
        return 0.0
    try:
        u, _, vt = np.linalg.svd(weight_arr, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    k = min(rank, u.shape[1], vt.shape[0])
    if k <= 0:
        return 0.0
    projected = np.dot(u[:, :k].T, np.dot(grad_arr, vt[:k, :].T))
    value = float(np.sum(projected * projected) / denom)
    return max(0.0, min(1.0, value))


class ReservoirCollector(object):
    def __init__(self, max_rows, seed):
        self.max_rows = int(max_rows)
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.n_seen = 0
        self.n_cols = None

    def add(self, value):
        arr = value.detach().float().cpu().numpy()
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[0] == 0:
            return
        if self.n_cols is None:
            self.n_cols = arr.shape[1]
        if arr.shape[1] != self.n_cols:
            raise ValueError("Cannot aggregate activations with feature dims %d and %d" % (self.n_cols, arr.shape[1]))
        for row in arr:
            self.n_seen += 1
            if len(self.rows) < self.max_rows:
                self.rows.append(row.copy())
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.max_rows:
                    self.rows[j] = row.copy()

    def matrix(self):
        if not self.rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(self.rows).astype(np.float32)


def module_lookup(model, path):
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError("Module path not found: %s" % path)
    return modules[path]


def tensor_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def model_weight_sha256(model):
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, param in model.named_parameters():
            digest.update(name.encode("utf-8"))
            arr = param.detach().cpu().contiguous().numpy()
            digest.update(arr.tobytes())
    return digest.hexdigest()


def extract_step0_gradient(model, target_modules, batch, loss_fn, head_modules=None):
    """Independent Stage 2 F3 extraction with strict weight integrity checks."""
    pre_hash = model_weight_sha256(model)
    original_flags = [(param, param.requires_grad) for param in model.parameters()]
    try:
        model.zero_grad(set_to_none=True)
        for param, _ in original_flags:
            param.requires_grad_(False)
        for module in target_modules.values():
            module.weight.requires_grad_(True)
        if head_modules is not None:
            for head in head_modules:
                for param in head.parameters():
                    param.requires_grad_(True)
        loss = loss_fn(model, batch)
        loss.backward()
        gradients = OrderedDict()
        weights = OrderedDict()
        traces = OrderedDict()
        for name, module in target_modules.items():
            if module.weight.grad is None:
                grad = torch.zeros_like(module.weight.detach())
            else:
                grad = module.weight.grad.detach().cpu().clone()
            gradients[name] = grad
            weights[name] = module.weight.detach().cpu().clone()
            traces[name] = float(torch.sum(grad.float() * grad.float()).item())
        if head_modules is None:
            head_trace = 0.0
        else:
            head_trace = 0.0
            for head in head_modules:
                for param in head.parameters():
                    if param.grad is not None:
                        head_trace += float(torch.sum(param.grad.detach().float() * param.grad.detach().float()).item())
        model.zero_grad(set_to_none=True)
    finally:
        for param, flag in original_flags:
            param.requires_grad_(flag)
    post_hash = model_weight_sha256(model)
    if pre_hash != post_hash:
        raise AssertionError("F3 weight integrity check failed: pre/post SHA256 mismatch")
    return {
        "loss": float(loss.detach().cpu().item()),
        "gradients": gradients,
        "weights": weights,
        "traces": traces,
        "head_trace": head_trace,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
    }


class SequenceHeadModel(nn.Module):
    def __init__(self, backbone, hidden_size, out_dim):
        super(SequenceHeadModel, self).__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def transformer_target_specs(backbone_name, with_wrapper_prefix=False):
    prefix = "backbone." if with_wrapper_prefix else ""
    specs = []
    n_layers = 3 if backbone_name == "chemberta2" else 12
    parts = [
        ("attention.self.query", "attention_query", "query", True),
        ("attention.self.key", "attention_key", "key", True),
        ("attention.self.value", "attention_value", "value", True),
        ("attention.output.dense", "attention_output", "attention_output_dense", False),
        ("intermediate.dense", "ffn_intermediate", "ffn_intermediate_dense", False),
        ("output.dense", "ffn_output", "ffn_output_dense", False),
    ]
    for layer_idx in range(n_layers):
        for suffix, group, label, historical in parts:
            bare_path = "encoder.layer.%d.%s" % (layer_idx, suffix)
            specs.append(
                {
                    "name": bare_path,
                    "path": prefix + bare_path,
                    "group": group,
                    "index": layer_idx,
                    "target_kind": label,
                    "historical": historical,
                }
            )
    return specs


def target_specs(backbone_name, with_wrapper_prefix=False):
    if backbone_name == "cmpnn":
        return list(CMPNN_TARGETS)
    return transformer_target_specs(backbone_name, with_wrapper_prefix=with_wrapper_prefix)


def target_set_specs(backbone_name, target_set, with_wrapper_prefix=False):
    specs = target_specs(backbone_name, with_wrapper_prefix=with_wrapper_prefix)
    if backbone_name == "cmpnn":
        return specs
    if target_set == "historical":
        return [spec for spec in specs if spec["historical"]]
    if target_set == "expanded":
        return specs
    raise ValueError("Unknown target set: %s" % target_set)


def build_cmpnn_model(num_tasks, dataset_type, device):
    from argparse import Namespace

    from chemprop.models import build_model

    args = Namespace(
        hidden_size=300,
        depth=3,
        num_tasks=num_tasks,
        dataset_type=dataset_type,
        dropout=0.0,
        activation="ReLU",
        atom_messages=False,
        undirected=False,
        features_only=False,
        use_input_features=False,
        features_dim=0,
        features_size=0,
        multiclass_num_classes=3,
        ffn_hidden_size=300,
        ffn_num_layers=2,
        bias=False,
        cuda=False,
        atom_fdim=133,
        bond_fdim=147,
    )
    model = build_model(args, "CMPNN")
    checkpoint_path = Path(os.environ.get("BAM_CMPNN_PRETRAINED", REPO_ROOT / "pretrained" / "original_CMPN_0623_1350_14000th_epoch.pkl"))
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.encoder.load_state_dict(state, strict=False)
    model.to(device)
    return model


def load_transformer_backbone(backbone_name, out_dim, device):
    from transformers import AutoModel, AutoTokenizer

    cfg = TRANSFORMER_BACKBONES[backbone_name]
    common = {
        "local_files_only": True,
        "trust_remote_code": cfg["trust_remote_code"],
        "revision": cfg["model_revision"],
    }
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_path"], **common)
    backbone = AutoModel.from_pretrained(cfg["hf_path"], **common)
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    model = SequenceHeadModel(backbone=backbone, hidden_size=hidden_size, out_dim=out_dim)
    model.to(device)
    return tokenizer, model, cfg["native_max_len"]


def encode_batch(tokenizer, smiles, max_len, device):
    encoded = tokenizer(
        list(smiles),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def masked_loss_tensor(logits, targets, task):
    mask = torch.isfinite(targets)
    if task == "regression":
        logits = logits.view_as(targets)
        diff = (logits - torch.nan_to_num(targets, nan=0.0))[mask]
        if diff.numel() == 0:
            return torch.sum(logits * 0.0)
        return torch.mean(diff * diff)
    target = torch.nan_to_num(targets, nan=0.0)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask.sum().item() == 0:
        return torch.sum(logits * 0.0)
    return raw[mask].mean()


def run_cmpnn_f1(model, dataset_name, dataset, train_indices, seed, split_source, output_dir):
    model.eval()
    train_smiles = take_indices(dataset["smiles"], train_indices)
    rows = []
    svd_payload = {}
    collectors = OrderedDict()
    hooks = []
    for spec in target_set_specs("cmpnn", "historical"):
        collector = ReservoirCollector(F1_MAX_ROWS, seed=1000 + seed + len(collectors))
        collectors[spec["name"]] = collector
        module = module_lookup(model, spec["path"])
        hooks.append(module.register_forward_pre_hook(lambda module, inputs, c=collector: c.add(inputs[0])))
    penultimate = ReservoirCollector(F1_MAX_ROWS, seed=2000 + seed)
    try:
        with torch.no_grad():
            for _, smiles_batch in batches(train_smiles, 64):
                mol_vecs, _, _ = model.encoder(smiles_batch)
                penultimate.add(mol_vecs)
    finally:
        for hook in hooks:
            hook.remove()

    base = {
        "backbone": "cmpnn",
        "dataset": dataset_name,
        "protocol": PROTOCOL,
        "seed": seed,
        "target_set": "historical",
        "layer_name": "penultimate_mol_vec",
        "layer_group": "penultimate",
        "layer_index": -1,
        "split_source": split_source,
    }
    row, singulars = activation_feature_row(base, penultimate.matrix(), n_seen=penultimate.n_seen)
    rows.append(row)
    svd_payload["penultimate_mol_vec"] = singulars

    for spec in target_set_specs("cmpnn", "historical"):
        collector = collectors[spec["name"]]
        base = {
            "backbone": "cmpnn",
            "dataset": dataset_name,
            "protocol": PROTOCOL,
            "seed": seed,
            "target_set": "historical",
            "layer_name": spec["name"],
            "layer_group": spec["group"],
            "layer_index": spec["index"],
            "split_source": split_source,
        }
        row, singulars = activation_feature_row(base, collector.matrix(), n_seen=collector.n_seen)
        rows.append(row)
        svd_payload[spec["name"]] = singulars
    npz_path = output_dir / "intermediates" / ("f1_cmpnn_%s_seed%d.npz" % (dataset_name, seed))
    np.savez_compressed(str(npz_path), **svd_payload)
    return rows


def run_transformer_f1(backbone_name, tokenizer, model, max_len, dataset_name, dataset, train_indices, seed, split_source, output_dir, batch_size, device):
    model.eval()
    train_smiles = take_indices(dataset["smiles"], train_indices)
    rows = []
    svd_payload = {}
    historical_collector = ReservoirCollector(F1_MAX_ROWS, seed=3000 + seed)
    hooks = []
    for spec in target_set_specs(backbone_name, "historical", with_wrapper_prefix=True):
        module = module_lookup(model, spec["path"])
        hooks.append(module.register_forward_pre_hook(lambda module, inputs, c=historical_collector: c.add(inputs[0])))
    penultimate = ReservoirCollector(F1_MAX_ROWS, seed=4000 + seed)
    try:
        with torch.no_grad():
            for _, smiles_batch in batches(train_smiles, batch_size):
                encoded = encode_batch(tokenizer, smiles_batch, max_len, device)
                pooled = model.encode(encoded["input_ids"], encoded["attention_mask"])
                penultimate.add(pooled)
    finally:
        for hook in hooks:
            hook.remove()
    base = {
        "backbone": backbone_name,
        "dataset": dataset_name,
        "protocol": PROTOCOL,
        "seed": seed,
        "target_set": "historical",
        "layer_name": "penultimate_sequence_pool",
        "layer_group": "penultimate",
        "layer_index": -1,
        "split_source": split_source,
    }
    row, singulars = activation_feature_row(base, penultimate.matrix(), n_seen=penultimate.n_seen)
    rows.append(row)
    svd_payload["penultimate_sequence_pool"] = singulars
    base = {
        "backbone": backbone_name,
        "dataset": dataset_name,
        "protocol": PROTOCOL,
        "seed": seed,
        "target_set": "historical",
        "layer_name": "historical_qkv_aggregate",
        "layer_group": "historical_qkv_aggregate",
        "layer_index": -1,
        "split_source": split_source,
    }
    row, singulars = activation_feature_row(base, historical_collector.matrix(), n_seen=historical_collector.n_seen)
    rows.append(row)
    svd_payload["historical_qkv_aggregate"] = singulars
    npz_path = output_dir / "intermediates" / ("f1_%s_%s_seed%d.npz" % (backbone_name, dataset_name, seed))
    np.savez_compressed(str(npz_path), **svd_payload)
    return rows


def run_f1(backbone_name, output_dir, device, f1_batch_size):
    log("F1 start for %s" % backbone_name)
    rows = []
    if backbone_name == "cmpnn":
        model_cache = {}
        for dataset_name in DATASET_ORDER:
            dataset = read_dataset(dataset_name)
            key = (len(dataset["target_cols"]), dataset["task"])
            if key not in model_cache:
                model_cache[key] = build_cmpnn_model(key[0], dataset["task"], device)
            model = model_cache[key]
            for seed in F1_SEEDS:
                splits, split_source = load_split(dataset_name, seed)
                rows.extend(run_cmpnn_f1(model, dataset_name, dataset, splits["train"], seed, split_source, output_dir))
                log("F1 cmpnn %s seed%d done" % (dataset_name, seed))
    else:
        max_tasks = max(len(read_dataset(name)["target_cols"]) for name in DATASET_ORDER)
        tokenizer, model, max_len = load_transformer_backbone(backbone_name, max_tasks, device)
        for dataset_name in DATASET_ORDER:
            dataset = read_dataset(dataset_name)
            needed = len(dataset["target_cols"])
            if model.head.out_features != needed:
                model.head = nn.Linear(model.head.in_features, needed).to(device)
            for seed in F1_SEEDS:
                splits, split_source = load_split(dataset_name, seed)
                rows.extend(
                    run_transformer_f1(
                        backbone_name,
                        tokenizer,
                        model,
                        max_len,
                        dataset_name,
                        dataset,
                        splits["train"],
                        seed,
                        split_source,
                        output_dir,
                        f1_batch_size,
                        device,
                    )
                )
                log("F1 %s %s seed%d done" % (backbone_name, dataset_name, seed))
    fieldnames = [
        "backbone",
        "dataset",
        "protocol",
        "seed",
        "target_set",
        "layer_name",
        "layer_group",
        "layer_index",
        "split_source",
        "n_samples_seen",
        "matrix_rows",
        "matrix_cols",
        "erank",
        "participation_ratio",
        "top4_energy",
        "top8_energy",
        "top16_energy",
        "anisotropy",
        "spectral_entropy",
    ]
    partial = output_dir / "intermediates" / ("bam_peft_r1_f1_activation_features__%s.csv" % backbone_name)
    write_csv(partial, rows, fieldnames)
    log("F1 wrote %s rows to %s" % (len(rows), partial))


def build_backbone_for_weights(backbone_name, device):
    if backbone_name == "cmpnn":
        return build_cmpnn_model(1, "regression", device)
    from transformers import AutoModel

    cfg = TRANSFORMER_BACKBONES[backbone_name]
    model = AutoModel.from_pretrained(
        cfg["hf_path"],
        local_files_only=True,
        trust_remote_code=cfg["trust_remote_code"],
        revision=cfg["model_revision"],
    )
    model.to(device)
    return model


def run_f2(backbone_name, output_dir, device):
    log("F2 start for %s" % backbone_name)
    model = build_backbone_for_weights(backbone_name, device)
    rows = []
    svd_payload = {}
    for target_set in ("historical", "expanded"):
        for spec in target_set_specs(backbone_name, target_set):
            module = module_lookup(model, spec["path"])
            weight = tensor_numpy(module.weight)
            base = {
                "backbone": backbone_name,
                "target_set": target_set,
                "layer_name": spec["name"],
                "layer_group": spec["group"],
                "layer_index": spec["index"],
            }
            row, singulars = weight_feature_row(base, weight)
            rows.append(row)
            svd_payload["%s__%s" % (target_set, sanitize(spec["name"]))] = singulars
    npz_path = output_dir / "intermediates" / ("f2_%s.npz" % backbone_name)
    np.savez_compressed(str(npz_path), **svd_payload)
    fieldnames = [
        "backbone",
        "target_set",
        "layer_name",
        "layer_group",
        "layer_index",
        "matrix_rows",
        "matrix_cols",
        "erank",
        "stable_rank",
        "top4_energy",
        "top8_energy",
        "top16_energy",
        "tail_slope",
        "condition_number",
        "sv_entropy",
    ]
    partial = output_dir / "intermediates" / ("bam_peft_r1_f2_weight_features__%s.csv" % backbone_name)
    write_csv(partial, rows, fieldnames)
    log("F2 wrote %s rows to %s" % (len(rows), partial))


def make_cmpnn_batch(dataset, train_indices, batch_size, scaler, device):
    selected = list(train_indices[:batch_size])
    smiles = take_indices(dataset["smiles"], selected)
    targets = apply_scaler(take_target_rows(dataset["targets"], selected), scaler)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    return {"smiles": smiles, "targets": target_tensor}


def make_transformer_batch(tokenizer, max_len, dataset, train_indices, batch_size, scaler, device):
    selected = list(train_indices[:batch_size])
    smiles = take_indices(dataset["smiles"], selected)
    targets = apply_scaler(take_target_rows(dataset["targets"], selected), scaler)
    encoded = encode_batch(tokenizer, smiles, max_len, device)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "targets": torch.tensor(targets, dtype=torch.float32, device=device),
    }


def run_f3(backbone_name, output_dir, device, requested_batch_size):
    log("F3 start for %s" % backbone_name)
    rows = []
    integrity = []
    if backbone_name == "cmpnn":
        model_cache = {}
        for dataset_name in DATASET_ORDER:
            dataset = read_dataset(dataset_name)
            splits, split_source = load_split(dataset_name, F3_SEED)
            scaler = compute_scaler(take_target_rows(dataset["targets"], splits["train"]), dataset["task"])
            batch_size = requested_batch_size or 32
            key = (len(dataset["target_cols"]), dataset["task"])
            if key not in model_cache:
                model_cache[key] = build_cmpnn_model(key[0], dataset["task"], device)
            model = model_cache[key]
            model.train()
            batch = make_cmpnn_batch(dataset, splits["train"], batch_size, scaler, device)
            expanded_specs = target_set_specs("cmpnn", "expanded")
            modules = OrderedDict((spec["name"], module_lookup(model, spec["path"])) for spec in expanded_specs)

            def loss_fn(local_model, local_batch):
                logits = local_model(local_batch["smiles"])
                return masked_loss_tensor(logits, local_batch["targets"], dataset["task"])

            result = extract_step0_gradient(model, modules, batch, loss_fn, head_modules=[model.ffn])
            rows.extend(
                materialize_f3_rows(
                    "cmpnn",
                    dataset_name,
                    split_source,
                    batch_size,
                    dataset["task"],
                    result,
                    expanded_specs,
                )
            )
            integrity.append(integrity_record("cmpnn", dataset_name, batch_size, result))
            save_f3_npz(output_dir, "cmpnn", dataset_name, result)
            log("F3 cmpnn %s done loss=%.6g" % (dataset_name, result["loss"]))
    else:
        tokenizer = None
        model = None
        max_len = None
        for dataset_name in DATASET_ORDER:
            dataset = read_dataset(dataset_name)
            splits, split_source = load_split(dataset_name, F3_SEED)
            scaler = compute_scaler(take_target_rows(dataset["targets"], splits["train"]), dataset["task"])
            batch_size = requested_batch_size or (16 if backbone_name == "molformer_c3" else 32)
            if model is None:
                tokenizer, model, max_len = load_transformer_backbone(backbone_name, len(dataset["target_cols"]), device)
            if model.head.out_features != len(dataset["target_cols"]):
                model.head = nn.Linear(model.head.in_features, len(dataset["target_cols"])).to(device)
            model.train()
            batch = make_transformer_batch(tokenizer, max_len, dataset, splits["train"], batch_size, scaler, device)
            expanded_specs = target_set_specs(backbone_name, "expanded", with_wrapper_prefix=True)
            modules = OrderedDict((spec["name"], module_lookup(model, spec["path"])) for spec in expanded_specs)

            def loss_fn(local_model, local_batch):
                logits = local_model(local_batch["input_ids"], local_batch["attention_mask"])
                return masked_loss_tensor(logits, local_batch["targets"], dataset["task"])

            result = extract_step0_gradient(model, modules, batch, loss_fn, head_modules=[model.head])
            rows.extend(
                materialize_f3_rows(
                    backbone_name,
                    dataset_name,
                    split_source,
                    batch_size,
                    dataset["task"],
                    result,
                    expanded_specs,
                )
            )
            integrity.append(integrity_record(backbone_name, dataset_name, batch_size, result))
            save_f3_npz(output_dir, backbone_name, dataset_name, result)
            log("F3 %s %s done loss=%.6g" % (backbone_name, dataset_name, result["loss"]))
    fieldnames = [
        "backbone",
        "dataset",
        "protocol",
        "seed",
        "target_set",
        "layer_name",
        "layer_group",
        "layer_index",
        "split_source",
        "batch_size",
        "task",
        "loss",
        "sha256_integrity_pass",
        "matrix_rows",
        "matrix_cols",
        "grad_erank",
        "grad_stable_rank",
        "grad_top4_energy",
        "grad_top8_energy",
        "grad_top16_energy",
        "grad_sv_entropy",
        "alignment_A4",
        "alignment_A8",
        "alignment_A16",
        "fisher_target_fraction",
        "fisher_layer_trace",
    ]
    partial = output_dir / "intermediates" / ("bam_peft_r1_f3_gradient_features__%s.csv" % backbone_name)
    write_csv(partial, rows, fieldnames)
    integrity_path = output_dir / "intermediates" / ("bam_peft_r1_f3_integrity__%s.json" % backbone_name)
    with open(str(integrity_path), "w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, sort_keys=True)
    log("F3 wrote %s rows to %s" % (len(rows), partial))


def materialize_f3_rows(backbone_name, dataset_name, split_source, batch_size, task, result, expanded_specs):
    rows = []
    specs_by_name = OrderedDict((spec["name"], spec) for spec in expanded_specs)
    if backbone_name == "cmpnn":
        target_sets = ("historical", "expanded")
    else:
        target_sets = ("historical", "expanded")
    for target_set in target_sets:
        set_specs = target_set_specs(backbone_name, target_set, with_wrapper_prefix=(backbone_name != "cmpnn"))
        denominator_trace = sum(result["traces"][spec["name"]] for spec in set_specs)
        for spec in set_specs:
            canonical = specs_by_name[spec["name"]]
            base = {
                "backbone": backbone_name,
                "dataset": dataset_name,
                "protocol": PROTOCOL,
                "seed": F3_SEED,
                "target_set": target_set,
                "layer_name": spec["name"],
                "layer_group": canonical["group"],
                "layer_index": canonical["index"],
                "split_source": split_source,
                "batch_size": batch_size,
                "task": task,
                "loss": as_float(result["loss"]),
                "sha256_integrity_pass": str(result["pre_hash"] == result["post_hash"]),
            }
            row, _ = gradient_feature_row(
                base,
                tensor_numpy(result["gradients"][spec["name"]]),
                tensor_numpy(result["weights"][spec["name"]]),
                result["traces"][spec["name"]],
                denominator_trace,
            )
            rows.append(row)
    return rows


def save_f3_npz(output_dir, backbone_name, dataset_name, result):
    payload = {}
    for name, grad in result["gradients"].items():
        payload["grad__%s" % sanitize(name)] = tensor_numpy(grad)
    for name, weight in result["weights"].items():
        payload["weight_singulars__%s" % sanitize(name)] = svd_values(tensor_numpy(weight), center=False).astype(np.float32)
    payload["loss"] = np.asarray([result["loss"]], dtype=np.float32)
    npz_path = output_dir / "intermediates" / ("f3_%s_%s_seed%d.npz" % (backbone_name, dataset_name, F3_SEED))
    np.savez_compressed(str(npz_path), **payload)


def integrity_record(backbone_name, dataset_name, batch_size, result):
    return {
        "backbone": backbone_name,
        "dataset": dataset_name,
        "protocol": PROTOCOL,
        "seed": F3_SEED,
        "batch_size": batch_size,
        "loss": result["loss"],
        "pre_hash": result["pre_hash"],
        "post_hash": result["post_hash"],
        "sha256_integrity_pass": result["pre_hash"] == result["post_hash"],
        "head_trace": result["head_trace"],
        "target_trace_sum": sum(result["traces"].values()),
    }


def merge_outputs(output_dir):
    ensure_dir(output_dir)
    intermediate = output_dir / "intermediates"
    f1_rows = collect_partials(intermediate, "bam_peft_r1_f1_activation_features__*.csv")
    f2_rows = collect_partials(intermediate, "bam_peft_r1_f2_weight_features__*.csv")
    f3_rows = collect_partials(intermediate, "bam_peft_r1_f3_gradient_features__*.csv")
    if f1_rows:
        write_csv(output_dir / "bam_peft_r1_f1_activation_features.csv", f1_rows, list(f1_rows[0].keys()))
    if f2_rows:
        write_csv(output_dir / "bam_peft_r1_f2_weight_features.csv", f2_rows, list(f2_rows[0].keys()))
        agg_rows = aggregate_f2_rows(f2_rows)
        write_csv(output_dir / "bam_peft_r1_f2_weight_features_aggregated.csv", agg_rows, list(agg_rows[0].keys()))
    if f3_rows:
        write_csv(output_dir / "bam_peft_r1_f3_gradient_features.csv", f3_rows, list(f3_rows[0].keys()))
    selfcheck = run_selfcheck(output_dir, f1_rows, f2_rows, f3_rows)
    with open(str(output_dir / "bam_peft_r1_stage2_selfcheck.json"), "w", encoding="utf-8") as handle:
        json.dump(selfcheck, handle, indent=2, sort_keys=True)
    log("merge complete: f1=%d f2=%d f3=%d" % (len(f1_rows), len(f2_rows), len(f3_rows)))


def collect_partials(intermediate, pattern):
    rows = []
    for path in sorted(intermediate.glob(pattern)):
        rows.extend(read_csv_rows(path))
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return rows


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def aggregate_f2_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["backbone"], row["target_set"], row["layer_group"])
        groups[key].append(row)
    out = []
    metrics = [
        "erank",
        "stable_rank",
        "top4_energy",
        "top8_energy",
        "top16_energy",
        "tail_slope",
        "condition_number",
        "sv_entropy",
    ]
    for key in sorted(groups.keys()):
        vals = groups[key]
        row = OrderedDict()
        row["backbone"], row["target_set"], row["layer_group"] = key
        row["n_layers"] = len(vals)
        for metric in metrics:
            arr = np.asarray([to_float(v, metric) for v in vals], dtype=np.float64)
            row[metric + "_mean"] = as_float(np.nanmean(arr))
            row[metric + "_median"] = as_float(np.nanmedian(arr))
            row[metric + "_min"] = as_float(np.nanmin(arr))
            row[metric + "_max"] = as_float(np.nanmax(arr))
        out.append(row)
    return out


def run_selfcheck(output_dir, f1_rows, f2_rows, f3_rows):
    checks = OrderedDict()
    checks["correlation_policy"] = CORRELATION_POLICY
    checks["row_counts"] = {
        "f1": len(f1_rows),
        "f2": len(f2_rows),
        "f3": len(f3_rows),
    }
    checks["nan_inf"] = {
        "f1": numeric_health(f1_rows),
        "f2": numeric_health(f2_rows),
        "f3": numeric_health(f3_rows),
    }
    checks["pairing_consistency"] = pairing_consistency(f1_rows, f2_rows, f3_rows)
    checks["sign_unit_consistency"] = sign_unit_consistency(f1_rows, f2_rows, f3_rows)
    checks["separation_smoke_validation"] = separation_smoke_validation(f2_rows, f3_rows)
    checks["f3_sha256_integrity"] = {
        "all_rows_pass": all(row.get("sha256_integrity_pass") == "True" for row in f3_rows) if f3_rows else False,
        "failing_rows": sum(1 for row in f3_rows if row.get("sha256_integrity_pass") != "True"),
    }
    integrity_records = []
    for path in sorted((output_dir / "intermediates").glob("bam_peft_r1_f3_integrity__*.json")):
        with open(str(path), "r", encoding="utf-8") as handle:
            integrity_records.extend(json.load(handle))
    checks["f3_integrity_records"] = {
        "records": len(integrity_records),
        "all_pass": all(item.get("sha256_integrity_pass") for item in integrity_records) if integrity_records else False,
    }
    return checks


def numeric_health(rows):
    bad = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if key in ("backbone", "dataset", "protocol", "target_set", "layer_name", "layer_group", "split_source", "task", "sha256_integrity_pass"):
                continue
            try:
                number = float(value)
            except Exception:
                continue
            if math.isnan(number) or math.isinf(number):
                bad.append({"row": idx, "key": key, "value": value})
    return {"ok": len(bad) == 0, "bad_count": len(bad), "examples": bad[:10]}


def pairing_consistency(f1_rows, f2_rows, f3_rows):
    f1_cells = set((r["backbone"], r["dataset"], r["seed"]) for r in f1_rows)
    expected_f1_cells = set()
    for backbone in ("cmpnn", "chemberta2", "molformer_c3"):
        for dataset in DATASET_ORDER:
            for seed in F1_SEEDS:
                expected_f1_cells.add((backbone, dataset, str(seed)))
    f3_cells = set((r["backbone"], r["dataset"], r["seed"]) for r in f3_rows)
    expected_f3_cells = set((backbone, dataset, str(F3_SEED)) for backbone in ("cmpnn", "chemberta2", "molformer_c3") for dataset in DATASET_ORDER)
    f2_pairs = set((r["backbone"], r["target_set"]) for r in f2_rows)
    expected_f2_pairs = set((backbone, target_set) for backbone in ("cmpnn", "chemberta2", "molformer_c3") for target_set in ("historical", "expanded"))
    return {
        "f1_expected_cells": len(expected_f1_cells),
        "f1_observed_cells": len(f1_cells),
        "f1_missing": sorted(list(expected_f1_cells - f1_cells))[:20],
        "f3_expected_cells": len(expected_f3_cells),
        "f3_observed_cells": len(f3_cells),
        "f3_missing": sorted(list(expected_f3_cells - f3_cells))[:20],
        "f2_expected_pairs": len(expected_f2_pairs),
        "f2_observed_pairs": len(f2_pairs),
        "f2_missing": sorted(list(expected_f2_pairs - f2_pairs))[:20],
    }


def sign_unit_consistency(f1_rows, f2_rows, f3_rows):
    issues = []
    for family, rows, keys in (
        ("f1", f1_rows, ("top4_energy", "top8_energy", "top16_energy")),
        ("f2", f2_rows, ("top4_energy", "top8_energy", "top16_energy")),
        ("f3", f3_rows, ("grad_top4_energy", "grad_top8_energy", "grad_top16_energy")),
    ):
        for idx, row in enumerate(rows):
            values = [to_float(row, key) for key in keys]
            if any(v < -1e-9 or v > 1.0000001 for v in values):
                issues.append({"family": family, "row": idx, "issue": "energy_fraction_out_of_range", "values": values})
            if not (values[0] <= values[1] + 1e-9 and values[1] <= values[2] + 1e-9):
                issues.append({"family": family, "row": idx, "issue": "topk_not_monotonic", "values": values})
    for row in f3_rows:
        fisher = to_float(row, "fisher_target_fraction")
        aligns = [to_float(row, key) for key in ("alignment_A4", "alignment_A8", "alignment_A16")]
        if fisher < -1e-9 or fisher > 1.0000001:
            issues.append({"family": "f3", "row": row.get("layer_name"), "issue": "fisher_fraction_out_of_range", "value": fisher})
        if any(v < -1e-9 or v > 1.0000001 for v in aligns):
            issues.append({"family": "f3", "row": row.get("layer_name"), "issue": "alignment_out_of_range", "value": aligns})
    return {"ok": len(issues) == 0, "issue_count": len(issues), "examples": issues[:20]}


def separation_smoke_validation(f2_rows, f3_rows):
    result = OrderedDict()
    for backbone in ("cmpnn", "chemberta2", "molformer_c3"):
        f2_hist = [r for r in f2_rows if r["backbone"] == backbone and r["target_set"] == "historical"]
        f3_hist = [r for r in f3_rows if r["backbone"] == backbone and r["target_set"] == "historical"]
        f2_erank = np.nanmean([to_float(r, "erank") for r in f2_hist]) if f2_hist else float("nan")
        f3_erank = np.nanmean([to_float(r, "grad_erank") for r in f3_hist]) if f3_hist else float("nan")
        f2_top8 = np.nanmean([to_float(r, "top8_energy") for r in f2_hist]) if f2_hist else float("nan")
        f3_top8 = np.nanmean([to_float(r, "grad_top8_energy") for r in f3_hist]) if f3_hist else float("nan")
        result[backbone] = {
            "historical_f2_mean_erank": as_float(f2_erank),
            "historical_f3_mean_grad_erank": as_float(f3_erank),
            "historical_f2_mean_top8_energy": as_float(f2_top8),
            "historical_f3_mean_grad_top8_energy": as_float(f3_top8),
        }
    chem_favoring = False
    try:
        chem_favoring = (
            float(result["chemberta2"]["historical_f2_mean_erank"]) < float(result["molformer_c3"]["historical_f2_mean_erank"])
            and float(result["chemberta2"]["historical_f3_mean_grad_erank"]) < float(result["molformer_c3"]["historical_f3_mean_grad_erank"])
        )
    except Exception:
        chem_favoring = False
    result["chemberta2_peft_favoring_low_rank_smoke"] = {
        "pass": bool(chem_favoring),
        "criterion": "ChemBERTa-2 historical F2/F3 mean effective ranks lower than MoLFormer-c3 historical means",
    }
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "intermediates")
    if args.merge_only:
        merge_outputs(output_dir)
        return
    if not args.backbone:
        raise SystemExit("--backbone is required unless --merge-only is set")
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    log("Stage2 extraction start backbone=%s device=%s output=%s" % (args.backbone, device, output_dir))
    if not args.skip_f1:
        run_f1(args.backbone, output_dir, device, args.f1_batch_size)
    if not args.skip_f2:
        run_f2(args.backbone, output_dir, device)
    if not args.skip_f3:
        run_f3(args.backbone, output_dir, device, args.f3_batch_size)
    log("Stage2 extraction done backbone=%s" % args.backbone)


if __name__ == "__main__":
    main()
