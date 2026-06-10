from argparse import Namespace
import sys
import builtins


def repair_featurization_module():
    try:
        import chemprop.features.featurization as feat_module
        if 'len' in feat_module.__dict__:
            current_val = feat_module.__dict__['len']
            if current_val is not builtins.len:
                print("\n" + "!" * 60)
                print(f"🚑 [Environment Alert] Fixing 'len' in featurization.py")
                feat_module.len = builtins.len
                if feat_module.len is builtins.len:
                    print(f"✅ Fix successful! Environment cleaned.")
                else:
                    print(f"❌ Fix failed, please check file permissions")
                print("!" * 60 + "\n")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ [Environment Repair] Warning: {e}")


repair_featurization_module()

from chemprop.features import mol2graph
from .cmpn import CMPN
from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights
import torch
import torch.nn as nn
from typing import Optional, List, Union
import torch.nn.functional as F
import math
import numpy as np


class CMPNAdapter(nn.Module):

    def __init__(self, cmpn_encoder):
        super(CMPNAdapter, self).__init__()
        self.cmpn = cmpn_encoder
        self.encoder = cmpn_encoder.encoder

    def forward(self, batch, features_batch=None):
        result = self.cmpn.forward(batch, features_batch)

        if isinstance(result, tuple):
            if len(result) == 3:
                return result  # (mol_vecs, atom_hiddens, a_scope)
            elif len(result) == 2:
                return (*result, None)  # (mol_vecs, atom_hiddens, None)
            else:
                return (result[0], None, None)  # (mol_vecs, None, None)
        else:
            return (result, None, None)  # (mol_vecs, None, None)


def validate_and_ensure_args(args: Namespace):
    missing_keys = []
    critical_params = [
        'hidden_size', 'depth', 'num_tasks', 'dataset_type',
        'dropout', 'activation', 'atom_messages', 'undirected'
    ]
    for param in critical_params:
        if not hasattr(args, param):
            if param == 'dropout':
                setattr(args, param, 0.0)
            elif param == 'activation':
                setattr(args, param, 'relu')
            elif param == 'atom_messages':
                setattr(args, param, False)
            elif param == 'undirected':
                setattr(args, param, False)
            else:
                missing_keys.append(param)

    if missing_keys:
        raise ValueError(f"🛑 [Config Error] args missing critical params: {missing_keys}")

    if not hasattr(args, 'features_dim'):
        if hasattr(args, 'features_size'):
            args.features_dim = args.features_size
        else:
            if getattr(args, 'use_input_features', False):
                raise ValueError("🛑 [Logic Error] use_input_features enabled but features_dim/features_size not found")
            else:
                args.features_dim = 0

    if not hasattr(args, 'features_only'): setattr(args, 'features_only', False)
    if not hasattr(args, 'use_input_features'): setattr(args, 'use_input_features', False)

    return True


class MoleculeModel(nn.Module):
    def __init__(self, classification: bool, multiclass: bool, pretrain: bool, args: Namespace = None):
        super(MoleculeModel, self).__init__()
        if args is None:
            raise ValueError("MoleculeModel requires args parameter")

        self.args = args
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.pretrain = pretrain

        self._last_atom_hiddens = None
        self._last_a_scope = None

    def create_encoder(self, args: Namespace, encoder_name):
        if encoder_name == 'CMPNN':
            cmpn = CMPN(
                args=args,
                atom_fdim=getattr(args, 'atom_fdim', 133),
                bond_fdim=getattr(args, 'bond_fdim', 147)
            )
            self.encoder = CMPNAdapter(cmpn)
        elif encoder_name == 'MPNN':
            self.encoder = MPN(args)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def create_ffn(self, args: Namespace):
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes

        if getattr(args, 'features_only', False):
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if getattr(args, 'use_input_features', False):
                first_linear_dim += args.features_dim
            if getattr(args, 'use_fg_concat', False):
                first_linear_dim += getattr(args, 'fg_num_features', 57)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        ffn_hidden_size = getattr(args, 'ffn_hidden_size', args.hidden_size)
        ffn_num_layers = getattr(args, 'ffn_num_layers', 2)
        output_size = args.output_size

        if ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, output_size)])
        self.ffn = nn.Sequential(*ffn)

    def forward(self, batch, features_batch=None, *, task_id: Optional[int] = 0):
        encoder_outputs = self.encoder(batch, features_batch)

        if isinstance(encoder_outputs, tuple):
            if len(encoder_outputs) == 3:
                mol_vecs, atom_hiddens, a_scope = encoder_outputs
            elif len(encoder_outputs) == 2:
                mol_vecs, atom_hiddens = encoder_outputs
                a_scope = None
            else:
                mol_vecs = encoder_outputs[0]
                atom_hiddens = None
                a_scope = None
        else:
            mol_vecs = encoder_outputs
            atom_hiddens = None
            a_scope = None

        self._last_atom_hiddens = atom_hiddens
        self._last_a_scope = a_scope
        if getattr(self.args, 'use_fg_concat', False):
            extractor = getattr(self.args, 'fg_feature_extractor', None)
            if extractor is None:
                raise ValueError('A3 FG-concat requires args.fg_feature_extractor')
            smiles_batch = batch.smiles_batch if hasattr(batch, 'smiles_batch') else batch
            fg_rows = [extractor.compute_from_smiles(smiles) for smiles in smiles_batch]
            fg_tensor = torch.FloatTensor(np.stack(fg_rows))
            if mol_vecs.is_cuda:
                fg_tensor = fg_tensor.cuda()
            mol_vecs = torch.cat([mol_vecs, fg_tensor], dim=1)

        output = self.ffn(mol_vecs)
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)
        return output


def build_model(args: Namespace, encoder_name: str) -> nn.Module:
    validate_and_ensure_args(args)
    args.output_size = args.num_tasks * (args.multiclass_num_classes if args.dataset_type == 'multiclass' else 1)
    model = MoleculeModel(args.dataset_type == 'classification', args.dataset_type == 'multiclass', False, args)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)
    initialize_weights(model)
    encoder = getattr(getattr(model, 'encoder', None), 'encoder', None)
    if hasattr(encoder, 'reset_fg_parameters'):
        encoder.reset_fg_parameters()
    return model


def build_pretrain_model(args: Namespace, encoder_name: str, num_tasks: int = None) -> nn.Module:
    validate_and_ensure_args(args)
    args.ffn_hidden_size = args.hidden_size // 2
    args.output_size = args.hidden_size
    model = MoleculeModel(args.dataset_type == 'classification', args.dataset_type == 'multiclass', True, args)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)
    initialize_weights(model)
    return model


__all__ = [
    'MoleculeModel', 'build_model', 'build_pretrain_model', 'CMPNAdapter'
]
