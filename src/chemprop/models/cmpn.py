from argparse import Namespace
from typing import List, Union, Tuple

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from torch_scatter import scatter_add
from chemprop.models.fg_gate import FGMagnitudeGate


class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)

        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)
        self.use_fg_dora = bool(getattr(args, 'use_fg_dora', False))
        self.fg_gated_layers = list(getattr(args, 'fg_gated_layers', []) or [])
        self.fg_gates = nn.ModuleDict()
        self.fg_taus = nn.ParameterDict()
        self._last_gate_stats = {}
        self.fg_tau_init = float(getattr(args, 'fg_tau_init', 0.1))
        if self.use_fg_dora:
            n_fg = int(getattr(args, 'fg_num_features', 57))
            gate_hidden_dim = int(getattr(args, 'fg_gate_hidden_dim', 64))
            a6_lora_m_static = getattr(args, 'A6_lora_m_static', None) or {}
            if not isinstance(a6_lora_m_static, dict):
                raise ValueError('A6_lora_m_static must be a dict keyed by layer name')
            self.register_buffer('a6_static_fold_ready', torch.tensor(bool(a6_lora_m_static), dtype=torch.bool))
            unsupported = sorted(set(self.fg_gated_layers) - {'W_i_atom', 'W_o'})
            if unsupported:
                raise ValueError(f'FG-DoRA D1 routing only supports W_i_atom/W_o, got {unsupported}')
            if a6_lora_m_static:
                missing_static = sorted(set(self.fg_gated_layers) - set(a6_lora_m_static.keys()))
                if missing_static:
                    raise ValueError(f'A6_lora_m_static missing gated layers: {missing_static}')
            for layer_name in self.fg_gated_layers:
                layer = getattr(self, layer_name)
                self.fg_gates[layer_name] = FGMagnitudeGate(
                    n_fg=n_fg,
                    out_features=layer.out_features,
                    hidden_dim=gate_hidden_dim,
                )
                self.fg_taus[layer_name] = nn.Parameter(torch.tensor([self.fg_tau_init], dtype=torch.float32))
                static_values = a6_lora_m_static.get(layer_name)
                if static_values is None:
                    static_tensor = torch.zeros(layer.out_features, dtype=torch.float32)
                else:
                    static_tensor = torch.tensor(static_values, dtype=torch.float32)
                    if static_tensor.shape != (layer.out_features,):
                        raise ValueError(
                            f'A6_lora_m_static[{layer_name}] shape mismatch: '
                            f'expected {(layer.out_features,)}, got {tuple(static_tensor.shape)}'
                        )
                self.register_buffer(f'a6_lora_m_static_{layer_name}', static_tensor)

    def reset_fg_parameters(self) -> None:
        for gate in self.fg_gates.values():
            gate.reset_fg_parameters()
        for tau in self.fg_taus.values():
            with torch.no_grad():
                tau.fill_(self.fg_tau_init)

    def has_a6_static_fold(self) -> bool:
        return (
            self.use_fg_dora
            and hasattr(self, 'a6_static_fold_ready')
            and bool(self.a6_static_fold_ready.detach().cpu().item())
        )

    def set_a6_lora_m_static(self, layer_name: str, lora_m: torch.Tensor) -> None:
        buffer_name = f'a6_lora_m_static_{layer_name}'
        if not hasattr(self, buffer_name):
            raise ValueError(f'A6 static fold buffer missing for layer {layer_name}')
        buffer = getattr(self, buffer_name)
        value = lora_m.detach().to(device=buffer.device, dtype=buffer.dtype)
        if value.shape != buffer.shape:
            raise ValueError(
                f'A6 static lora_m shape mismatch for {layer_name}: '
                f'expected {tuple(buffer.shape)}, got {tuple(value.shape)}'
            )
        with torch.no_grad():
            buffer.copy_(value)
            self.a6_static_fold_ready.fill_(True)

    def get_a6_lora_m_static(self, layer_name: str) -> torch.Tensor:
        if not self.has_a6_static_fold():
            return None
        buffer_name = f'a6_lora_m_static_{layer_name}'
        if not hasattr(self, buffer_name):
            return None
        return getattr(self, buffer_name)

    def _use_a6_static_fold(self) -> bool:
        return (
            getattr(self.args, 'ablation_method', None) == 'A6'
            and not self.training
            and self.has_a6_static_fold()
        )

    def _ratio_per_row(self,
                       layer_name: str,
                       fg_features: torch.Tensor,
                       a_scope: List[Tuple[int, int]],
                       n_rows: int,
                       device: torch.device) -> torch.Tensor:
        if not self.use_fg_dora or layer_name not in self.fg_gates:
            return None
        if fg_features is None:
            raise ValueError(f'FG-DoRA layer {layer_name} needs mol_graph.get_fg_features()')

        fg_features = fg_features.to(device)
        gate_out = self.fg_gates[layer_name](fg_features)
        tau = F.softplus(self.fg_taus[layer_name]) / 10.0
        ratio_per_mol = torch.exp(tau * gate_out)
        ratio_per_row = ratio_per_mol.new_ones(n_rows, ratio_per_mol.shape[1])
        for mol_idx, (start, n_atoms) in enumerate(a_scope):
            assert start >= 1, 'a_scope start should be >= 1 (index 0 is dummy)'
            ratio_per_row[start:start + n_atoms] = ratio_per_mol[mol_idx]

        self._last_gate_stats[layer_name] = {
            'h_norm': float(gate_out.detach().norm().cpu().item()),
            'h_std': float(gate_out.detach().std(unbiased=False).cpu().item()),
            'ratio_mean': float(ratio_per_mol.detach().mean().cpu().item()),
            'ratio_std': float(ratio_per_mol.detach().std(unbiased=False).cpu().item()),
        }
        return ratio_per_row

    def forward(self, mol_graph, features_batch=None) -> Tuple[torch.FloatTensor, torch.FloatTensor, List]:
        """
        Forward pass with three return values

        Returns:
            mol_vecs: [batch_size, hidden_size] - molecule vectors
            atom_hiddens: [num_atoms, hidden_size] - atom features
            a_scope: List[(start, size)] - atom range per molecule
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                f_atoms.cuda(), f_bonds.cuda(),
                a2b.cuda(), b2a.cuda(), b2revb.cuda())
        use_a6_static_fold = self._use_a6_static_fold()
        fg_features = mol_graph.get_fg_features() if self.use_fg_dora and not use_a6_static_fold else None

        atom_lora_m = self.get_a6_lora_m_static('W_i_atom') if use_a6_static_fold else None
        if atom_lora_m is not None:
            input_atom = self.W_i_atom(f_atoms, lora_m_override=atom_lora_m)
        else:
            atom_ratio = self._ratio_per_row('W_i_atom', fg_features, a_scope, f_atoms.shape[0], f_atoms.device)
            if atom_ratio is None:
                input_atom = self.W_i_atom(f_atoms)
            else:
                input_atom = self.W_i_atom(f_atoms, magnitude_ratio_per_row=atom_ratio)

        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            rev_message = message_bond[b2revb]
            message_bond = message_atom[b2a] - rev_message

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        w_o_lora_m = self.get_a6_lora_m_static('W_o') if use_a6_static_fold else None
        if w_o_lora_m is not None:
            atom_hiddens = self.act_func(self.W_o(agg_message, lora_m_override=w_o_lora_m))
        else:
            w_o_ratio = self._ratio_per_row('W_o', fg_features, a_scope, agg_message.shape[0], agg_message.device)
            if w_o_ratio is None:
                atom_hiddens = self.act_func(self.W_o(agg_message))
            else:
                atom_hiddens = self.act_func(self.W_o(agg_message, magnitude_ratio_per_row=w_o_ratio))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs, atom_hiddens, a_scope


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])

        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                         (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, batch,
                features_batch: List[np.ndarray] = None) -> Union[
        torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, List]]:
        """
        Forward pass returning encoder outputs

        Returns:
            (mol_vecs, atom_hiddens, a_scope) - tuple with three elements
        """
        if not self.graph_input:
            batch = mol2graph(batch, self.args)

        mol_vecs, atom_hiddens, a_scope = self.encoder.forward(batch, features_batch)

        return mol_vecs, atom_hiddens, a_scope
