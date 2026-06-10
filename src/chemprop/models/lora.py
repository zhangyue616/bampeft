import math
from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_LORA_TARGET_LAYERS = ['W_i_atom', 'W_i_bond', 'W_h_0', 'W_h_1', 'W_o', 'lr']


class LoRADoRALinear(nn.Module):
    """LoRA wrapper for a Linear layer, with DoRA reserved for a later stage."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, use_dora: bool = False):
        super(LoRADoRALinear, self).__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRADoRALinear expects nn.Linear, got {type(base_linear).__name__}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_dora = use_dora

        if use_dora:
            with torch.no_grad():
                self.lora_m = nn.Parameter(torch.norm(base_linear.weight, dim=1))

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_linear.out_features, rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(
        self,
        x: torch.Tensor,
        magnitude_ratio_per_row: torch.Tensor = None,
        lora_m_override: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        FG-DoRA forward.

        Args:
            x: input tensor [batch_rows, in_features].
            magnitude_ratio_per_row: optional ratio (NOT absolute magnitude).
                None uses vanilla DoRA magnitude m = self.lora_m.
                ones uses m = self.lora_m * 1 = self.lora_m, identical to None.
                ratio uses m = self.lora_m * ratio for FG-DoRA.
            lora_m_override: optional absolute DoRA magnitude vector [out_features].
                Used by A6 after folding the train-set mean ratio into lora_m.

        Mathematical equivalence:
            gate output ratio = exp(tau * h(FG)); forward uses
            m = self.lora_m * ratio = m_0 * exp(tau * h(FG)).
        """
        if magnitude_ratio_per_row is not None and lora_m_override is not None:
            raise ValueError('magnitude_ratio_per_row and lora_m_override are mutually exclusive')
        if magnitude_ratio_per_row is not None:
            if magnitude_ratio_per_row.dim() != 2:
                raise ValueError('magnitude_ratio_per_row must be [batch_rows, out_features]')
            if magnitude_ratio_per_row.shape != (x.shape[0], self.base.out_features):
                raise ValueError(
                    'magnitude_ratio_per_row shape mismatch: '
                    f'expected {(x.shape[0], self.base.out_features)}, got {tuple(magnitude_ratio_per_row.shape)}'
                )
        if lora_m_override is not None:
            if not self.use_dora:
                raise ValueError('lora_m_override requires use_dora=True')
            if lora_m_override.dim() != 1:
                raise ValueError('lora_m_override must be [out_features]')
            if lora_m_override.shape != (self.base.out_features,):
                raise ValueError(
                    'lora_m_override shape mismatch: '
                    f'expected {(self.base.out_features,)}, got {tuple(lora_m_override.shape)}'
                )

        if self.use_dora:
            delta = self.lora_B @ self.lora_A
            V = self.base.weight + self.scaling * delta
            if magnitude_ratio_per_row is None:
                V_norm = torch.norm(V, dim=1, keepdim=True)
                assert V_norm.min().item() > 1e-8, f"DoRA weight norm too small: {V_norm.min().item():.2e}"
                lora_m = self.lora_m if lora_m_override is None else lora_m_override
                weight = lora_m.unsqueeze(1) * V / V_norm
                return F.linear(x, weight, self.base.bias)

            V_row_norm = torch.norm(V, dim=1)
            assert V_row_norm.min().item() > 1e-8, f"DoRA weight norm too small: {V_row_norm.min().item():.2e}"
            Vx = x @ V.t()
            m = self.lora_m.unsqueeze(0) * magnitude_ratio_per_row
            output = (m / V_row_norm.unsqueeze(0)) * Vx
            if self.base.bias is not None:
                output = output + self.base.bias.unsqueeze(0)
            return output

        base_out = self.base(x)
        lora_out = (x.matmul(self.lora_A.t())).matmul(self.lora_B.t())
        if magnitude_ratio_per_row is not None:
            lora_out = lora_out * magnitude_ratio_per_row
        return base_out + lora_out * self.scaling


def _get_cmpn_encoder(model: nn.Module) -> nn.Module:
    try:
        return model.encoder.encoder
    except AttributeError as exc:
        raise AttributeError("Expected model.encoder.encoder CMPNEncoder path for LoRA wrapping") from exc


def wrap_lora_layers(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_layer_names: Iterable[str] = None,
    use_dora: bool = False,
) -> List[str]:
    target_layer_names = list(target_layer_names or DEFAULT_LORA_TARGET_LAYERS)
    encoder = _get_cmpn_encoder(model)
    wrapped = []

    for layer_name in target_layer_names:
        if not hasattr(encoder, layer_name):
            raise AttributeError(f"CMPNEncoder has no layer named {layer_name}")

        layer = getattr(encoder, layer_name)
        if isinstance(layer, LoRADoRALinear):
            raise ValueError(f"Layer {layer_name} is already wrapped with LoRADoRALinear")
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Layer {layer_name} must be nn.Linear, got {type(layer).__name__}")

        setattr(encoder, layer_name, LoRADoRALinear(layer, rank=rank, alpha=alpha, use_dora=use_dora))
        wrapped.append(layer_name)

    return wrapped


def freeze_base_params(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if 'fg_gates' in name or 'fg_taus' in name:
            param.requires_grad = True
        elif 'lora_' in name:
            param.requires_grad = True
        elif name.startswith('ffn.') or '.ffn.' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_encoder_train_head_only(model: nn.Module) -> None:
    """Freeze encoder parameters and train only FFN/head parameters."""
    for name, param in model.named_parameters():
        if name.startswith('ffn.') or '.ffn.' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
