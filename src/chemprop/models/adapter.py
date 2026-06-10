from typing import Iterable, List

import torch
from torch import nn


DEFAULT_ADAPTER_TARGET_LAYERS = ['W_i_atom', 'W_i_bond', 'W_h_0', 'W_h_1', 'W_o', 'lr']


class AdapterGNNLiteBlock(nn.Module):
    """Output-side AdapterGNN-lite residual bottleneck block."""

    def __init__(self, hidden_dim: int = 300, bottleneck_dim: int = 8, scale_init: float = 0.01):
        super(AdapterGNNLiteBlock, self).__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"AdapterGNNLiteBlock expected last dim {self.hidden_dim}, got {y.shape[-1]}"
            )

        adapter_out = self.up_proj(self.activation(self.down_proj(y)))
        original_shape = adapter_out.shape
        flat = adapter_out.reshape(-1, self.hidden_dim)
        flat = self.batch_norm(flat)
        adapter_out = flat.reshape(original_shape)
        return y + self.scale * adapter_out


class AdapterGNNLinear(nn.Module):
    """Wrap a frozen Linear layer with a trainable AdapterGNN-lite block."""

    def __init__(self, base_linear: nn.Linear, bottleneck_dim: int = 8, scale_init: float = 0.01):
        super(AdapterGNNLinear, self).__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"AdapterGNNLinear expects nn.Linear, got {type(base_linear).__name__}")

        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        self.adapter = AdapterGNNLiteBlock(
            hidden_dim=base_linear.out_features,
            bottleneck_dim=bottleneck_dim,
            scale_init=scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.base(x))


def _get_cmpn_encoder(model: nn.Module) -> nn.Module:
    try:
        return model.encoder.encoder
    except AttributeError as exc:
        raise AttributeError("Expected model.encoder.encoder CMPNEncoder path for Adapter wrapping") from exc


def wrap_adapter_layers(
    model: nn.Module,
    target_layer_names: Iterable[str] = None,
    bottleneck_dim: int = 8,
    scale_init: float = 0.01,
) -> List[str]:
    target_layer_names = list(target_layer_names or DEFAULT_ADAPTER_TARGET_LAYERS)
    encoder = _get_cmpn_encoder(model)
    wrapped = []

    for layer_name in target_layer_names:
        if not hasattr(encoder, layer_name):
            raise AttributeError(f"CMPNEncoder has no layer named {layer_name}")

        layer = getattr(encoder, layer_name)
        if isinstance(layer, AdapterGNNLinear):
            raise ValueError(f"Layer {layer_name} is already wrapped with AdapterGNNLinear")
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Layer {layer_name} must be nn.Linear, got {type(layer).__name__}")

        setattr(
            encoder,
            layer_name,
            AdapterGNNLinear(layer, bottleneck_dim=bottleneck_dim, scale_init=scale_init),
        )
        wrapped.append(layer_name)

    return wrapped


def freeze_adapter_train_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if '.adapter.' in name:
            param.requires_grad = True
        elif name.startswith('ffn.') or '.ffn.' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
