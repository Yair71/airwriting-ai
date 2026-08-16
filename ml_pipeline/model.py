"""1D-ResNet + BiGRU + temporal attention unistroke classifier.

Data contract
-------------
Input : float32 (B, 63, 6) = [dx, dy, sin(theta), cos(theta), dtheta, norm_velocity]
Output: float32 (B, n_classes) logits
Params: ~0.4M (no size cap — accuracy first)
CPU   : target < 5 ms / sample unbatched
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

SEQ_LEN = 63
N_FEATURES = 6


@dataclass(frozen=True)
class ModelConfig:
    n_features: int = N_FEATURES
    n_classes: int = 100
    hidden_dim: int = 96
    gru_layers: int = 2
    stem_channels: int = 48
    dropout: float = 0.15
    seq_len: int = SEQ_LEN
    # Kept so older checkpoints / MorphNet configs still deserialize.
    conv_channels: tuple[int, ...] = (48, 96, 96)
    kernel_sizes: tuple[int, ...] = (5, 3, 3)
    gru_hidden: int | None = None
    attn_dim: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> ModelConfig:
        conv = payload.get("conv_channels", (48, 96, 96))
        kernels = payload.get("kernel_sizes", (5, 3, 3))
        hidden = payload.get("gru_hidden")
        hidden_dim = int(payload.get("hidden_dim", hidden if hidden is not None else 96))
        return cls(
            n_features=int(payload.get("n_features", N_FEATURES)),
            n_classes=int(payload.get("n_classes", 100)),
            hidden_dim=hidden_dim,
            gru_layers=int(payload.get("gru_layers", payload.get("num_layers", 2))),
            stem_channels=int(payload.get("stem_channels", 48)),
            dropout=float(payload.get("dropout", 0.15)),
            seq_len=int(payload.get("seq_len", SEQ_LEN)),
            conv_channels=tuple(int(c) for c in conv),
            kernel_sizes=tuple(int(k) for k in kernels),
            gru_hidden=hidden_dim,
            attn_dim=payload.get("attn_dim", None),
        )


class TemporalAttention(nn.Module):
    """Additive (Bahdanau) attention over the time axis."""

    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1 or attn_dim < 1:
            raise ValueError("attention dimensions must be positive")
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(self.proj(seq))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return context, weights


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so padding keeps length 63")
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.skip_bn = nn.Identity() if in_ch == out_ch else nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(self.drop(y)))
        return self.act(y + self.skip_bn(self.skip(x)))


class UnistrokeNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if cfg.n_features < 1:
            raise ValueError("n_features must be positive")
        if cfg.hidden_dim < 1 or cfg.gru_layers < 1:
            raise ValueError("hidden_dim and gru_layers must be positive")
        if cfg.stem_channels < 1:
            raise ValueError("stem_channels must be positive")

        self.config = cfg
        h = cfg.hidden_dim
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.n_features, cfg.stem_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.stem_channels),
            nn.SiLU(inplace=True),
        )
        self.res = nn.Sequential(
            ResBlock1d(cfg.stem_channels, cfg.stem_channels, kernel=3, dropout=cfg.dropout),
            ResBlock1d(cfg.stem_channels, h, kernel=3, dropout=cfg.dropout),
            ResBlock1d(h, h, kernel=3, dropout=cfg.dropout),
        )
        self.conv_drop = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(
            input_size=h,
            hidden_size=h,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        gru_out = h * 2
        attn_dim = int(cfg.attn_dim) if cfg.attn_dim else gru_out
        self.attn = TemporalAttention(gru_out, attn_dim)
        self.head_drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(gru_out, cfg.n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = self.config.hidden_dim
                param.data[hidden : 2 * hidden].fill_(1.0)

    @property
    def morphnet_batchnorms(self) -> list[nn.BatchNorm1d]:
        bns: list[nn.BatchNorm1d] = []
        for module in (self.stem, self.res):
            for child in module.modules():
                if isinstance(child, nn.BatchNorm1d):
                    bns.append(child)
        return bns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x.transpose(1, 2))
        hidden = self.res(hidden)
        hidden = self.conv_drop(hidden)
        gru_out, _ = self.gru(hidden.transpose(1, 2))
        context, _weights = self.attn(gru_out)
        return self.classifier(self.head_drop(context))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_flops(model: UnistrokeNet, seq_len: int | None = None) -> int:
    """Rough multiply-add count for one sample (not a profiler)."""
    cfg = model.config
    length = int(seq_len or cfg.seq_len)
    flops = 0
    flops += 2 * cfg.n_features * cfg.stem_channels * 5 * length
    h = cfg.hidden_dim
    # three residual blocks: (stem,stem), (stem,h), (h,h); two k=3 convs each
    flops += 2 * (2 * cfg.stem_channels * cfg.stem_channels * 3 * length)
    flops += 2 * cfg.stem_channels * h * 3 * length + 2 * h * h * 3 * length
    flops += 2 * (2 * h * h * 3 * length)
    # BiGRU: 3 gates, two directions, gru_layers stacked (layer 0 input h, later 2h)
    for layer in range(cfg.gru_layers):
        in_size = h if layer == 0 else 2 * h
        flops += 2 * length * 2 * 3 * (in_size * h + h * h)
    gru_out = h * 2
    attn_dim = int(cfg.attn_dim) if cfg.attn_dim else gru_out
    flops += 2 * length * gru_out * attn_dim
    flops += 2 * length * attn_dim
    flops += 2 * gru_out * cfg.n_classes
    return int(flops)


def fp32_bytes(n_params: int) -> int:
    return int(n_params * 4)
