"""PyTorch backend (NVIDIA GPUs through CUDA; CPU/MPS elsewhere)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from ..weights import Weights
from .common import BASE_LEARNING_RATE, GRAD_CLIP_NORM, LABEL_SMOOTHING, WEIGHT_DECAY


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + x)


class _Net(nn.Module):
    """Same architecture as ``train/model.py``; module names mirror it."""

    def __init__(self, channels: int = 32, n_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(n_blocks))
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, 64)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        # Flatten as (H, W, C): the canonical layout of `policy_fc`.
        p = self.policy_fc(p.permute(0, 2, 3, 1).flatten(1))
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return p, v.squeeze(-1)


def _canonical_name(name: str) -> str:
    """torch state-dict key → canonical (MLX) key."""
    return "stem.layers." + name[len("stem.") :] if name.startswith("stem.") else name


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchModel:
    def __init__(self) -> None:
        self.device = _device()
        self._cuda = self.device.type == "cuda"
        # Leave cudnn.benchmark off: the self-play batch size changes from
        # round to round while games start, and every new shape would trigger
        # a fresh (slow) algorithm search.
        self.net = _Net().to(self.device).to(memory_format=torch.channels_last)
        self._optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    def _state(self) -> dict[str, torch.Tensor]:
        # `num_batches_tracked` is bookkeeping, not a weight.
        return {
            k: v
            for k, v in self.net.state_dict().items()
            if not k.endswith("num_batches_tracked")
        }

    def weights(self) -> Weights:
        weights: Weights = {}
        for name, tensor in self._state().items():
            if tensor.ndim == 4:  # conv: (out, in, kH, kW) → (out, kH, kW, in)
                tensor = tensor.permute(0, 2, 3, 1)
            array = tensor.detach().float().cpu().contiguous().numpy()
            weights[_canonical_name(name)] = array
        return weights

    def load_weights(self, weights: Weights) -> int:
        state = self._state()
        update: dict[str, torch.Tensor] = {}
        for name, current in state.items():
            arr = weights.get(_canonical_name(name))
            if arr is None:
                continue
            tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            if tensor.ndim == 4:  # (out, kH, kW, in) → (out, in, kH, kW)
                tensor = tensor.permute(0, 3, 1, 2)
            if tuple(tensor.shape) != tuple(current.shape):
                continue
            update[name] = tensor
        self.net.load_state_dict(update, strict=False)
        return len(update)

    def _input(self, planes: np.ndarray) -> torch.Tensor:
        # `torch.tensor` copies, so read-only buffers (the self-play batch) are fine.
        x = torch.tensor(planes, dtype=torch.float32, device=self.device)
        return x.contiguous(memory_format=torch.channels_last)

    def infer(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        with (
            torch.inference_mode(),
            # Half precision roughly doubles inference throughput on CUDA.
            torch.autocast("cuda", dtype=torch.float16, enabled=self._cuda),
        ):
            logits, values = self.net(self._input(planes))
        policies = F.softmax(logits.float(), dim=-1)
        return policies.cpu().numpy(), values.float().cpu().numpy()

    def train_step(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        policy_weights: np.ndarray,
        value_weights: np.ndarray,
    ) -> float:
        self.net.train()

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
                self.device
            )

        x = self._input(planes)
        phase = x[:, 3, 0, 0]
        policy_targets = (1.0 - LABEL_SMOOTHING) * tensor(
            policies
        ) + LABEL_SMOOTHING / 64.0
        value_targets = tensor(outcomes)
        policy_w = tensor(policy_weights)
        value_w = tensor(value_weights) * (0.5 + 1.5 * phase)

        logits, values = self.net(x)
        policy_loss_per = -(policy_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        policy_loss = (policy_loss_per * policy_w).sum() / policy_w.sum().clamp(min=1.0)
        value_loss = (
            (values - value_targets) ** 2 * value_w
        ).sum() / value_w.sum().clamp(min=1.0)
        loss = policy_loss + value_loss

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP_NORM)
        self._optimizer.step()
        return float(loss.item())

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
