import mlx.core as mx
import mlx.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, channels: int = 32, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm(channels),
            nn.ReLU(),
        )
        self.blocks = [ResBlock(channels) for _ in range(n_blocks)]
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm(2)
        self.policy_fc = nn.Linear(2 * 64, 64)
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, x):
        # x: (B, 3, 8, 8) — MLX uses (B, H, W, C) internally but we transpose
        # MLX Conv2d expects (B, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 1))  # → (B, 8, 8, 3)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        # Policy
        p = nn.relu(self.policy_bn(self.policy_conv(x)))  # (B, 8, 8, 2)
        p = p.reshape(p.shape[0], -1)  # (B, 128)
        p = self.policy_fc(p)  # (B, 64)
        # Value
        v = nn.relu(self.value_bn(self.value_conv(x)))  # (B, 8, 8, 1)
        v = v.reshape(v.shape[0], -1)  # (B, 64)
        v = nn.relu(self.value_fc1(v))
        v = mx.tanh(self.value_fc2(v))  # (B, 1)
        return p, v.squeeze(-1)
