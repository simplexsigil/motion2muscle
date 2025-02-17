import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange as rea


class FConvMotionToMuscleModel(nn.Module):
    def __init__(self, input_width=263, output_width=402, width=256, num_layers=8):
        super().__init__()

        self.input_proj = nn.Sequential(nn.Conv1d(input_width, width, kernel_size=3, padding=1), nn.ReLU())

        self.conv_layers = nn.ModuleList([nn.Conv1d(width, width, kernel_size=3, padding=1) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(width)

        self.output_proj = nn.Conv1d(width, output_width, kernel_size=1)

    def forward(self, x):
        # x is expected to be in shape (batch, time, channels)
        x = rea(x, "b t c -> b c t")

        x = self.input_proj(x)

        for layer in self.conv_layers:
            residual = x
            x = F.relu(layer(x))
            x = x + residual

        # Apply LayerNorm
        x = rea(x, "b c t -> b t c")
        x = self.norm(x)
        x = rea(x, "b t c -> b c t")

        x = self.output_proj(x)

        # Convert back to (batch, time, channels)
        x = rea(x, "b c t -> b t c")

        return x
