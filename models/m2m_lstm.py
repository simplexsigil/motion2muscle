import torch
import torch.nn as nn
from einops import rearrange as rea


class LSTMMotionToMuscleModel(nn.Module):
    def __init__(self, input_width=263, output_width=402, hidden_size=256, num_layers=1):
        super().__init__()

        blocks = []
        blocks.append(nn.Conv1d(input_width, hidden_size, 3, 1, 1))
        blocks.append(nn.ReLU())

        self.input_projection = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=True,  # Set to bidirectional
        )

        # Adjust the output projection to account for bidirectional output
        self.output_proj = nn.Linear(hidden_size * 2, output_width)

    def forward(self, x):
        # Preprocess
        x = x.float()
        x = rea(x, "b l d -> b d l")  # for Conv1D over time.

        x = self.input_projection(x)

        x = rea(x, "b d l -> l b d")  # for LSTM without batch first setting.

        # LSTM layers
        x, _ = self.lstm(x)

        x = rea(x, "l b d -> b l d")

        # Output projection
        x = self.output_proj(x)

        return x
