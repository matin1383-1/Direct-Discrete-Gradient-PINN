"""Neural networks and exact hard-constraint output transformations."""
from __future__ import annotations

from math import pi

import torch
from torch import nn


class FullyConnectedTanh(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int = 1):
        super().__init__()
        widths = [input_dim, *hidden_layers, output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            # Glorot uniform is the initialization used in the paper protocol.
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for layer in self.layers[:-1]:
            value = torch.tanh(layer(value))
        return self.layers[-1](value)


class ConstrainedSolution(nn.Module):
    """Wrap a raw network with the problem's prescribed output transformation."""

    def __init__(self, problem_name: str, raw_network: nn.Module):
        super().__init__()
        self.problem_name = problem_name
        self.raw_network = raw_network

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        raw = self.raw_network(coordinates)
        x = coordinates[:, 0:1]
        if self.problem_name == "poisson1d":
            return x + torch.tanh(x) * torch.tanh(pi - x) * raw
        if self.problem_name == "poisson2d":
            y = coordinates[:, 1:2]
            return x * y * (1.0 - x) * (1.0 - y) * raw
        if self.problem_name == "burgers":
            t = coordinates[:, 1:2]
            return (1.0 - x**2) * (1.0 - torch.exp(-t)) * raw - torch.sin(pi * x)
        if self.problem_name == "allen_cahn":
            t = coordinates[:, 1:2]
            return t * (1.0 - x**2) * raw + x**2 * torch.cos(pi * x)
        if self.problem_name == "heat":
            return raw
        raise ValueError(f"unknown problem {self.problem_name!r}")


def make_network(problem, *, device=None, dtype=torch.float32) -> ConstrainedSolution:
    raw = FullyConnectedTanh(problem.input_dim, problem.hidden_layers, 1)
    return ConstrainedSolution(problem.name, raw).to(device=device, dtype=dtype)
