# Author: Lorenzo Luzi
# Date: October 2022
"""Utility functions that estimates the Lipschitz constant of a model.
"""
import torch


def get_lipschitz(model, w, h, device='cpu'):
    # Make the coords.
    x = torch.linspace(-1, 1, w).to(device)
    y = torch.linspace(-1, 1, h).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    # Input.
    coords = torch.nn.Parameter(coords)
    # Output.
    J = torch.zeros(1, h * w, 2, 3)
    for i in range(3):
        y = model(coords)
        # Calc norms.
        y_channel = y[:, :, i]
        # Sum and backprop.
        loss = y_channel.sum()
        loss.backward()
        J[:, :, :, i] = coords.grad.clone()
    L = J.norm(p=2, dim=[2, 3])
    return L
