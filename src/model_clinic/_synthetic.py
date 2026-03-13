"""Synthetic model generators for testing and demos.

Generate deliberately broken PyTorch models to test model-clinic's
detection and treatment capabilities. Useful for:
- CI/CD pipeline testing
- Documentation demos
- Verifying detector coverage
- Educational examples
"""

import torch
import torch.nn as nn


def make_healthy_mlp(hidden=256, layers=4):
    """Create a healthy MLP state dict with no issues.

    Returns:
        state_dict: dict of param_name -> tensor
    """
    sd = {}
    for i in range(layers):
        in_dim = hidden
        out_dim = hidden
        # Kaiming init
        w = torch.randn(out_dim, in_dim) * (2.0 / in_dim) ** 0.5
        b = torch.zeros(out_dim)
        sd[f"layers.{i}.linear.weight"] = w
        sd[f"layers.{i}.linear.bias"] = b
        # LayerNorm
        sd[f"layers.{i}.norm.weight"] = torch.ones(hidden)
        sd[f"layers.{i}.norm.bias"] = torch.zeros(hidden)
    # Output head
    sd["lm_head.weight"] = torch.randn(1000, hidden) * 0.02
    return sd


def make_dead_neuron_model(hidden=256, layers=4, dead_pct=0.3):
    """Create a model with dead neurons.

    Args:
        dead_pct: fraction of neurons to zero out per layer
    """
    sd = make_healthy_mlp(hidden, layers)
    for i in range(layers):
        w = sd[f"layers.{i}.linear.weight"]
        n_dead = int(w.shape[0] * dead_pct)
        w[:n_dead] = 0  # Zero out rows
    return sd


def make_nan_model(hidden=256, layers=4, nan_layer=2, nan_count=10):
    """Create a model with NaN values in specific layers."""
    sd = make_healthy_mlp(hidden, layers)
    w = sd[f"layers.{nan_layer}.linear.weight"]
    flat = w.flatten()
    indices = torch.randperm(flat.numel())[:nan_count]
    flat[indices] = float("nan")
    sd[f"layers.{nan_layer}.linear.weight"] = flat.reshape(w.shape)
    return sd


def make_exploding_model(hidden=256, layers=4, scale=1000):
    """Create a model with exploding weight norms."""
    sd = make_healthy_mlp(hidden, layers)
    for i in range(layers):
        sd[f"layers.{i}.linear.weight"] *= scale
    return sd


def make_norm_drift_model(hidden=256, layers=4, drift=3.0):
    """Create a model with drifted LayerNorm weights."""
    sd = make_healthy_mlp(hidden, layers)
    for i in range(layers):
        sd[f"layers.{i}.norm.weight"] = torch.ones(hidden) * drift
    return sd


def make_collapsed_model(hidden=256, layers=4):
    """Create a model showing token collapse (identical lm_head rows)."""
    sd = make_healthy_mlp(hidden, layers)
    # Make many lm_head rows identical
    template = sd["lm_head.weight"][0].clone()
    sd["lm_head.weight"][:500] = template  # 50% of vocab identical
    return sd


def make_heavy_tails_model(hidden=256, layers=4, kurtosis_target=100):
    """Create a model with heavy-tailed weight distributions."""
    sd = make_healthy_mlp(hidden, layers)
    for i in range(layers):
        w = sd[f"layers.{i}.linear.weight"]
        # Add extreme outliers
        n_outliers = max(1, w.numel() // 100)
        flat = w.flatten()
        outlier_idx = torch.randperm(flat.numel())[:n_outliers]
        flat[outlier_idx] = torch.randn(n_outliers) * 50
        sd[f"layers.{i}.linear.weight"] = flat.reshape(w.shape)
    return sd


def make_duplicate_rows_model(hidden=256, layers=4):
    """Create a model with identical weight rows."""
    sd = make_healthy_mlp(hidden, layers)
    w = sd["layers.1.linear.weight"]
    # Make first 10 rows identical
    w[:10] = w[0]
    return sd


def make_stuck_gates_model(hidden=256, layers=4):
    """Create a model with stuck gates."""
    sd = make_healthy_mlp(hidden, layers)
    sd["layers.0.gate"] = torch.tensor(-20.0)  # Stuck closed
    sd["layers.2.gate"] = torch.tensor(20.0)   # Stuck open
    return sd


def make_corrupted_model(hidden=256, layers=4):
    """Create a model with weight corruption (all-zero layer, constant values)."""
    sd = make_healthy_mlp(hidden, layers)
    sd["layers.1.linear.weight"] = torch.zeros(hidden, hidden)  # All zeros
    sd["layers.3.linear.weight"] = torch.ones(hidden, hidden) * 0.5  # Constant
    return sd


def make_everything_broken(hidden=256, layers=6):
    """Create a model with EVERY type of issue for demo purposes.

    This is the kitchen-sink model -- every detector should find something.
    """
    sd = make_healthy_mlp(hidden, layers)

    # Dead neurons (layer 0)
    sd["layers.0.linear.weight"][:50] = 0

    # NaN (layer 1)
    w = sd["layers.1.linear.weight"]
    w.flatten()[0] = float("nan")
    w.flatten()[1] = float("inf")

    # Exploding norm (layer 2)
    sd["layers.2.linear.weight"] *= 500

    # Norm drift (layer 3)
    sd["layers.3.norm.weight"] = torch.ones(hidden) * 5.0

    # Saturated weights (layer 4)
    w = sd["layers.4.linear.weight"]
    w_max = w.abs().max().item()
    mask = torch.rand_like(w) < 0.5
    w[mask] = w_max  # 50% at max

    # Identical rows (layer 5)
    sd["layers.5.linear.weight"][:20] = sd["layers.5.linear.weight"][0]

    # Stuck gates
    sd["layers.0.gate"] = torch.tensor(-20.0)
    sd["layers.4.gate"] = torch.tensor(20.0)

    # Token collapse
    template = sd["lm_head.weight"][0].clone()
    sd["lm_head.weight"][:600] = template

    # All-zero corruption
    sd["layers.3.linear.bias"] = torch.zeros(hidden)

    return sd


# Convenience dict for CLI/demos
SYNTHETIC_MODELS = {
    "healthy": make_healthy_mlp,
    "dead-neurons": make_dead_neuron_model,
    "nan": make_nan_model,
    "exploding": make_exploding_model,
    "norm-drift": make_norm_drift_model,
    "collapsed": make_collapsed_model,
    "heavy-tails": make_heavy_tails_model,
    "duplicate-rows": make_duplicate_rows_model,
    "stuck-gates": make_stuck_gates_model,
    "corrupted": make_corrupted_model,
    "everything-broken": make_everything_broken,
}
