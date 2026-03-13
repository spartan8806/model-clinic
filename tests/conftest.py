"""Shared test fixtures for model-clinic."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def tiny_state_dict():
    """A minimal state dict with known properties for testing."""
    torch.manual_seed(42)
    return {
        "embed_tokens.weight": torch.randn(100, 64),          # embedding
        "layers.0.attention.q_proj.weight": torch.randn(64, 64),
        "layers.0.attention.k_proj.weight": torch.randn(64, 64),
        "layers.0.attention.v_proj.weight": torch.randn(64, 64),
        "layers.0.attention.o_proj.weight": torch.randn(64, 64),
        "layers.0.mlp.gate_proj.weight": torch.randn(256, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 256),
        "layers.0.attn_norm.weight": torch.ones(64),          # healthy norm
        "layers.0.ffn_norm.weight": torch.ones(64),           # healthy norm
        "layers.1.attention.q_proj.weight": torch.randn(64, 64),
        "layers.1.attention.k_proj.weight": torch.randn(64, 64),
        "layers.1.attention.v_proj.weight": torch.randn(64, 64),
        "layers.1.mlp.gate_proj.weight": torch.randn(256, 64),
        "layers.1.mlp.down_proj.weight": torch.randn(64, 256),
        "final_norm.weight": torch.ones(64),
        "lm_head.weight": torch.randn(100, 64),
    }


@pytest.fixture
def sick_state_dict():
    """A state dict with known pathologies."""
    torch.manual_seed(42)
    sd = {
        # Healthy
        "embed_tokens.weight": torch.randn(100, 64),
        "layers.0.attention.q_proj.weight": torch.randn(64, 64),
        "layers.0.attention.k_proj.weight": torch.randn(64, 64),
        "layers.0.attention.v_proj.weight": torch.randn(64, 64),

        # Dead neurons: row 0 and 1 are zero
        "layers.0.mlp.down_proj.weight": torch.randn(64, 256),

        # Drifted norm (threshold is 1.5: |mean - 1.0| > 1.5 → mean must exceed 2.5)
        "final_norm.weight": torch.full((64,), 3.0),

        # Gate stuck closed
        "wrapper/gate": torch.tensor(-10.0),

        # Gate stuck open
        "wrapper/pre_memory_gate": torch.tensor(10.0),

        # NaN
        "bad_param": torch.randn(32, 32),

        # Exploding norm
        "exploding_layer.weight": torch.randn(64, 64) * 100,
    }

    # Zero out rows 0 and 1
    sd["layers.0.mlp.down_proj.weight"][0] = 0
    sd["layers.0.mlp.down_proj.weight"][1] = 0

    # Inject NaN
    sd["bad_param"][5, 5] = float("nan")

    return sd
