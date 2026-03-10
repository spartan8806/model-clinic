"""Tests for the unified loader."""

import torch
import tempfile
import os
from model_clinic._loader import load_state_dict, build_meta, save_state_dict


class TestLoadStateDict:

    def test_load_generic_checkpoint(self, tmp_path):
        sd = {"weight": torch.randn(32, 32), "bias": torch.randn(32)}
        path = tmp_path / "model.pt"
        torch.save({"model_state_dict": sd}, str(path))

        loaded, meta = load_state_dict(str(path))
        assert "weight" in loaded
        assert "bias" in loaded
        assert loaded["weight"].shape == (32, 32)

    def test_load_flat_checkpoint(self, tmp_path):
        sd = {"weight": torch.randn(32, 32), "bias": torch.randn(32)}
        path = tmp_path / "model.pt"
        torch.save(sd, str(path))

        loaded, meta = load_state_dict(str(path))
        assert "weight" in loaded

    def test_load_composite_checkpoint(self, tmp_path):
        ck = {
            "model_type": "ATLESQwen",
            "wrapper_state_dict": {
                "gate": torch.tensor(-3.0),
                "layers": {"0.weight": torch.randn(64, 64)},
            },
            "memory_state_dict": {"keys": torch.randn(256, 64)},
        }
        path = tmp_path / "model.pt"
        torch.save(ck, str(path))

        loaded, meta = load_state_dict(str(path))
        assert "wrapper/gate" in loaded
        assert "wrapper/layers/0.weight" in loaded
        assert "memory_state_dict/keys" in loaded


class TestBuildMeta:

    def test_infer_shape(self, tiny_state_dict):
        meta = build_meta(tiny_state_dict)
        assert meta.hidden_size == 64
        assert meta.num_layers == 2
        assert meta.vocab_size == 100
        assert meta.num_tensors == len(tiny_state_dict)


class TestSaveStateDict:

    def test_save_and_reload(self, tmp_path):
        sd = {"weight": torch.randn(32, 32), "bias": torch.randn(32)}
        orig_path = tmp_path / "original.pt"
        torch.save({"model_state_dict": sd}, str(orig_path))

        # Modify
        modified_sd, _ = load_state_dict(str(orig_path))
        modified_sd["weight"] = torch.zeros(32, 32)

        # Save
        out_path = tmp_path / "treated.pt"
        save_state_dict(modified_sd, str(orig_path), str(out_path))

        # Reload and verify
        reloaded, _ = load_state_dict(str(out_path))
        assert torch.allclose(reloaded["weight"], torch.zeros(32, 32))
