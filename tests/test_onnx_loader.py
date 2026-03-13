"""Tests for ONNX and TensorRT loader support, GGUF stub, and format auto-detection."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure src is on path (conftest.py also does this, but be explicit)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_clinic._loader import load_onnx, load_tensorrt, load_state_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_onnx_module():
    """Return a minimal mock of the onnx package with numpy_helper.to_array."""
    import numpy as np

    onnx_mod = types.ModuleType("onnx")
    numpy_helper_mod = types.ModuleType("onnx.numpy_helper")

    # Fake TensorProto-like initializer
    class FakeInitializer:
        def __init__(self, name, arr):
            self.name = name
            self._arr = arr

    def to_array(init):
        return init._arr

    numpy_helper_mod.to_array = to_array
    onnx_mod.numpy_helper = numpy_helper_mod

    # Fake graph / model structure
    class FakeGraph:
        def __init__(self, initializers):
            self.initializer = initializers

    class FakeModel:
        def __init__(self, initializers):
            self.graph = FakeGraph(initializers)

    def fake_load(path):
        arr1 = np.random.randn(4, 8).astype(np.float32)
        arr2 = np.random.randn(4).astype(np.float32)
        return FakeModel([
            FakeInitializer("weight", arr1),
            FakeInitializer("bias", arr2),
        ])

    onnx_mod.load = fake_load

    return onnx_mod, numpy_helper_mod


# ---------------------------------------------------------------------------
# load_onnx — ImportError when onnx not installed
# ---------------------------------------------------------------------------

class TestLoadOnnxImportError:

    def test_raises_import_error_when_onnx_missing(self, tmp_path):
        """load_onnx must raise ImportError with pip install hint when onnx absent.

        We block the import by injecting a sentinel value of None for both the
        'onnx' and 'numpy' keys in sys.modules.  Python treats a None value in
        sys.modules as "this module does not exist", which causes `import onnx`
        inside load_onnx() to raise ImportError without triggering a real re-import
        (which would hit circular-import issues on Python 3.14).
        """
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        # Collect every onnx-related key currently in sys.modules
        onnx_keys = [k for k in list(sys.modules) if k == "onnx" or k.startswith("onnx.")]
        saved = {k: sys.modules[k] for k in onnx_keys}

        # Block all onnx sub-modules
        block = {k: None for k in onnx_keys} if onnx_keys else {"onnx": None, "onnx.numpy_helper": None}

        with patch.dict(sys.modules, block):
            with pytest.raises(ImportError) as exc_info:
                load_onnx(str(dummy))

        msg = str(exc_info.value)
        assert "onnx" in msg.lower()
        assert "pip install" in msg
        assert "model-clinic[onnx]" in msg


# ---------------------------------------------------------------------------
# load_onnx — with mocked onnx
# ---------------------------------------------------------------------------

class TestLoadOnnxMocked:

    def test_returns_dict_of_tensors(self, tmp_path):
        """load_onnx should return a state dict of torch.Tensors."""
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        onnx_mod, numpy_helper_mod = _make_fake_onnx_module()

        with patch.dict(sys.modules, {
            "onnx": onnx_mod,
            "onnx.numpy_helper": numpy_helper_mod,
        }):
            sd = load_onnx(str(dummy))

        assert isinstance(sd, dict)
        assert "weight" in sd
        assert "bias" in sd
        assert isinstance(sd["weight"], torch.Tensor)
        assert isinstance(sd["bias"], torch.Tensor)
        assert sd["weight"].shape == (4, 8)
        assert sd["bias"].shape == (4,)

    def test_tensor_dtype_float32(self, tmp_path):
        """Tensors loaded from ONNX float32 initializers should be float32."""
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        onnx_mod, numpy_helper_mod = _make_fake_onnx_module()

        with patch.dict(sys.modules, {
            "onnx": onnx_mod,
            "onnx.numpy_helper": numpy_helper_mod,
        }):
            sd = load_onnx(str(dummy))

        assert sd["weight"].dtype == torch.float32

    def test_empty_model_returns_empty_dict(self, tmp_path):
        """An ONNX model with no initializers should return an empty dict."""
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        onnx_mod, numpy_helper_mod = _make_fake_onnx_module()

        class EmptyGraph:
            initializer = []

        class EmptyModel:
            graph = EmptyGraph()

        onnx_mod.load = lambda path: EmptyModel()

        with patch.dict(sys.modules, {
            "onnx": onnx_mod,
            "onnx.numpy_helper": numpy_helper_mod,
        }):
            sd = load_onnx(str(dummy))

        assert sd == {}


# ---------------------------------------------------------------------------
# Auto-detection: .onnx extension routes to load_onnx
# ---------------------------------------------------------------------------

class TestAutoDetectOnnx:

    def test_dot_onnx_routes_to_load_onnx(self, tmp_path):
        """load_state_dict('.onnx') should call load_onnx and return source='onnx'."""
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        onnx_mod, numpy_helper_mod = _make_fake_onnx_module()

        with patch.dict(sys.modules, {
            "onnx": onnx_mod,
            "onnx.numpy_helper": numpy_helper_mod,
        }):
            sd, meta = load_state_dict(str(dummy))

        assert meta.get("source") == "onnx"
        assert "weight" in sd
        assert "bias" in sd

    def test_dot_onnx_returns_tensors(self, tmp_path):
        """All values in the returned state dict should be torch.Tensors."""
        dummy = tmp_path / "model.onnx"
        dummy.write_bytes(b"fake")

        onnx_mod, numpy_helper_mod = _make_fake_onnx_module()

        with patch.dict(sys.modules, {
            "onnx": onnx_mod,
            "onnx.numpy_helper": numpy_helper_mod,
        }):
            sd, _ = load_state_dict(str(dummy))

        for v in sd.values():
            assert isinstance(v, torch.Tensor)


# ---------------------------------------------------------------------------
# Auto-detection: .engine / .trt routes to load_tensorrt
# ---------------------------------------------------------------------------

class TestAutoDetectTensorRT:

    def _mock_tensorrt(self):
        """Build a minimal mock tensorrt module."""
        trt_mod = types.ModuleType("tensorrt")

        class FakeLogger:
            WARNING = 0
            def __init__(self, level): pass

        class FakeBinding:
            pass

        class FakeEngine:
            num_bindings = 2
            def get_binding_name(self, i): return f"binding_{i}"
            def get_binding_shape(self, i): return (1, 4) if i == 0 else (1,)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeRuntime:
            def __init__(self, logger): pass
            def deserialize_cuda_engine(self, data): return FakeEngine()

        trt_mod.Logger = FakeLogger
        trt_mod.Runtime = FakeRuntime
        return trt_mod

    def test_dot_engine_routes_to_load_tensorrt(self, tmp_path):
        """.engine files should route to load_tensorrt and return source='tensorrt'."""
        dummy = tmp_path / "model.engine"
        dummy.write_bytes(b"\x00" * 16)

        trt_mod = self._mock_tensorrt()

        with patch.dict(sys.modules, {"tensorrt": trt_mod}):
            sd, meta = load_state_dict(str(dummy))

        assert meta.get("source") == "tensorrt"
        assert isinstance(sd, dict)

    def test_dot_trt_routes_to_load_tensorrt(self, tmp_path):
        """.trt files should also route to load_tensorrt."""
        dummy = tmp_path / "model.trt"
        dummy.write_bytes(b"\x00" * 16)

        trt_mod = self._mock_tensorrt()

        with patch.dict(sys.modules, {"tensorrt": trt_mod}):
            sd, meta = load_state_dict(str(dummy))

        assert meta.get("source") == "tensorrt"

    def test_tensorrt_import_error_when_missing(self, tmp_path):
        """load_tensorrt raises ImportError with install hint when tensorrt absent."""
        dummy = tmp_path / "model.engine"
        dummy.write_bytes(b"\x00" * 16)

        # Block tensorrt by injecting None sentinel; avoids real re-import attempts
        trt_keys = [k for k in list(sys.modules) if k == "tensorrt" or k.startswith("tensorrt.")]
        block = {k: None for k in trt_keys} if trt_keys else {"tensorrt": None}

        with patch.dict(sys.modules, block):
            with pytest.raises(ImportError) as exc_info:
                load_tensorrt(str(dummy))

        msg = str(exc_info.value)
        assert "tensorrt" in msg.lower()
        assert "nvidia" in msg.lower() or "pip install" in msg.lower()

    def test_engine_returns_binding_shapes(self, tmp_path):
        """load_tensorrt should return binding shape tensors in the dict."""
        dummy = tmp_path / "model.engine"
        dummy.write_bytes(b"\x00" * 16)

        trt_mod = self._mock_tensorrt()

        with patch.dict(sys.modules, {"tensorrt": trt_mod}):
            sd = load_tensorrt(str(dummy))

        # At least one binding key should exist
        binding_keys = [k for k in sd if k.startswith("binding/")]
        assert len(binding_keys) > 0
        for k in binding_keys:
            assert isinstance(sd[k], torch.Tensor)


# ---------------------------------------------------------------------------
# GGUF: helpful error
# ---------------------------------------------------------------------------

class TestGgufError:

    def test_gguf_raises_not_implemented(self, tmp_path):
        """.gguf files should raise NotImplementedError with conversion hint."""
        dummy = tmp_path / "model.gguf"
        dummy.write_bytes(b"GGUF fake data")

        with pytest.raises(NotImplementedError) as exc_info:
            load_state_dict(str(dummy))

        msg = str(exc_info.value)
        assert "gguf" in msg.lower()
        assert "llama" in msg.lower() or "convert" in msg.lower()

    def test_gguf_error_mentions_supported_formats(self, tmp_path):
        """The GGUF error should list supported formats."""
        dummy = tmp_path / "model.gguf"
        dummy.write_bytes(b"GGUF fake data")

        with pytest.raises(NotImplementedError) as exc_info:
            load_state_dict(str(dummy))

        msg = str(exc_info.value)
        assert ".onnx" in msg or "onnx" in msg.lower()
        assert ".pt" in msg or ".safetensors" in msg


# ---------------------------------------------------------------------------
# Optional: full round-trip with real onnx (skipped if not installed)
# ---------------------------------------------------------------------------

try:
    import onnx as _onnx_check  # noqa: F401
    # Verify onnx is actually usable (Python 3.14 has circular import issues)
    _ = _onnx_check.version.version  # type: ignore[attr-defined]
    _ONNX_AVAILABLE = True
except (ImportError, AttributeError):
    _ONNX_AVAILABLE = False


@pytest.mark.skipif(not _ONNX_AVAILABLE, reason="onnx not installed")
class TestLoadOnnxReal:
    """Integration tests using a real tiny ONNX model built from scratch."""

    def _make_tiny_onnx(self, path: str):
        """Create a minimal valid ONNX model with one weight initializer."""
        import onnx
        import numpy as np
        from onnx import numpy_helper, TensorProto, helper

        # Single Linear: y = x @ weight.T + bias
        weight = np.random.randn(4, 8).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)

        weight_init = numpy_helper.from_array(weight, name="fc.weight")
        bias_init = numpy_helper.from_array(bias, name="fc.bias")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

        gemm = helper.make_node(
            "Gemm",
            inputs=["X", "fc.weight", "fc.bias"],
            outputs=["Y"],
            transB=1,
        )

        graph = helper.make_graph([gemm], "tiny", [X], [Y], [weight_init, bias_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, path)
        return weight, bias

    def test_real_onnx_load(self, tmp_path):
        """Load a real tiny ONNX model and verify weight tensors match."""
        onnx_path = str(tmp_path / "tiny.onnx")
        weight_np, bias_np = self._make_tiny_onnx(onnx_path)

        sd = load_onnx(onnx_path)

        assert "fc.weight" in sd
        assert "fc.bias" in sd
        assert sd["fc.weight"].shape == (4, 8)
        assert sd["fc.bias"].shape == (4,)
        assert torch.allclose(sd["fc.weight"], torch.from_numpy(weight_np))
        assert torch.allclose(sd["fc.bias"], torch.from_numpy(bias_np))

    def test_real_onnx_via_load_state_dict(self, tmp_path):
        """End-to-end: load_state_dict routes .onnx to load_onnx correctly."""
        onnx_path = str(tmp_path / "tiny.onnx")
        self._make_tiny_onnx(onnx_path)

        sd, meta = load_state_dict(onnx_path)

        assert meta["source"] == "onnx"
        assert "fc.weight" in sd
        assert isinstance(sd["fc.weight"], torch.Tensor)

    def test_real_onnx_build_meta(self, tmp_path):
        """build_meta should not crash on an ONNX-loaded state dict."""
        from model_clinic._loader import build_meta

        onnx_path = str(tmp_path / "tiny.onnx")
        self._make_tiny_onnx(onnx_path)

        sd, _ = load_state_dict(onnx_path)
        meta = build_meta(sd, source="onnx")

        assert meta.num_tensors == 2
        assert meta.num_params > 0
        assert meta.source == "onnx"
