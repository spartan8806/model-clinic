"""Unified CLI for model-clinic.

Usage:
    model-clinic exam checkpoint.pt
    model-clinic exam checkpoint.pt --runtime
    model-clinic treat checkpoint.pt --save treated.pt
    model-clinic xray checkpoint.pt
    model-clinic diff a.pt b.pt
    model-clinic compare before.pt after.pt
    model-clinic health checkpoint.pt
    model-clinic surgery checkpoint.pt --interactive
    model-clinic ablate --model model --hf --heads --layer 12
    model-clinic neurons --prompts "What is 2+2?"
    model-clinic attention "What is the capital of France?"
    model-clinic logit-lens "The sky is"
    model-clinic badge checkpoint.pt
    model-clinic badge checkpoint.pt --svg
    model-clinic badge checkpoint.pt --svg -o badge.svg
    model-clinic badge checkpoint.pt --model-card
    model-clinic badge checkpoint.pt --model-card -o CARD_SNIPPET.md
    model-clinic autopsy dead_model.pt
    model-clinic prune-suggest model.pt
"""

import sys


def main():
    """Route to subcommand or show help."""
    # For sub-tools, delegate directly to their main() with sys.argv rewritten.
    # This avoids duplicating argparse definitions and keeps each tool self-contained.
    if len(sys.argv) < 2:
        _print_help()
        return

    command = sys.argv[1]

    # exam and treat are handled by the clinic module
    if command in ("exam", "treat"):
        from model_clinic.clinic import cli_main
        cli_main()
        return

    if command == "convert":
        _cmd_convert()
        return

    # Plugin commands
    if command == "plugins":
        sys.argv = ["model-clinic plugins"] + sys.argv[2:]
        from model_clinic._tools.plugins_cmd import main as plugins_main
        plugins_main()
        return

    if command == "new-plugin":
        sys.argv = ["model-clinic new-plugin"] + sys.argv[2:]
        from model_clinic._tools.new_plugin import main as new_plugin_main
        new_plugin_main()
        return

    # Map subcommand -> tool module
    _TOOL_MAP = {
        "xray": "model_clinic._tools.xray",
        "diff": "model_clinic._tools.diff",
        "compare": "model_clinic._tools.compare",
        "health": "model_clinic._tools.health",
        "surgery": "model_clinic._tools.surgery",
        "ablate": "model_clinic._tools.ablate",
        "neurons": "model_clinic._tools.neurons",
        "attention": "model_clinic._tools.attention",
        "logit-lens": "model_clinic._tools.logit_lens",
        "validate": "model_clinic._tools.validate",
        "report": "model_clinic._tools.report",
        "demo": "model_clinic._tools.demo",
        "badge": "model_clinic._tools.badge",
        "mri": "model_clinic._tools.mri",
        "autopsy": "model_clinic._tools.autopsy",
        "prune-suggest": "model_clinic._tools.prune_suggest",
        "dashboard": "model_clinic._tools.dashboard",
        "spectral": "model_clinic._tools.spectral_cmd",
        "graft": "model_clinic._tools.graft_cmd",
        "activation-audit": "model_clinic._tools.activation_cmd",
        "activation-repair": "model_clinic._tools.activation_cmd",
    }

    if command in _TOOL_MAP:
        # Remove the subcommand from argv so the tool's argparse works correctly
        sys.argv = [f"model-clinic {command}"] + sys.argv[2:]
        import importlib
        mod = importlib.import_module(_TOOL_MAP[command])
        mod.main()
        return

    if command in ("--help", "-h"):
        _print_help()
        return

    print(f"Unknown command: {command}", file=sys.stderr)
    _print_help()
    sys.exit(1)


def _cmd_convert():
    """Show format conversion guidance when a user passes an unsupported file."""
    path = sys.argv[2] if len(sys.argv) > 2 else None
    ext = ""
    if path:
        from pathlib import Path as _Path
        ext = _Path(path).suffix.lower()

    if ext == ".gguf" or (path and "gguf" in path.lower()):
        print(
            f"model-clinic does not directly analyze GGUF files.\n"
            f"To analyze, convert first:\n"
            f"  pip install llama-cpp-python\n"
            f"  python -c \"from llama_cpp import Llama; m = Llama('{path or 'model.gguf'}'); m.save_state()\"\n"
            f"\n"
            f"Supported formats for analysis: .pt .pth .bin .safetensors .onnx"
        )
    elif ext in (".pt", ".pth", ".bin", ".safetensors", ".onnx", ".engine", ".trt"):
        print(
            f"{path or ext} is already a supported format.\n"
            f"Run: model-clinic exam {path or ext}"
        )
    else:
        print(
            "model-clinic convert — Format conversion guidance\n"
            "\n"
            "Supported formats for analysis:\n"
            "  .pt / .pth / .bin   PyTorch checkpoint (native)\n"
            "  .safetensors        Safetensors (pip install model-clinic[safetensors])\n"
            "  .onnx               ONNX model    (pip install model-clinic[onnx])\n"
            "  .engine / .trt      TensorRT      (requires tensorrt; metadata only)\n"
            "\n"
            "Unsupported formats and how to convert them:\n"
            "\n"
            "  GGUF (.gguf):\n"
            "    pip install llama-cpp-python\n"
            "    python -c \"from llama_cpp import Llama; m = Llama('model.gguf'); m.save_state()\"\n"
            "\n"
            "  GGML (.ggml / .bin from llama.cpp):\n"
            "    Convert to ONNX via optimum-cli, then: model-clinic exam model.onnx\n"
            "\n"
            "Usage: model-clinic convert <path>"
        )


def _print_help():
    print("""model-clinic — Diagnose, treat, and understand neural network models

Commands:
  exam        Diagnose model health, show treatment plan
  treat       Diagnose and apply fixes
  xray        Per-parameter weight stats (shape, norm, sparsity)
  diff        Compare two checkpoints param-by-param
  compare     Compare health impact between two checkpoints
  health      Quick health check (dead neurons, norms, gates)
  surgery     Direct parameter modification (interactive REPL)
  ablate      Disable parts systematically, measure impact
  neurons     Profile neuron activations across prompts
  attention   Attention patterns per head per layer
  logit-lens  Watch predictions form layer by layer
  validate    Verify a checkpoint loads and infers correctly
  report      Generate an HTML diagnostic report
  dashboard   Serve an interactive model health dashboard in your browser
  demo        Generate and examine a synthetic broken model
  badge       Generate a health badge URL, SVG, or model card snippet
  convert     Show how to convert unsupported formats (GGUF, etc.) for analysis
  mri         Deep per-layer weight analysis using SVD decomposition
  autopsy     Deep forensic analysis for dead or severely broken models
  prune-suggest   Static pruning opportunity analysis (no forward pass)
  spectral    SVD spectral analysis and denoising (repair high condition numbers)
  graft       Cross-checkpoint grafting: merge healthiest parts of multiple checkpoints
  activation-audit   Per-layer activation stats (requires model class)
  activation-repair  Detect and fix destructive layers (requires model class)
  plugins     List installed model-clinic plugins
  new-plugin  Scaffold a new plugin package

Run 'model-clinic <command> --help' for details on each command.
""")


if __name__ == "__main__":
    main()
