"""Unified CLI for model-clinic.

Usage:
    model-clinic exam checkpoint.pt
    model-clinic exam checkpoint.pt --runtime
    model-clinic treat checkpoint.pt --save treated.pt
    model-clinic xray checkpoint.pt
    model-clinic diff a.pt b.pt
    model-clinic health checkpoint.pt
    model-clinic surgery checkpoint.pt --interactive
    model-clinic ablate --model model --hf --heads --layer 12
    model-clinic neurons --prompts "What is 2+2?"
    model-clinic attention "What is the capital of France?"
    model-clinic logit-lens "The sky is"
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

    # Map subcommand -> tool module
    _TOOL_MAP = {
        "xray": "model_clinic._tools.xray",
        "diff": "model_clinic._tools.diff",
        "health": "model_clinic._tools.health",
        "surgery": "model_clinic._tools.surgery",
        "ablate": "model_clinic._tools.ablate",
        "neurons": "model_clinic._tools.neurons",
        "attention": "model_clinic._tools.attention",
        "logit-lens": "model_clinic._tools.logit_lens",
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


def _print_help():
    print("""model-clinic — Diagnose, treat, and understand neural network models

Commands:
  exam        Diagnose model health, show treatment plan
  treat       Diagnose and apply fixes
  xray        Per-parameter weight stats (shape, norm, sparsity)
  diff        Compare two checkpoints param-by-param
  health      Quick health check (dead neurons, norms, gates)
  surgery     Direct parameter modification (interactive REPL)
  ablate      Disable parts systematically, measure impact
  neurons     Profile neuron activations across prompts
  attention   Attention patterns per head per layer
  logit-lens  Watch predictions form layer by layer

Run 'model-clinic <command> --help' for details on each command.
""")


if __name__ == "__main__":
    main()
