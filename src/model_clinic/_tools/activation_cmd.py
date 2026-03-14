"""CLI commands for activation-guided repair (Level 5).

Usage:
    model-clinic activation-audit <model_path> --model-class dotted.path [--data calibration.jsonl]
    model-clinic activation-repair <model_path> --model-class dotted.path [--data calibration.jsonl] [--strategy shrink] [-o repaired.pt]
"""

import argparse
import importlib
import sys

import torch


def _load_model_class(dotted_path: str):
    """Import and return the class at *dotted_path* (e.g. 'my_pkg.Model')."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"--model-class must be a dotted path like 'pkg.Module', got {dotted_path!r}"
        )
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _load_model(args):
    """Instantiate the model and load its state dict."""
    cls = _load_model_class(args.model_class)
    model = cls()
    sd = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    return model


def _get_calibration(args):
    """Return a list of input tensors for calibration."""
    if args.data:
        from model_clinic._repair.calibration import load_calibration_data
        return load_calibration_data(args.data)
    else:
        from model_clinic._repair.calibration import generate_random_calibration
        print("No --data provided, using random calibration data.", file=sys.stderr)
        return generate_random_calibration(num_samples=20, seq_length=64)


# ---------------------------------------------------------------------------
# activation-audit
# ---------------------------------------------------------------------------

def main_audit():
    parser = argparse.ArgumentParser(
        description="Run an activation audit on a model checkpoint."
    )
    parser.add_argument("model", help="Path to .pt checkpoint")
    parser.add_argument(
        "--model-class", required=True,
        help="Dotted Python path to the model class (e.g. 'my_pkg.MyModel')",
    )
    parser.add_argument("--data", default=None, help="Calibration data file (.jsonl/.txt/.pt)")
    parser.add_argument("--device", default="cpu", help="Device to run on (default: cpu)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output in JSON format")
    args = parser.parse_args()

    from model_clinic._repair.activation import (
        activation_audit, find_destructive_layers,
    )

    print(f"Loading model from {args.model} ...", file=sys.stderr)
    model = _load_model(args)
    calibration = _get_calibration(args)

    print(f"Running activation audit ({len(calibration)} samples) ...", file=sys.stderr)
    stats = activation_audit(model, calibration, device=args.device)
    destructive = find_destructive_layers(stats)

    if args.json_output:
        import json
        out = {
            "layers": {
                name: {
                    "input_rank": round(s.input_rank, 4),
                    "output_rank": round(s.output_rank, 4),
                    "rank_change": round(s.rank_change, 4),
                    "input_entropy": round(s.input_entropy, 4),
                    "output_entropy": round(s.output_entropy, 4),
                    "entropy_change": round(s.entropy_change, 4),
                    "cosine_similarity": round(s.cosine_similarity, 4),
                    "norm_ratio": round(s.norm_ratio, 4),
                }
                for name, s in stats.items()
            },
            "destructive": destructive,
        }
        print(json.dumps(out, indent=2))
        return

    # Table output
    print(f"\n{'Layer':<45s} {'InRank':>7s} {'OutRank':>8s} {'dRank':>7s} "
          f"{'InEnt':>6s} {'OutEnt':>7s} {'dEnt':>6s} {'Cos':>5s} {'Norm':>6s} {'Status'}")
    print("-" * 120)
    for name, s in stats.items():
        status = "DESTRUCTIVE" if name in destructive else "ok"
        print(
            f"  {name:<43s} {s.input_rank:7.2f} {s.output_rank:8.2f} "
            f"{s.rank_change:+7.2f} {s.input_entropy:6.2f} {s.output_entropy:7.2f} "
            f"{s.entropy_change:+6.2f} {s.cosine_similarity:5.2f} "
            f"{s.norm_ratio:6.2f}  {status}"
        )

    print(f"\nTotal layers: {len(stats)}  |  Destructive: {len(destructive)}")
    if destructive:
        print(f"Destructive layers: {', '.join(destructive)}")


# ---------------------------------------------------------------------------
# activation-repair
# ---------------------------------------------------------------------------

def main_repair():
    parser = argparse.ArgumentParser(
        description="Audit and repair destructive layers in a model checkpoint."
    )
    parser.add_argument("model", help="Path to .pt checkpoint")
    parser.add_argument(
        "--model-class", required=True,
        help="Dotted Python path to the model class (e.g. 'my_pkg.MyModel')",
    )
    parser.add_argument("--data", default=None, help="Calibration data file (.jsonl/.txt/.pt)")
    parser.add_argument("--device", default="cpu", help="Device to run on (default: cpu)")
    parser.add_argument("--strategy", default="shrink",
                        choices=["shrink", "passthrough", "interpolate"],
                        help="Repair strategy (default: shrink)")
    parser.add_argument("--shrink-factor", type=float, default=0.1,
                        help="Factor for shrink strategy (default: 0.1)")
    parser.add_argument("-o", "--output", default="repaired.pt",
                        help="Output path (default: repaired.pt)")
    args = parser.parse_args()

    from model_clinic._repair.activation import (
        activation_audit, find_destructive_layers, activation_repair,
    )

    print(f"Loading model from {args.model} ...", file=sys.stderr)
    model = _load_model(args)
    calibration = _get_calibration(args)

    print(f"Running activation audit ({len(calibration)} samples) ...", file=sys.stderr)
    stats = activation_audit(model, calibration, device=args.device)
    destructive = find_destructive_layers(stats)

    if not destructive:
        print("No destructive layers found. Model is clean.")
        return

    print(f"Found {len(destructive)} destructive layers: {', '.join(destructive)}")
    print(f"Applying '{args.strategy}' repair ...")

    sd, repaired = activation_repair(
        model, destructive,
        strategy=args.strategy,
        shrink_factor=args.shrink_factor,
    )

    torch.save(sd, args.output)
    print(f"Repaired {len(repaired)} layers. Saved to {args.output}")
    for name in repaired:
        print(f"  - {name}")


# ---------------------------------------------------------------------------
# Unified entry point (called from cli.py routing)
# ---------------------------------------------------------------------------

def main():
    """Dispatched from cli.py — should not normally be called directly."""
    # This gets invoked as 'activation-audit' or 'activation-repair' from the
    # CLI router.  The sys.argv[0] tells us which sub-command.
    if "repair" in sys.argv[0]:
        main_repair()
    else:
        main_audit()


if __name__ == "__main__":
    main()
