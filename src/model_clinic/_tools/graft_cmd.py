"""Graft — cross-checkpoint grafting CLI.

Pick the healthiest version of each parameter (or module) from multiple
checkpoints and merge them into a single checkpoint.

Usage:
    model-clinic graft checkpoint1.pt checkpoint2.pt
    model-clinic graft checkpoint1.pt checkpoint2.pt -o merged.pt
    model-clinic graft checkpoint1.pt checkpoint2.pt --manifest manifest.json
    model-clinic graft checkpoint1.pt checkpoint2.pt --module-level
    model-clinic graft checkpoint1.pt checkpoint2.pt --module-level --depth 3
"""

import argparse
import json
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic graft",
        description="Cross-checkpoint grafting: merge the healthiest parts of multiple checkpoints.",
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Paths to checkpoint files (.pt, .pth, .safetensors, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        default="grafted.pt",
        help="Output path for merged checkpoint (default: grafted.pt)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output path for JSON manifest (default: no manifest file)",
    )
    parser.add_argument(
        "--module-level",
        action="store_true",
        default=False,
        help="Graft at module level instead of individual parameters",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Module grouping depth for --module-level (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output manifest as JSON to stdout instead of human-readable report",
    )

    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        print("Warning: only one checkpoint provided — output will be a copy.", file=sys.stderr)

    from model_clinic._repair.graft import graft, graft_modules

    print(f"Loading {len(args.checkpoints)} checkpoint(s)...", file=sys.stderr)

    if args.module_level:
        merged, manifest = graft_modules(args.checkpoints, module_level=True, depth=args.depth)
    else:
        merged, manifest = graft(args.checkpoints)

    # Save merged checkpoint
    torch.save(merged, args.output)
    print(f"Saved merged checkpoint to {args.output}", file=sys.stderr)

    # Save manifest if requested
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest.to_json(), f, indent=2)
        print(f"Saved manifest to {args.manifest}", file=sys.stderr)

    # Output
    if args.json:
        json.dump(manifest.to_json(), sys.stdout, indent=2)
        print(file=sys.stdout)
    else:
        manifest.print_report()


if __name__ == "__main__":
    main()
