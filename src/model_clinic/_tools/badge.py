"""Badge: Generate a health badge or model card snippet for a model.

Usage:
    model-clinic badge model.pt
    model-clinic badge model.pt --svg
    model-clinic badge model.pt --svg -o badge.svg
    model-clinic badge model.pt --model-card
    model-clinic badge model.pt --model-card -o CARD_SNIPPET.md
    model-clinic badge model.pt --hf
"""

import argparse
import sys

from model_clinic._loader import load_state_dict, build_meta
from model_clinic.clinic import diagnose
from model_clinic._health_score import compute_health_score
from model_clinic._badge import (
    generate_badge_url,
    generate_badge_svg,
    generate_model_card_snippet,
    save_badge_svg,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a health badge or model card snippet",
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true",
                        help="Load as HuggingFace model")
    parser.add_argument("--svg", action="store_true",
                        help="Output inline SVG instead of a shields.io URL")
    parser.add_argument("--model-card", action="store_true",
                        help="Output a full model card markdown snippet")
    parser.add_argument("--model-name", default="",
                        help="Optional model name for the model card header")
    parser.add_argument("-o", "--output", default=None,
                        help="Save output to this file instead of printing to stdout")
    args = parser.parse_args()

    # Load and diagnose
    print(f"Loading: {args.model}", file=sys.stderr)
    state_dict, raw_meta = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict,
                      source=raw_meta.get("source", "unknown"),
                      extra=raw_meta)

    print("Running diagnostics...", file=sys.stderr)
    findings = diagnose(state_dict)
    health = compute_health_score(findings)

    print(f"Health: {health.overall}/100 {health.grade}", file=sys.stderr)

    # Generate the requested output
    if args.model_card:
        model_name = args.model_name or args.model
        content = generate_model_card_snippet(health, findings, model_name=model_name)
    elif args.svg:
        content = generate_badge_svg(health)
    else:
        content = generate_badge_url(health)

    if args.output:
        if args.svg and not args.model_card:
            # Use the dedicated save helper so the path is explicit
            save_badge_svg(health, args.output)
        else:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(content)
                if not content.endswith("\n"):
                    fh.write("\n")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
