"""Report: Generate an HTML diagnostic report.

Usage:
    model-clinic report checkpoint.pt
    model-clinic report checkpoint.pt --output report.html
    model-clinic report model_name --hf
    model-clinic report checkpoint.pt --runtime
    model-clinic report checkpoint.pt --debug
"""

import argparse
import os
import re
import sys
from datetime import datetime

from model_clinic._loader import load_state_dict, load_model, build_meta
from model_clinic.clinic import diagnose, prescribe, diagnose_runtime
from model_clinic._health_score import compute_health_score
from model_clinic._report import generate_report


def _auto_filename(model_path):
    """Generate report filename from model name + datetime."""
    # Extract a clean name from the model path
    base = os.path.basename(model_path)
    # Remove extensions
    name = re.sub(r'\.(pt|pth|safetensors|bin|ckpt)$', '', base)
    # Clean up characters that are bad in filenames
    name = re.sub(r'[^\w\-.]', '_', name)
    # Truncate long names
    if len(name) > 60:
        name = name[:60]
    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"{name}_{now}.html"


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML model-clinic report"
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path (default: auto-generated from model name + timestamp)")
    parser.add_argument("--hf", action="store_true",
                        help="Load as HuggingFace model")
    parser.add_argument("--runtime", action="store_true",
                        help="Include runtime diagnostics (requires transformers)")
    parser.add_argument("--debug", action="store_true",
                        help="Include debug section with raw detector data for verification")
    args = parser.parse_args()

    # Auto-generate filename if not specified
    output_path = args.output or _auto_filename(args.model)

    print(f"Loading: {args.model}")
    state_dict, raw_meta = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=raw_meta.get("source", "unknown"),
                      extra=raw_meta)

    print("Diagnosing...")
    findings = diagnose(state_dict, meta=raw_meta)

    if args.runtime:
        print("Running runtime diagnostics...")
        model, tokenizer, device = load_model(args.model, hf=args.hf)
        findings.extend(diagnose_runtime(model, tokenizer, device))
        del model

    prescriptions = prescribe(findings)
    health = compute_health_score(findings)

    print(f"Generating report -> {output_path}")
    generate_report(state_dict, findings, prescriptions, health, meta,
                    output_path, debug=args.debug)
    print(f"Done. Open {output_path} in a browser.")


if __name__ == "__main__":
    main()
