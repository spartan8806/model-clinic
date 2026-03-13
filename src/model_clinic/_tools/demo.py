"""Demo command -- generate a synthetic broken model and examine it."""

import argparse
import os
import sys
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic demo",
        description="Generate and examine a synthetic broken model for demonstration",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="everything-broken",
        help="Preset name (default: everything-broken). Use --list to see all.",
    )
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument(
        "--save", type=str, default=None, help="Save the synthetic model to a file"
    )
    parser.add_argument(
        "--treat", action="store_true", help="Also run treatment after diagnosis"
    )
    parser.add_argument(
        "--report", type=str, default=None, help="Generate HTML report to this path"
    )
    args = parser.parse_args()

    from model_clinic._synthetic import SYNTHETIC_MODELS

    if args.list:
        print("Available synthetic model presets:")
        for name in sorted(SYNTHETIC_MODELS.keys()):
            print(f"  {name}")
        return

    if args.preset not in SYNTHETIC_MODELS:
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        print(
            f"Available: {', '.join(sorted(SYNTHETIC_MODELS.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate
    print(f"Generating synthetic model: {args.preset}")
    factory = SYNTHETIC_MODELS[args.preset]
    state_dict = factory()

    # Save to temp file or specified path
    save_path = args.save
    cleanup_temp = False
    if not save_path:
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        save_path = tmp.name
        tmp.close()
        cleanup_temp = True

    torch.save(state_dict, save_path)
    print(f"Saved to {save_path}")
    param_count = sum(
        t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)
    )
    print(f"  {len(state_dict)} tensors, {param_count:,} parameters")

    # Run exam
    print()
    from model_clinic._health_score import compute_health_score, print_health_score
    from model_clinic._loader import build_meta
    from model_clinic.clinic import diagnose, prescribe, print_exam

    meta = build_meta(state_dict, source="synthetic")
    findings = diagnose(state_dict)
    prescriptions = prescribe(findings)

    print_exam(findings, prescriptions, explain=True)

    health = compute_health_score(findings)
    print_health_score(health)

    errors = sum(1 for f in findings if f.severity == "ERROR")
    warns = sum(1 for f in findings if f.severity == "WARN")
    print(f"{'=' * 80}")
    if errors:
        print(f"VERDICT: UNHEALTHY ({errors} errors, {warns} warnings)")
    elif warns:
        print(f"VERDICT: OK with {warns} warning(s)")
    else:
        print("VERDICT: HEALTHY")
    print(f"{'=' * 80}")

    # Optional: treat
    if args.treat:
        from model_clinic.clinic import apply_treatment

        print(f"\n{'=' * 80}")
        print("APPLYING TREATMENTS")
        print(f"{'=' * 80}")
        for i, rx in enumerate(prescriptions):
            result = apply_treatment(state_dict, rx)
            status = "OK" if result.success else "FAIL"
            print(f"  [{status}] Rx #{i + 1} {rx.name}: {result.description}")

        # Re-diagnose
        after_findings = diagnose(state_dict)
        after_health = compute_health_score(after_findings)
        print(
            f"\nAfter treatment: {after_health.overall}/100 {after_health.grade} "
            f"(was {health.overall}/100 {health.grade})"
        )

    # Optional: HTML report
    if args.report:
        from model_clinic._report import generate_report

        generate_report(state_dict, findings, prescriptions, health, meta, args.report)
        print(f"\nHTML report saved to {args.report}")

    # Cleanup temp file if we created one
    if cleanup_temp:
        try:
            os.unlink(save_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
