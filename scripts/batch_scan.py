"""Batch scan all checkpoints with model-clinic.

Runs diagnose + health score on every .pt checkpoint found on the server.
Outputs a sorted table showing which models are healthy, which are broken,
and what's wrong with each. Great for validating model-clinic against
known-good and known-bad training runs.

Usage:
    python3 batch_scan.py [--dir ~/] [--output scan_results.json]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import model_clinic as mc


def scan_checkpoint(path):
    """Scan a single checkpoint. Returns dict with results or error."""
    result = {
        "path": str(path),
        "name": Path(path).stem,
        "parent": str(Path(path).parent.name),
    }

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        # Unwrap common checkpoint formats
        if isinstance(data, dict):
            if "model_state_dict" in data:
                sd = data["model_state_dict"]
                result["tag"] = data.get("tag", "")
            elif "state_dict" in data:
                sd = data["state_dict"]
            elif "model" in data:
                sd = data["model"]
            else:
                # Check if it's a raw state dict (has tensor values)
                has_tensors = any(isinstance(v, torch.Tensor) for v in data.values())
                if has_tensors:
                    sd = data
                else:
                    result["error"] = f"Unknown format, keys: {list(data.keys())[:5]}"
                    return result
        elif isinstance(data, torch.Tensor):
            # Single tensor file (steering vectors, etc.)
            result["error"] = "Single tensor, not a model"
            result["shape"] = list(data.shape)
            return result
        else:
            result["error"] = f"Unknown type: {type(data).__name__}"
            return result

        # Basic stats
        n_tensors = len(sd)
        total_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        result["tensors"] = n_tensors
        result["params"] = total_params
        result["params_m"] = round(total_params / 1e6, 1)
        result["size_mb"] = round(os.path.getsize(path) / 1e6, 1)

        # Diagnose
        findings = mc.diagnose(sd)
        score = mc.compute_health_score(findings)

        result["score"] = score.overall
        result["grade"] = score.grade
        result["categories"] = score.categories
        result["findings_count"] = len(findings)
        result["findings"] = []
        for f in findings:
            result["findings"].append({
                "severity": f.severity,
                "condition": f.condition,
                "param": f.param_name,
            })

        # Severity breakdown
        sevs = {}
        for f in findings:
            sevs[f.severity] = sevs.get(f.severity, 0) + 1
        result["severities"] = sevs

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch scan checkpoints with model-clinic")
    parser.add_argument("--dir", default=os.path.expanduser("~/"),
                        help="Root directory to search for .pt files")
    parser.add_argument("--output", default="scan_results.json",
                        help="Output JSON file")
    parser.add_argument("--skip-venv", action="store_true", default=True,
                        help="Skip virtualenv directories")
    args = parser.parse_args()

    # Find all .pt files
    print(f"Scanning {args.dir} for checkpoints...")
    checkpoints = []
    for root, dirs, files in os.walk(args.dir):
        # Skip venvs and site-packages
        if args.skip_venv:
            dirs[:] = [d for d in dirs if d not in ("venv", ".venv", "env", "site-packages",
                                                     "__pycache__", ".git", "node_modules")]
            if "site-packages" in root:
                continue
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                full = os.path.join(root, f)
                # Skip tiny files (< 1KB, probably not models)
                if os.path.getsize(full) > 1024:
                    checkpoints.append(full)

    print(f"Found {len(checkpoints)} checkpoint files\n")

    # Scan each
    results = []
    for i, path in enumerate(sorted(checkpoints)):
        rel = os.path.relpath(path, args.dir)
        print(f"[{i+1}/{len(checkpoints)}] {rel}...", end=" ", flush=True)
        t0 = time.time()
        result = scan_checkpoint(path)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"SKIP ({result['error'][:60]})")
        else:
            grade = result.get("grade", "?")
            score = result.get("score", "?")
            findings = result.get("findings_count", "?")
            params = result.get("params_m", "?")
            print(f"{score}/100 ({grade}) | {findings} findings | {params}M params | {elapsed:.1f}s")

        results.append(result)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    # Print summary table
    scored = [r for r in results if "score" in r]
    scored.sort(key=lambda r: r["score"])

    print(f"\n{'='*80}")
    print(f"  MODEL HEALTH SUMMARY — {len(scored)} models scanned")
    print(f"{'='*80}")
    print(f"  {'Score':>5}  {'Grade':>5}  {'Findings':>8}  {'Params':>8}  Path")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*40}")

    for r in scored:
        rel = os.path.relpath(r["path"], args.dir)
        print(f"  {r['score']:>5}  {r['grade']:>5}  {r['findings_count']:>8}  "
              f"{r['params_m']:>7.1f}M  {rel}")

    # Grade distribution
    grades = {}
    for r in scored:
        grades[r["grade"]] = grades.get(r["grade"], 0) + 1
    print(f"\nGrade distribution: {grades}")

    # Most common findings
    all_conditions = {}
    for r in scored:
        for f in r.get("findings", []):
            key = f["condition"]
            all_conditions[key] = all_conditions.get(key, 0) + 1
    print(f"\nMost common findings:")
    for cond, count in sorted(all_conditions.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}x  {cond}")

    skipped = [r for r in results if "error" in r]
    if skipped:
        print(f"\nSkipped {len(skipped)} files (not model checkpoints)")


if __name__ == "__main__":
    main()
