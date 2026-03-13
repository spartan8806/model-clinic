"""Validate: Verify a checkpoint loads and infers correctly.

Usage:
    model-clinic validate checkpoint.pt
    model-clinic validate checkpoint.pt --hf
    model-clinic validate model_name --hf
    model-clinic validate checkpoint.pt --generate --hf
    model-clinic validate checkpoint.pt --json
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

from model_clinic._loader import load_state_dict, load_model


def _format_size(nbytes):
    """Format byte count as human-readable string."""
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024 ** 2:.2f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


def _dtype_bytes(dtype):
    """Bytes per element for a given dtype."""
    return {
        torch.float16: 2, torch.bfloat16: 2,
        torch.float32: 4, torch.float64: 8,
        torch.int8: 1, torch.uint8: 1,
        torch.int16: 2, torch.int32: 4, torch.int64: 8,
        torch.bool: 1,
    }.get(dtype, 4)


def _check_load(path, hf):
    """Load checkpoint, return (state_dict, meta, result_dict)."""
    result = {"check": "load", "status": "PASS", "details": {}}
    try:
        sd, meta = load_state_dict(path, hf=hf)
    except Exception as e:
        result["status"] = "FAIL"
        result["message"] = str(e)
        return None, None, result

    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    num_tensors = len(tensors)
    num_params = sum(t.numel() for t in tensors.values())
    total_bytes = sum(t.numel() * _dtype_bytes(t.dtype) for t in tensors.values())

    result["details"] = {
        "tensors": num_tensors,
        "parameters": num_params,
        "size_bytes": total_bytes,
        "size_human": _format_size(total_bytes),
    }
    result["message"] = (
        f"{num_tensors} tensors, "
        f"{_format_param_count(num_params)} parameters "
        f"({_format_size(total_bytes)})"
    )
    return sd, meta, result


def _format_param_count(n):
    """Format parameter count: 494123456 -> '494M'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _check_integrity(sd):
    """Check for NaN/Inf in any tensor."""
    result = {"check": "integrity", "status": "PASS", "details": {}}
    bad_tensors = []

    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0 or inf_count > 0:
            entry = {"name": name}
            if nan_count > 0:
                entry["nan_count"] = nan_count
            if inf_count > 0:
                entry["inf_count"] = inf_count
            bad_tensors.append(entry)

    if bad_tensors:
        result["status"] = "FAIL"
        result["details"]["bad_tensors"] = bad_tensors
        result["message"] = f"{len(bad_tensors)} tensors contain NaN/Inf"
    else:
        result["message"] = "all tensors finite"

    return result


def _check_shapes(sd):
    """Check for zero-dim issues or empty tensors."""
    result = {"check": "shapes", "status": "PASS", "details": {}}
    bad_shapes = []

    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = tensor.shape
        if tensor.numel() == 0:
            bad_shapes.append({"name": name, "shape": list(shape), "issue": "empty tensor"})
        elif any(s == 0 for s in shape):
            bad_shapes.append({"name": name, "shape": list(shape), "issue": "zero dimension"})

    if bad_shapes:
        result["status"] = "FAIL"
        result["details"]["bad_shapes"] = bad_shapes
        result["message"] = f"{len(bad_shapes)} tensors have invalid shapes"
    else:
        result["message"] = "all valid"

    return result


def _check_dtypes(sd):
    """Report dtype distribution."""
    result = {"check": "dtypes", "status": "INFO", "details": {}}
    dtype_counts = Counter()

    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        dtype_counts[str(tensor.dtype)] += 1

    result["details"]["distribution"] = dict(dtype_counts)
    parts = [f"{dtype} ({count} tensors)" for dtype, count in sorted(dtype_counts.items())]
    result["message"] = ", ".join(parts)

    return result


def _check_generate(path, hf):
    """Load as HF model, generate a short response, verify output."""
    result = {"check": "generate", "status": "PASS", "details": {}}

    if not hf:
        result["status"] = "SKIP"
        result["message"] = "requires --hf flag"
        return result

    try:
        from model_clinic._utils import device_auto
        model, tokenizer, device = load_model(path, hf=True)
    except Exception as e:
        result["status"] = "FAIL"
        result["message"] = f"failed to load model: {e}"
        return result

    prompt = "The quick brown fox"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        num_tokens = len(generated_ids)
    except Exception as e:
        result["status"] = "FAIL"
        result["message"] = f"generation failed: {e}"
        return result
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["details"]["num_tokens"] = num_tokens
    result["details"]["text_preview"] = generated_text[:100]

    if num_tokens == 0 or not generated_text:
        result["status"] = "FAIL"
        result["message"] = "empty output (0 tokens generated)"
        return result

    # Check for collapsed output (all same token)
    unique_ids = len(set(generated_ids.tolist()))
    if num_tokens > 5 and unique_ids == 1:
        result["status"] = "FAIL"
        result["message"] = f"collapsed output ({num_tokens} tokens, 1 unique)"
        return result

    result["message"] = f"coherent output ({num_tokens} tokens)"
    return result


def _print_result(r):
    """Print a single check result to stdout."""
    tag = r["status"]
    label = r["check"].capitalize()
    msg = r.get("message", "")

    if tag == "PASS":
        print(f"  [PASS] {label}: {msg}")
    elif tag == "FAIL":
        print(f"  [FAIL] {label}: {msg}")
        # Print detail lines for failures
        if "bad_tensors" in r.get("details", {}):
            for bt in r["details"]["bad_tensors"]:
                parts = []
                if bt.get("nan_count"):
                    parts.append(f"{bt['nan_count']} NaN values")
                if bt.get("inf_count"):
                    parts.append(f"{bt['inf_count']} Inf values")
                print(f"    - {bt['name']}: {', '.join(parts)}")
        if "bad_shapes" in r.get("details", {}):
            for bs in r["details"]["bad_shapes"]:
                print(f"    - {bs['name']}: shape={bs['shape']} ({bs['issue']})")
    elif tag == "INFO":
        print(f"  [INFO] {label}: {msg}")
    elif tag == "SKIP":
        print(f"  [SKIP] {label}: {msg}")


def validate(path, hf=False, generate=False):
    """Run all validation checks and return list of results.

    Args:
        path: Path to checkpoint or HF model name.
        hf: Whether to treat path as a HuggingFace model.
        generate: Whether to run the generation check.

    Returns:
        List of result dicts, each with keys: check, status, message, details.
    """
    results = []

    # 1. Load
    sd, meta, load_result = _check_load(path, hf)
    results.append(load_result)
    if load_result["status"] == "FAIL":
        return results

    # 2. Integrity
    results.append(_check_integrity(sd))

    # 3. Shapes
    results.append(_check_shapes(sd))

    # 4. Dtypes
    results.append(_check_dtypes(sd))

    # 5. Generate (optional)
    if generate:
        results.append(_check_generate(path, hf))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate a checkpoint loads and infers correctly"
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument(
        "--generate", action="store_true",
        help="Run generation check (requires --hf)"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()

    if args.generate and not args.hf:
        print("Error: --generate requires --hf", file=sys.stderr)
        sys.exit(1)

    results = validate(args.model, hf=args.hf, generate=args.generate)

    if args.json:
        failures = sum(1 for r in results if r["status"] == "FAIL")
        output = {
            "model": args.model,
            "valid": failures == 0,
            "checks": results,
        }
        print(json.dumps(output, indent=2, default=str))
        sys.exit(0 if failures == 0 else 1)

    # Human-readable output
    print(f"Validating: {args.model}")
    print()

    for r in results:
        _print_result(r)

    failures = sum(1 for r in results if r["status"] == "FAIL")
    print()
    if failures == 0:
        print("RESULT: VALID")
    else:
        print(f"RESULT: INVALID ({failures} check{'s' if failures != 1 else ''} failed)")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
