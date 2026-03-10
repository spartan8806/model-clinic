"""Surgery: Direct parameter modification and testing.

Operations:
  scale    - Scale a neuron/param by a factor
  zero     - Zero out a neuron, head, or parameter
  clamp    - Clamp values to a range
  set      - Set a scalar param to an exact value
  reset    - Reset a param to its init value
  noise    - Add calibrated noise

Usage:
    model-surgery model.pt --hf --op scale --target "layers.23.mlp.down_proj" --neuron 490 --factor 0.1 --save fixed.pt
    model-surgery model.pt --hf --op zero --target "layers.12.self_attn" --head 5 --save fixed.pt
    model-surgery model.pt --hf --interactive
"""

import argparse
import json
import sys

import torch

from model_clinic._loader import load_model
from model_clinic._utils import device_auto, find_param


def _get_num_heads(model):
    """Get number of attention heads from model config."""
    config = model.config
    if hasattr(config, "num_attention_heads"):
        return config.num_attention_heads
    if hasattr(config, "n_head"):
        return config.n_head
    if hasattr(config, "num_heads"):
        return config.num_heads
    raise ValueError("Cannot detect num_attention_heads from model config")


def generate_test(model, tokenizer, device, prompt, max_tokens=80):
    """Quick generation test."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=False)

    resp = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True).strip()
    return resp


def op_scale(model, param, neuron=None, head=None, factor=0.1):
    """Scale a neuron or entire param."""
    with torch.no_grad():
        if neuron is not None:
            if param.dim() >= 2:
                old_val = param[neuron].norm().item()
                param[neuron] *= factor
                new_val = param[neuron].norm().item()
                return f"Scaled neuron {neuron}: norm {old_val:.4f} -> {new_val:.4f}"
            elif param.dim() == 1:
                old_val = param[neuron].item()
                param[neuron] *= factor
                new_val = param[neuron].item()
                return f"Scaled neuron {neuron}: {old_val:.6f} -> {new_val:.6f}"
        elif head is not None and param.dim() >= 2:
            num_heads = _get_num_heads(model)
            total = param.shape[0]
            head_dim = total // num_heads
            start = head * head_dim
            end = start + head_dim
            old_val = param[start:end].norm().item()
            param[start:end] *= factor
            new_val = param[start:end].norm().item()
            return f"Scaled head {head} (rows {start}:{end}): norm {old_val:.4f} -> {new_val:.4f}"
        else:
            old_val = param.norm().item()
            param *= factor
            new_val = param.norm().item()
            return f"Scaled entire param: norm {old_val:.4f} -> {new_val:.4f}"


def op_zero(model, param, neuron=None, head=None):
    """Zero out a neuron, head, or entire param."""
    with torch.no_grad():
        if neuron is not None:
            if param.dim() >= 2:
                param[neuron].zero_()
                return f"Zeroed neuron {neuron} (row in {list(param.shape)})"
            elif param.dim() == 1:
                old = param[neuron].item()
                param[neuron] = 0
                return f"Zeroed neuron {neuron}: {old:.6f} -> 0"
        elif head is not None and param.dim() >= 2:
            num_heads = _get_num_heads(model)
            total = param.shape[0]
            head_dim = total // num_heads
            start = head * head_dim
            end = start + head_dim
            param[start:end].zero_()
            return f"Zeroed head {head} (rows {start}:{end})"
        else:
            param.zero_()
            return f"Zeroed entire param {list(param.shape)}"


def op_clamp(param, min_val=-1.0, max_val=1.0):
    """Clamp all values to a range."""
    with torch.no_grad():
        old_min = param.min().item()
        old_max = param.max().item()
        param.clamp_(min_val, max_val)
        new_min = param.min().item()
        new_max = param.max().item()
        return f"Clamped [{old_min:.4f}, {old_max:.4f}] -> [{new_min:.4f}, {new_max:.4f}]"


def op_set(param, value):
    """Set a scalar param to exact value."""
    with torch.no_grad():
        if param.dim() == 0:
            old = param.item()
            param.fill_(value)
            extra = ""
            if abs(value) < 10:
                extra = f" (sigmoid: {torch.sigmoid(torch.tensor(old)).item():.6f} -> {torch.sigmoid(torch.tensor(value)).item():.6f})"
            return f"Set: {old:.6f} -> {value:.6f}{extra}"
        else:
            old_mean = param.mean().item()
            param.fill_(value)
            return f"Set all to {value} (was mean={old_mean:.6f})"


def op_reset(param, method="kaiming"):
    """Reset param to init values."""
    with torch.no_grad():
        old_norm = param.norm().item()
        if method == "zeros":
            param.zero_()
        elif method == "ones":
            param.fill_(1.0)
        elif method == "kaiming":
            if param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param)
            else:
                param.zero_()
        elif method == "xavier":
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                param.zero_()
        new_norm = param.norm().item()
        return f"Reset ({method}): norm {old_norm:.4f} -> {new_norm:.4f}"


def interactive_mode(model, tokenizer, device):
    """Interactive surgery session."""
    print("\n=== INTERACTIVE SURGERY ===")
    print("Commands:")
    print("  list [filter]        - List parameters")
    print("  info <name>          - Show param stats")
    print("  scale <name> <factor> [neuron=N]")
    print("  zero <name> [neuron=N] [head=N]")
    print("  set <name> <value>")
    print("  clamp <name> <min> <max>")
    print("  reset <name> [method]")
    print("  test <prompt>        - Generate response")
    print("  undo                 - Undo last operation")
    print("  save <path>          - Save modified model")
    print("  quit")
    print()

    history = []

    while True:
        try:
            cmd = input("surgery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()

        if action in ("quit", "q"):
            break

        elif action == "list":
            filt = parts[1] if len(parts) > 1 else ""
            for name, param in model.named_parameters():
                if filt in name:
                    shape = "x".join(str(s) for s in param.shape)
                    print(f"  {name:<60s} {shape:>15s} norm={param.float().norm().item():.4f}")

        elif action == "info":
            if len(parts) < 2:
                print("  Usage: info <name>")
                continue
            matches = find_param(model, parts[1])
            for name, param in matches:
                t = param.float()
                print(f"  {name}")
                print(f"    shape: {list(param.shape)}")
                print(f"    dtype: {param.dtype}")
                print(f"    mean:  {t.mean().item():+.6f}")
                print(f"    std:   {t.std().item():.6f}")
                print(f"    range: [{t.min().item():+.6f}, {t.max().item():+.6f}]")
                print(f"    norm:  {t.norm().item():.6f}")
                if param.dim() == 0:
                    print(f"    value: {t.item():.6f}")
                    print(f"    sigmoid: {torch.sigmoid(t).item():.6f}")

        elif action == "scale":
            try:
                rest = parts[1].split()
                target = rest[0]
                factor = float(rest[1])
                neuron = None
                for r in rest[2:]:
                    if r.startswith("neuron="):
                        neuron = int(r.split("=")[1])
                matches = find_param(model, target)
                for name, param in matches:
                    history.append((name, param.data.clone()))
                    result = op_scale(model, param, neuron=neuron, factor=factor)
                    print(f"  {name}: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif action == "zero":
            try:
                rest = parts[1].split()
                target = rest[0]
                neuron = None
                head = None
                for r in rest[1:]:
                    if r.startswith("neuron="):
                        neuron = int(r.split("=")[1])
                    if r.startswith("head="):
                        head = int(r.split("=")[1])
                matches = find_param(model, target)
                for name, param in matches:
                    history.append((name, param.data.clone()))
                    result = op_zero(model, param, neuron=neuron, head=head)
                    print(f"  {name}: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif action == "set":
            try:
                rest = parts[1].split()
                target = rest[0]
                value = float(rest[1])
                matches = find_param(model, target)
                for name, param in matches:
                    history.append((name, param.data.clone()))
                    result = op_set(param, value)
                    print(f"  {name}: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif action == "clamp":
            try:
                rest = parts[1].split()
                target = rest[0]
                min_val = float(rest[1])
                max_val = float(rest[2])
                matches = find_param(model, target)
                for name, param in matches:
                    history.append((name, param.data.clone()))
                    result = op_clamp(param, min_val, max_val)
                    print(f"  {name}: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif action == "reset":
            try:
                rest = parts[1].split()
                target = rest[0]
                method = rest[1] if len(rest) > 1 else "kaiming"
                matches = find_param(model, target)
                for name, param in matches:
                    history.append((name, param.data.clone()))
                    result = op_reset(param, method)
                    print(f"  {name}: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif action == "test":
            prompt = parts[1] if len(parts) > 1 else "What is the capital of France?"
            print(f"  Q: {prompt}")
            resp = generate_test(model, tokenizer, device, prompt)
            print(f"  A: {resp[:300]}")

        elif action == "undo":
            if history:
                name, old_data = history.pop()
                for n, param in model.named_parameters():
                    if n == name:
                        param.data = old_data
                        print(f"  Undid change to {name}")
                        break
            else:
                print("  Nothing to undo")

        elif action == "save":
            path = parts[1] if len(parts) > 1 else "surgery_output.pt"
            torch.save(model.state_dict(), path)
            print(f"  Saved to {path}")

        else:
            print(f"  Unknown command: {action}")


def main():
    parser = argparse.ArgumentParser(description="Model surgery - modify and test parameters")
    parser.add_argument("model", help="Path to HF model (local or hub name)")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model (default behavior)")
    parser.add_argument("--op", choices=["scale", "zero", "clamp", "set", "reset", "noise"],
                        help="Operation to perform")
    parser.add_argument("--target", type=str, help="Parameter name (or substring)")
    parser.add_argument("--neuron", type=int, default=None, help="Target neuron index")
    parser.add_argument("--head", type=int, default=None, help="Target attention head")
    parser.add_argument("--factor", type=float, default=0.1, help="Scale factor")
    parser.add_argument("--value", type=float, default=None, help="Value for set operation")
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val", help="Clamp min")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val", help="Clamp max")
    parser.add_argument("--method", type=str, default="kaiming", help="Reset method")
    parser.add_argument("--test", type=str, default=None, help="Test prompt (run before and after)")
    parser.add_argument("--save", type=str, default=None, help="Save modified model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    model, tokenizer, device = load_model(args.model, hf=True)
    model.eval()

    if args.interactive:
        interactive_mode(model, tokenizer, device)
        return

    if not args.op or not args.target:
        print("Need --op and --target (or use --interactive)")
        return

    matches = find_param(model, args.target)
    if not matches:
        print(f"No parameter matching '{args.target}'")
        print("Available params containing similar names:")
        for name, _ in model.named_parameters():
            parts = args.target.split(".")
            if any(p in name for p in parts):
                print(f"  {name}")
        return

    if args.test:
        print(f"\nBEFORE:")
        resp = generate_test(model, tokenizer, device, args.test)
        print(f"  Q: {args.test}")
        print(f"  A: {resp[:300]}")

    print(f"\nApplying {args.op} to {len(matches)} param(s):")
    for name, param in matches:
        if args.op == "scale":
            result = op_scale(model, param, neuron=args.neuron, head=args.head, factor=args.factor)
        elif args.op == "zero":
            result = op_zero(model, param, neuron=args.neuron, head=args.head)
        elif args.op == "clamp":
            result = op_clamp(param, args.min_val, args.max_val)
        elif args.op == "set":
            result = op_set(param, args.value)
        elif args.op == "reset":
            result = op_reset(param, args.method)
        else:
            result = "Unknown op"
        print(f"  {name}: {result}")

    if args.test:
        print(f"\nAFTER:")
        resp = generate_test(model, tokenizer, device, args.test)
        print(f"  Q: {args.test}")
        print(f"  A: {resp[:300]}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
