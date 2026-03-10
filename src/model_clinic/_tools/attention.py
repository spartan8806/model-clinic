"""Attention: Attention pattern analysis per head per layer.

Usage:
    model-attention "What is the capital of France?"
    model-attention "The cat sat on the mat" --layer 0 --head 3
    model-attention "Who are you?" --all-layers
"""

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_clinic._utils import device_auto


def capture_attention(model, tokenizer, text, device, target_layer=None, target_head=None):
    """Run text through model and capture attention weights."""
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
    tokens = [tokenizer.decode(t).strip() for t in input_ids[0]]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
        )

    attention_data = []
    for layer_idx, attn in enumerate(outputs.attentions):
        if target_layer is not None and layer_idx != target_layer:
            continue

        attn = attn[0]
        num_heads = attn.size(0)
        seq_len = attn.size(1)

        layer_data = {
            "layer": layer_idx,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "heads": [],
        }

        for head_idx in range(num_heads):
            if target_head is not None and head_idx != target_head:
                continue

            head_attn = attn[head_idx].float()

            entropy = -(head_attn * head_attn.clamp(min=1e-10).log()).sum(dim=-1)
            avg_entropy = entropy.mean().item()
            max_entropy = torch.log(torch.tensor(float(seq_len))).item()

            focus = 1.0 - (avg_entropy / max_entropy) if max_entropy > 0 else 0

            last_attn = head_attn[-1]
            top_vals, top_ids = last_attn.topk(min(5, seq_len))

            top_attended = []
            for j in range(len(top_ids)):
                pos = top_ids[j].item()
                top_attended.append({
                    "position": pos,
                    "token": tokens[pos] if pos < len(tokens) else "?",
                    "weight": top_vals[j].item(),
                })

            diag_weight = head_attn.diag().mean().item()
            bos_weight = head_attn[:, 0].mean().item()

            if seq_len > 5:
                recent_weight = head_attn[:, -5:].sum(dim=-1).mean().item()
            else:
                recent_weight = 1.0

            head_data = {
                "head": head_idx,
                "avg_entropy": round(avg_entropy, 4),
                "focus_score": round(focus, 4),
                "self_attn_ratio": round(diag_weight, 4),
                "bos_attn": round(bos_weight, 4),
                "recent_attn": round(recent_weight, 4),
                "last_token_attends_to": top_attended,
            }

            layer_data["heads"].append(head_data)

        attention_data.append(layer_data)

    return attention_data, tokens


def classify_head(head_data):
    """Classify attention head type."""
    if head_data["bos_attn"] > 0.5:
        return "BOS-sink"
    if head_data["self_attn_ratio"] > 0.3:
        return "self-attn"
    if head_data["recent_attn"] > 0.8:
        return "recent"
    if head_data["focus_score"] > 0.7:
        return "focused"
    if head_data["focus_score"] < 0.2:
        return "diffuse"
    return "mixed"


def print_attention_results(attention_data, tokens, verbose=False):
    """Pretty print attention analysis."""
    print(f"\nTokens ({len(tokens)}): {' '.join(repr(t) for t in tokens[-15:])}")

    for layer_data in attention_data:
        layer = layer_data["layer"]
        print(f"\n{'='*80}")
        print(f"Layer {layer} ({layer_data['num_heads']} heads, seq_len={layer_data['seq_len']})")
        print(f"{'='*80}")
        print(f"  {'Head':>4s} {'Entropy':>8s} {'Focus':>7s} {'Type':<10s} {'Self':>6s} {'BOS':>6s} {'Last token attends to'}")
        print(f"  {'-'*4} {'-'*8} {'-'*7} {'-'*10} {'-'*6} {'-'*6} {'-'*40}")

        for hd in layer_data["heads"]:
            head_type = classify_head(hd)
            top_str = ", ".join(
                f"{repr(t['token'])}({t['weight']:.2f})" for t in hd["last_token_attends_to"][:3]
            )
            print(
                f"  {hd['head']:>4d}"
                f" {hd['avg_entropy']:>8.3f}"
                f" {hd['focus_score']:>7.3f}"
                f" {head_type:<10s}"
                f" {hd['self_attn_ratio']:>6.3f}"
                f" {hd['bos_attn']:>6.3f}"
                f" {top_str}"
            )

    if len(attention_data) > 3:
        print(f"\n{'='*80}")
        print(f"HEAD TYPE SUMMARY")
        print(f"{'='*80}")
        type_counts = defaultdict(int)
        for ld in attention_data:
            for hd in ld["heads"]:
                type_counts[classify_head(hd)] += 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<12s}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Attention pattern analysis")
    parser.add_argument("prompt", help="Text to analyze")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model")
    parser.add_argument("--layer", type=int, default=None, help="Target specific layer")
    parser.add_argument("--head", type=int, default=None, help="Target specific head")
    parser.add_argument("--all-layers", action="store_true", help="Show all layers")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export", type=str, default=None, help="Export to JSON")
    args = parser.parse_args()

    device = device_auto()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Prompt: {repr(args.prompt)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    target_layer = args.layer
    if not args.all_layers and target_layer is None:
        num_layers = len(model.model.layers)
        target_layers = [0, num_layers // 2, num_layers - 1]
        all_data = []
        for tl in target_layers:
            data, tokens = capture_attention(model, tokenizer, args.prompt, device,
                                            target_layer=tl, target_head=args.head)
            all_data.extend(data)
        print_attention_results(all_data, tokens, verbose=args.verbose)

        if args.export:
            with open(args.export, "w") as f:
                json.dump({"prompt": args.prompt, "model": args.model, "layers": all_data}, f, indent=2, default=str)
            print(f"\nExported to {args.export}")
        return

    data, tokens = capture_attention(model, tokenizer, args.prompt, device,
                                    target_layer=target_layer, target_head=args.head)
    print_attention_results(data, tokens, verbose=args.verbose)

    if args.export:
        with open(args.export, "w") as f:
            json.dump({"prompt": args.prompt, "model": args.model, "layers": data}, f, indent=2, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
