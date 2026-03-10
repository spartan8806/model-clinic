"""Logit Lens: Watch predictions form layer by layer.

Usage:
    model-logit-lens "What is the capital of France?"
    model-logit-lens "What is 2+2?" --top 5
    model-logit-lens "The sky is" --last-token --all-layers
"""

import argparse
import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_clinic._utils import device_auto


def get_layer_outputs(model, input_ids):
    """Run model and capture hidden states at every layer."""
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states, outputs.logits


def logit_lens_analysis(model, tokenizer, text, top_k=10, last_token_only=False, show_all=False):
    """Project each layer's hidden state through lm_head."""
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    with torch.no_grad():
        hidden_states, final_logits = get_layer_outputs(model, input_ids)

    lm_head = model.lm_head
    final_norm = model.model.norm

    num_layers = len(hidden_states) - 1
    seq_len = input_ids.size(1)

    if last_token_only:
        positions = [seq_len - 1]
    else:
        positions = list(range(max(0, seq_len - 3), seq_len))

    results = []

    if show_all:
        layer_indices = list(range(num_layers + 1))
    else:
        layer_indices = [0]
        layer_indices.extend(range(4, num_layers - 3, 4))
        layer_indices.extend(range(max(num_layers - 3, 1), num_layers + 1))
        layer_indices = sorted(set(layer_indices))

    for pos in positions:
        pos_result = {
            "position": pos,
            "token": tokens[pos].strip(),
            "layers": [],
        }

        for layer_idx in layer_indices:
            h = hidden_states[layer_idx][:, pos, :]

            h_normed = final_norm(h)
            logits = lm_head(h_normed)

            probs = F.softmax(logits.float(), dim=-1)
            top_probs, top_ids = probs.topk(top_k, dim=-1)

            top_tokens = []
            for j in range(top_k):
                tok = tokenizer.decode(top_ids[0, j])
                prob = top_probs[0, j].item()
                top_tokens.append({"token": tok.strip(), "prob": prob})

            entropy = -(probs * probs.clamp(min=1e-10).log()).sum(-1).item()

            layer_label = "embed" if layer_idx == 0 else f"L{layer_idx - 1}"
            pos_result["layers"].append({
                "layer": layer_label,
                "layer_idx": layer_idx,
                "entropy": entropy,
                "top": top_tokens,
            })

        results.append(pos_result)

    return results, tokens, num_layers


def print_lens_results(results, tokens, num_layers, top_k):
    """Pretty print the logit lens results."""
    tok_strs = [repr(t.strip()) for t in tokens[-10:]]
    print(f"\nTokens ({len(tokens)}): {' '.join(tok_strs)}")
    print(f"Layers: {num_layers}")

    for pos_result in results:
        pos = pos_result["position"]
        tok = pos_result["token"]
        print(f"\n{'='*80}")
        print(f"Position {pos}: {repr(tok)}")
        print(f"{'='*80}")
        print(f"{'Layer':<8} {'Entropy':>8} | {'Top predictions'}")
        print(f"{'-'*8} {'-'*8}-+-{'-'*60}")

        for layer_data in pos_result["layers"]:
            layer = layer_data["layer"]
            entropy = layer_data["entropy"]
            top = layer_data["top"]

            def safe(s):
                return s.encode("ascii", errors="replace").decode("ascii")
            top_str = "  ".join(
                f"{safe(repr(t['token']))}({t['prob']:.1%})" for t in top[:5]
            )

            print(f"{layer:<8} {entropy:>7.2f} | {top_str}")


def main():
    parser = argparse.ArgumentParser(description="Logit lens - see predictions form layer by layer")
    parser.add_argument("prompt", help="Text prompt to analyze")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model")
    parser.add_argument("--top", type=int, default=5, help="Top-k predictions per layer")
    parser.add_argument("--last-token", action="store_true", help="Only analyze last token")
    parser.add_argument("--all-layers", action="store_true", help="Show every layer")
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
        output_hidden_states=True,
    ).to(device)
    model.eval()

    results, tokens, num_layers = logit_lens_analysis(
        model, tokenizer, args.prompt,
        top_k=args.top,
        last_token_only=args.last_token,
        show_all=args.all_layers,
    )

    print_lens_results(results, tokens, num_layers, args.top)

    if args.export:
        with open(args.export, "w") as f:
            json.dump({"prompt": args.prompt, "model": args.model, "results": results}, f, indent=2)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
