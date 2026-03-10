"""Ablate: Systematically disable parts and measure impact.

Usage:
    model-ablate --model Qwen/Qwen2.5-0.5B-Instruct --hf --heads --layer 12
    model-ablate --model Qwen/Qwen2.5-0.5B-Instruct --hf --neurons --layer 23 --top 20
    model-ablate --model Qwen/Qwen2.5-0.5B-Instruct --hf --layers
"""

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_clinic._utils import device_auto


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who are you?",
    "Explain how gravity works.",
    "What are the three states of matter?",
]


def load_hf(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32
    ).to(device)
    return model, tokenizer


def eval_generation(model, tokenizer, device, prompts, max_tokens=60):
    """Quick eval - returns avg response quality score."""
    model.eval()
    results = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=False)

        resp = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True).strip()
        word_count = len(resp.split())

        has_rep = any(
            resp.count(resp[j:j+20]) > 2
            for j in range(0, min(len(resp), 100), 20)
            if len(resp[j:j+20]) == 20
        )

        coherent = word_count >= 3 and not has_rep
        results.append({
            "prompt": prompt,
            "response": resp[:150],
            "coherent": coherent,
            "word_count": word_count,
        })

    coherent_count = sum(1 for r in results if r["coherent"])
    return coherent_count, len(prompts), results


def eval_perplexity(model, tokenizer, device, prompts):
    """Quick perplexity on the prompts."""
    model.eval()
    total_loss = 0
    count = 0

    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "The answer is well known."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, return_tensors="pt").to(device)

        if encoded["input_ids"].size(1) < 5:
            continue

        with torch.no_grad():
            outputs = model(input_ids=encoded["input_ids"], labels=encoded["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return round(ppl, 2)


def ablate_heads(model, tokenizer, device, layer_idx, prompts):
    """Ablate each attention head in a layer."""
    layer = model.model.layers[layer_idx]
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    print(f"  Baseline...")
    base_score, total, _ = eval_generation(model, tokenizer, device, prompts)
    base_ppl = eval_perplexity(model, tokenizer, device, prompts)
    print(f"  Baseline: {base_score}/{total} coherent, PPL={base_ppl}")

    results = []

    for head_idx in range(num_heads):
        start = head_idx * head_dim
        end = start + head_dim

        saved = {}
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            saved[proj_name] = proj.weight.data[start:end].clone()
            if proj.bias is not None:
                saved[f"{proj_name}_bias"] = proj.bias.data[start:end].clone()

        o_proj = layer.self_attn.o_proj
        saved["o_proj"] = o_proj.weight.data[:, start:end].clone()

        with torch.no_grad():
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(layer.self_attn, proj_name)
                proj.weight.data[start:end] = 0
                if proj.bias is not None:
                    proj.bias.data[start:end] = 0
            o_proj.weight.data[:, start:end] = 0

        score, _, _ = eval_generation(model, tokenizer, device, prompts)
        ppl = eval_perplexity(model, tokenizer, device, prompts)
        impact = base_score - score
        ppl_change = ppl - base_ppl

        results.append({
            "head": head_idx,
            "score": score,
            "score_impact": impact,
            "ppl": ppl,
            "ppl_change": ppl_change,
        })

        with torch.no_grad():
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(layer.self_attn, proj_name)
                proj.weight.data[start:end] = saved[proj_name]
                if f"{proj_name}_bias" in saved:
                    proj.bias.data[start:end] = saved[f"{proj_name}_bias"]
            o_proj.weight.data[:, start:end] = saved["o_proj"]

        tag = "CRITICAL" if impact > 0 else ("SAFE" if impact == 0 else "HELPFUL?")
        print(f"    Head {head_idx:2d}: {score}/{total} ({tag:>8s}) PPL={ppl:.1f} ({ppl_change:+.1f})")

    return results, base_score, base_ppl


def ablate_neurons(model, tokenizer, device, layer_idx, prompts, top_n=20):
    """Ablate top-firing neurons in MLP."""
    layer = model.model.layers[layer_idx]

    activations = []
    hook_data = {}

    def hook_fn(module, input, output):
        hook_data["act"] = output.detach().float()

    hook = layer.mlp.register_forward_hook(hook_fn)

    model.eval()
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        activations.append(hook_data["act"].squeeze(0))

    hook.remove()

    all_acts = torch.cat(activations, dim=0)
    neuron_max = all_acts.abs().max(dim=0).values
    top_indices = neuron_max.argsort(descending=True)[:top_n]

    print(f"  Baseline...")
    base_score, total, _ = eval_generation(model, tokenizer, device, prompts)
    base_ppl = eval_perplexity(model, tokenizer, device, prompts)
    print(f"  Baseline: {base_score}/{total} coherent, PPL={base_ppl}")
    print(f"  Testing top {top_n} neurons by activation magnitude...")

    results = []
    down_proj = layer.mlp.down_proj

    for rank, neuron_idx in enumerate(top_indices):
        n = neuron_idx.item()
        max_act = neuron_max[n].item()

        saved_col = down_proj.weight.data[:, n].clone()

        with torch.no_grad():
            down_proj.weight.data[:, n] = 0

        score, _, _ = eval_generation(model, tokenizer, device, prompts)
        ppl = eval_perplexity(model, tokenizer, device, prompts)
        impact = base_score - score
        ppl_change = ppl - base_ppl

        results.append({
            "rank": rank,
            "neuron": n,
            "max_activation": max_act,
            "score": score,
            "score_impact": impact,
            "ppl": ppl,
            "ppl_change": ppl_change,
        })

        with torch.no_grad():
            down_proj.weight.data[:, n] = saved_col

        tag = "CRITICAL" if impact > 0 else ("SAFE" if impact == 0 else "HELPFUL?")
        print(f"    n{n:<5d} (max_act={max_act:>8.2f}): {score}/{total} ({tag:>8s}) PPL={ppl:.1f} ({ppl_change:+.1f})")

    return results, base_score, base_ppl


def ablate_layers(model, tokenizer, device, prompts):
    """Ablate entire layers."""
    num_layers = len(model.model.layers)

    print(f"  Baseline...")
    base_score, total, _ = eval_generation(model, tokenizer, device, prompts)
    base_ppl = eval_perplexity(model, tokenizer, device, prompts)
    print(f"  Baseline: {base_score}/{total} coherent, PPL={base_ppl}")

    results = []

    for layer_idx in range(num_layers):
        skip_active = [True]

        def skip_hook(module, input, output):
            if skip_active[0]:
                if isinstance(output, tuple):
                    return (input[0],) + output[1:]
                return input[0]
            return output

        hook = model.model.layers[layer_idx].register_forward_hook(skip_hook)

        score, _, _ = eval_generation(model, tokenizer, device, prompts)
        ppl = eval_perplexity(model, tokenizer, device, prompts)
        impact = base_score - score
        ppl_change = ppl - base_ppl

        hook.remove()

        results.append({
            "layer": layer_idx,
            "score": score,
            "score_impact": impact,
            "ppl": ppl,
            "ppl_change": ppl_change,
        })

        tag = "CRITICAL" if impact > 0 else ("SAFE" if impact == 0 else "??")
        print(f"    Layer {layer_idx:2d}: {score}/{total} ({tag:>8s}) PPL={ppl:.1f} ({ppl_change:+.1f})")

    return results, base_score, base_ppl


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model")
    parser.add_argument("--hf", action="store_true", help="Load as HF model")
    parser.add_argument("--heads", action="store_true", help="Ablate attention heads")
    parser.add_argument("--neurons", action="store_true", help="Ablate MLP neurons")
    parser.add_argument("--layers", action="store_true", help="Ablate entire layers")
    parser.add_argument("--layer", type=int, default=23, help="Target layer for heads/neurons")
    parser.add_argument("--top", type=int, default=20, help="Top N neurons to test")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompts")
    parser.add_argument("--export", type=str, default=None, help="Export to JSON")
    args = parser.parse_args()

    device = device_auto()
    prompts = args.prompts or DEFAULT_PROMPTS

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")

    model, tokenizer = load_hf(args.model, device)
    model.eval()

    all_results = {}

    if args.heads:
        print(f"\n{'='*60}")
        print(f"ABLATING ATTENTION HEADS - Layer {args.layer}")
        print(f"{'='*60}")
        results, base_score, base_ppl = ablate_heads(
            model, tokenizer, device, args.layer, prompts
        )
        all_results["heads"] = {
            "layer": args.layer,
            "baseline_score": base_score,
            "baseline_ppl": base_ppl,
            "results": results,
        }

    if args.neurons:
        print(f"\n{'='*60}")
        print(f"ABLATING MLP NEURONS - Layer {args.layer}")
        print(f"{'='*60}")
        results, base_score, base_ppl = ablate_neurons(
            model, tokenizer, device, args.layer, prompts, top_n=args.top
        )
        all_results["neurons"] = {
            "layer": args.layer,
            "baseline_score": base_score,
            "baseline_ppl": base_ppl,
            "results": results,
        }

    if args.layers:
        print(f"\n{'='*60}")
        print(f"ABLATING ENTIRE LAYERS")
        print(f"{'='*60}")
        results, base_score, base_ppl = ablate_layers(
            model, tokenizer, device, prompts
        )
        all_results["layers"] = {
            "baseline_score": base_score,
            "baseline_ppl": base_ppl,
            "results": results,
        }

    if not any([args.heads, args.neurons, args.layers]):
        print("Specify --heads, --neurons, or --layers")
        return

    if args.export:
        with open(args.export, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
