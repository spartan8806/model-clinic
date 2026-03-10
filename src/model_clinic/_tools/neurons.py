"""Neurons: Profile neuron activations across prompts.

Usage:
    model-neurons --prompts "What is 2+2?" "Who made you?"
    model-neurons --file prompts.txt --top 50
    model-neurons --layer 12 --export neurons.json
"""

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_clinic._utils import device_auto


def capture_activations(model, tokenizer, prompts, device, target_layer=None):
    """Run prompts and capture MLP activations at each layer."""
    hooks = []
    captured = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captured[name] = out.detach().float()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if target_layer is not None and i != target_layer:
            continue
        hook = layer.mlp.register_forward_hook(make_hook(f"mlp.{i}"))
        hooks.append(hook)
        hook2 = layer.self_attn.register_forward_hook(make_hook(f"attn.{i}"))
        hooks.append(hook2)

    model.eval()
    results = {
        "per_prompt": [],
        "per_layer": defaultdict(lambda: {
            "activation_means": [],
            "activation_maxes": [],
            "dead_counts": [],
            "neuron_variances": [],
        }),
    }

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        captured.clear()

        with torch.no_grad():
            model(input_ids=input_ids, output_hidden_states=True)

        prompt_data = {"prompt": prompt, "layers": {}}

        for name, act in sorted(captured.items()):
            act_2d = act.squeeze(0)

            neuron_mean = act_2d.mean(dim=0)
            neuron_max = act_2d.abs().max(dim=0).values
            neuron_std = act_2d.std(dim=0) if act_2d.size(0) > 1 else torch.zeros_like(neuron_mean)

            dead_mask = neuron_max < 1e-6
            dead_count = dead_mask.sum().item()
            total_neurons = neuron_mean.size(0)

            prompt_data["layers"][name] = {
                "shape": list(act.shape),
                "mean": act_2d.mean().item(),
                "std": act_2d.std().item(),
                "max": act_2d.abs().max().item(),
                "dead_neurons": dead_count,
                "total_neurons": total_neurons,
                "dead_pct": dead_count / total_neurons if total_neurons > 0 else 0,
                "top_neurons": [],
            }

            top_indices = neuron_max.argsort(descending=True)[:20]
            for idx in top_indices:
                i = idx.item()
                prompt_data["layers"][name]["top_neurons"].append({
                    "index": i,
                    "max_activation": neuron_max[i].item(),
                    "mean_activation": neuron_mean[i].item(),
                    "std": neuron_std[i].item(),
                })

            results["per_layer"][name]["activation_means"].append(neuron_mean)
            results["per_layer"][name]["activation_maxes"].append(neuron_max)
            results["per_layer"][name]["dead_counts"].append(dead_count)

        results["per_prompt"].append(prompt_data)

    for h in hooks:
        h.remove()

    cross_prompt = {}
    for name, data in results["per_layer"].items():
        if len(data["activation_means"]) < 2:
            continue
        stacked = torch.stack(data["activation_means"])
        cross_var = stacked.var(dim=0)
        cross_mean = stacked.mean(dim=0)

        top_var_indices = cross_var.argsort(descending=True)[:30]

        cross_prompt[name] = {
            "avg_dead": sum(data["dead_counts"]) / len(data["dead_counts"]),
            "top_variable_neurons": [],
        }
        for idx in top_var_indices:
            i = idx.item()
            cross_prompt[name]["top_variable_neurons"].append({
                "index": i,
                "variance": cross_var[i].item(),
                "mean": cross_mean[i].item(),
            })

    results["cross_prompt"] = cross_prompt
    return results


def print_results(results, verbose=False):
    """Pretty print neuron analysis."""
    print(f"\n{'='*80}")
    print(f"NEURON ANALYSIS - {len(results['per_prompt'])} prompts")
    print(f"{'='*80}")

    for pd in results["per_prompt"]:
        print(f"\n  Prompt: {repr(pd['prompt'])}")
        for name, ld in sorted(pd["layers"].items()):
            dead_str = f"dead={ld['dead_neurons']}/{ld['total_neurons']}({ld['dead_pct']:.1%})"
            print(f"    {name:<12s} mean={ld['mean']:+.4f}  std={ld['std']:.4f}"
                  f"  max={ld['max']:.4f}  {dead_str}")

            if verbose and ld["top_neurons"]:
                top3 = ld["top_neurons"][:5]
                top_str = ", ".join(f"n{t['index']}={t['max_activation']:.3f}" for t in top3)
                print(f"                 top: {top_str}")

    if results.get("cross_prompt"):
        print(f"\n{'='*80}")
        print(f"DISCRIMINATIVE NEURONS (high variance across prompts)")
        print(f"{'='*80}")

        for name, cpd in sorted(results["cross_prompt"].items()):
            print(f"\n  {name}: avg_dead={cpd['avg_dead']:.0f}")
            if cpd["top_variable_neurons"]:
                print(f"    {'Neuron':>8s} {'Variance':>12s} {'Mean':>12s}")
                for n in cpd["top_variable_neurons"][:10]:
                    print(f"    n{n['index']:<7d} {n['variance']:>12.6f} {n['mean']:>+12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Neuron analyzer")
    parser.add_argument("--prompts", nargs="+", default=None, help="Prompts to analyze")
    parser.add_argument("--file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model")
    parser.add_argument("--layer", type=int, default=None, help="Target specific layer")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--export", type=str, default=None, help="Export to JSON")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Who are you?",
            "Explain quantum entanglement.",
            "The sky is blue because",
            "Write a poem about cats.",
        ]

    device = device_auto()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    if args.layer is not None:
        print(f"Target layer: {args.layer}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    results = capture_activations(model, tokenizer, prompts, device, target_layer=args.layer)
    print_results(results, verbose=args.verbose)

    if args.export:
        export = {
            "model": args.model,
            "prompts": prompts,
            "per_prompt": results["per_prompt"],
            "cross_prompt": {
                k: {
                    "avg_dead": v["avg_dead"],
                    "top_variable_neurons": v["top_variable_neurons"],
                }
                for k, v in results.get("cross_prompt", {}).items()
            },
        }
        with open(args.export, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
