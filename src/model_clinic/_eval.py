"""Shared evaluation functions for model-clinic.

Unified generation testing, perplexity, coherence scoring.
Used by clinic, ablate, surgery, and runtime modules.
"""

import torch
import torch.nn.functional as F

from model_clinic._utils import safe_str


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who are you?",
    "Explain how gravity works.",
    "What are the three states of matter?",
]


def generate(model, tokenizer, device, prompt, max_tokens=60):
    """Generate a response from a prompt. Returns response string."""
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=max_tokens, do_sample=False
        )

    return tokenizer.decode(
        out[0][input_ids.size(1):], skip_special_tokens=True
    ).strip()


def _check_repetition(text, n_sizes=(3, 4, 5)):
    """Check for n-gram repetition. Returns max repetition ratio across n sizes."""
    words = text.split()
    if len(words) < 5:
        return 0.0

    max_ratio = 0.0
    for n in n_sizes:
        if len(words) < n:
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        unique = len(set(ngrams))
        ratio = 1.0 - (unique / len(ngrams))
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def eval_coherence(model, tokenizer, device, prompts=None, max_tokens=60):
    """Test generation coherence across prompts.

    Returns:
        (coherent_count, total, details_list)
    """
    prompts = prompts or DEFAULT_PROMPTS
    model.eval()
    coherent = 0
    results = []

    for prompt in prompts:
        resp = generate(model, tokenizer, device, prompt, max_tokens)
        words = len(resp.split())
        rep_ratio = _check_repetition(resp)

        ok = words >= 3 and rep_ratio < 0.5
        if ok:
            coherent += 1

        results.append({
            "prompt": prompt,
            "response": resp[:200],
            "coherent": ok,
            "word_count": words,
            "repetition": round(rep_ratio, 3),
        })

    return coherent, len(prompts), results


def eval_perplexity(model, tokenizer, device, prompts=None):
    """Compute average perplexity across prompts.

    Returns:
        float: perplexity value
    """
    prompts = prompts or DEFAULT_PROMPTS
    model.eval()
    total_loss = 0
    count = 0

    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "The answer is well known."},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = prompt + " The answer is well known."

        encoded = tokenizer(text, return_tensors="pt").to(device)
        if encoded["input_ids"].size(1) < 5:
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                labels=encoded["input_ids"],
            )
            total_loss += outputs.loss.item()
            count += 1

    avg = total_loss / max(count, 1)
    return round(torch.exp(torch.tensor(avg)).item(), 2)


def eval_logit_entropy(model, tokenizer, device, prompts=None):
    """Check logit distribution health across prompts.

    Returns dict with:
        avg_entropy: average entropy across all last-token predictions
        avg_top1_prob: average top-1 probability
        vocab_used: number of unique top-1 tokens across prompts
        collapsed: True if model appears to have generation collapse
    """
    prompts = prompts or DEFAULT_PROMPTS
    model.eval()

    entropies = []
    top1_probs = []
    top1_tokens = set()

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            last_logits = logits[0, -1].float()

            probs = F.softmax(last_logits, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
            top1_prob = probs.max().item()
            top1_id = probs.argmax().item()

            entropies.append(entropy)
            top1_probs.append(top1_prob)
            top1_tokens.add(top1_id)

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0
    avg_top1 = sum(top1_probs) / len(top1_probs) if top1_probs else 1.0

    return {
        "avg_entropy": round(avg_entropy, 3),
        "avg_top1_prob": round(avg_top1, 4),
        "vocab_used": len(top1_tokens),
        "total_prompts": len(prompts),
        "collapsed": avg_top1 > 0.95 or len(top1_tokens) <= 1,
    }


def eval_diversity(model, tokenizer, device, prompts=None, max_tokens=60):
    """Measure generation diversity across prompts.

    Returns dict with distinct-N scores and response similarity.
    """
    prompts = prompts or DEFAULT_PROMPTS
    model.eval()

    all_tokens = []
    responses = []

    for prompt in prompts:
        resp = generate(model, tokenizer, device, prompt, max_tokens)
        responses.append(resp)
        words = resp.split()
        all_tokens.extend(words)

    if not all_tokens:
        return {"distinct_1": 0, "distinct_2": 0, "distinct_3": 0, "all_same": True}

    # Distinct-N: unique n-grams / total n-grams
    def distinct_n(tokens, n):
        if len(tokens) < n:
            return 0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0

    # Check if all responses are identical
    all_same = len(set(responses)) <= 1

    return {
        "distinct_1": round(distinct_n(all_tokens, 1), 3),
        "distinct_2": round(distinct_n(all_tokens, 2), 3),
        "distinct_3": round(distinct_n(all_tokens, 3), 3),
        "all_same": all_same,
        "unique_responses": len(set(responses)),
    }
