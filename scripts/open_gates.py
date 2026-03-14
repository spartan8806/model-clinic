"""Phase 2: Open memory gates and train memory to contribute.

The repaired model has clean memory banks (Xavier init from surgical_repair.py)
but gates are at sigmoid(-5) ≈ 0.7% — memory contributes nothing.

This script:
1. Sets read_gate and write_gate to 0 (sigmoid(0) = 50%)
2. Unfreezes memory banks + gates only (5.2M params)
3. Trains with next-token prediction loss
4. The memory must learn to HELP, not hurt — loss must stay equal or improve

If loss degrades, the memory is destructive and the architecture can't learn.
If loss improves, the living memory system works.

Usage:
    python3 open_gates.py [--checkpoint PATH] [--data PATH] [--steps 1000]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model definition (same as surgical_repair.py) ────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MemoryBank(nn.Module):
    def __init__(self, capacity, dim):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(capacity, dim) * 0.01)
        self.values = nn.Parameter(torch.randn(capacity, dim) * 0.01)
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def read(self, query):
        # query: (B, T, D)
        # keys: (C, D) -> (1, 1, C, D)
        scores = F.cosine_similarity(
            query.unsqueeze(-2),  # (B, T, 1, D)
            self.keys.unsqueeze(0).unsqueeze(0),  # (1, 1, C, D)
            dim=-1,
        )  # (B, T, C)
        weights = F.softmax(scores, dim=-1)  # (B, T, C)
        # values: (C, D) -> (1, 1, C, D)
        return (weights.unsqueeze(-1) * self.values.unsqueeze(0).unsqueeze(0)).sum(-2)


class InternalMemory(nn.Module):
    def __init__(self, dim, tier_sizes=(256, 512, 1024, 512, 256)):
        super().__init__()
        self.tiers = nn.ModuleList([MemoryBank(s, dim) for s in tier_sizes])
        self.read_gate = nn.Parameter(torch.tensor(-5.0))
        self.write_gate = nn.Parameter(torch.tensor(-5.0))

    def read(self, query):
        gate = torch.sigmoid(self.read_gate)
        result = torch.zeros_like(query)
        for tier in self.tiers:
            result = result + tier.read(query)
        return result * gate


class RecurrentTransformer(nn.Module):
    def __init__(self, vocab_size=32000, dim=1024, n_layers=16, n_heads=8, mlp_dim=4096):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.internal_memory = InternalMemory(dim)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        mem = self.internal_memory.read(x)
        x = x + mem
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @classmethod
    def from_state_dict(cls, state_dict):
        embed_w = state_dict["embed_tokens.weight"]
        vocab_size, dim = embed_w.shape
        n_layers = 0
        while f"layers.{n_layers}.attn_norm.weight" in state_dict:
            n_layers += 1
        n_heads = dim // 128 if dim >= 128 else dim // 64
        mlp_dim = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]
        model = cls(vocab_size, dim, n_layers, n_heads, mlp_dim)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        return model


# ── Calibration data ──────────────────────────────────────────────────────

def load_training_data(path, max_samples=2000, max_length=256):
    """Load JSONL and tokenize with character-level encoding (no tokenizer needed)."""
    batches = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("text", obj.get("prompt", obj.get("content", "")))
                if not text or len(text) < 32:
                    continue
                # Character-level encoding (simple, works without tokenizer)
                ids = [min(ord(c), 31999) for c in text[:max_length]]
                if len(ids) >= 32:
                    batches.append(torch.tensor(ids, dtype=torch.long))
            except json.JSONDecodeError:
                continue
    return batches


def make_batch(samples, batch_size=4):
    """Collate variable-length samples into padded batches."""
    batch = samples[:batch_size]
    max_len = max(s.size(0) for s in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        input_ids[i, :s.size(0)] = s
        labels[i, :s.size(0)] = s  # Self-supervised: predict same tokens
    return input_ids, labels


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Open memory gates and train memory to contribute")
    parser.add_argument("--checkpoint",
                        default="/home/conner/atles-prototype/checkpoints/merged_final_REPAIRED.pt")
    parser.add_argument("--data",
                        default="/home/conner/atles-prototype/textbook_rho1.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gate-init", type=float, default=0.0,
                        help="Initial gate value. 0=50%%, -2=12%%, 2=88%%")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading checkpoint: {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = data["model_state_dict"]
    print(f"  Tag: {data.get('tag', 'unknown')}")

    model = RecurrentTransformer.from_state_dict(sd)

    # ── Open the gates ────────────────────────────────────────────────────
    print(f"\n[2/5] Opening memory gates...")
    old_read = model.internal_memory.read_gate.item()
    old_write = model.internal_memory.write_gate.item()
    print(f"  read_gate:  {old_read:.4f} (sigmoid={torch.sigmoid(torch.tensor(old_read)).item():.4f})")
    print(f"  write_gate: {old_write:.4f} (sigmoid={torch.sigmoid(torch.tensor(old_write)).item():.4f})")

    with torch.no_grad():
        model.internal_memory.read_gate.fill_(args.gate_init)
        model.internal_memory.write_gate.fill_(args.gate_init)

    new_read = model.internal_memory.read_gate.item()
    print(f"  -> read_gate:  {new_read:.4f} (sigmoid={torch.sigmoid(torch.tensor(new_read)).item():.4f})")
    print(f"  -> write_gate: {new_read:.4f} (sigmoid={torch.sigmoid(torch.tensor(new_read)).item():.4f})")

    # ── Freeze everything except memory ───────────────────────────────────
    print(f"\n[3/5] Freezing non-memory params...")
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if "internal_memory" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Trainable (memory): {trainable_count:,} params ({trainable_count/1e6:.1f}M)")
    print(f"  Frozen (transformer): {frozen_count:,} params ({frozen_count/1e6:.1f}M)")

    model = model.to(device)
    model.train()

    # ── Measure baseline loss (before training) ───────────────────────────
    print(f"\n[4/5] Loading training data: {args.data}")
    all_samples = load_training_data(args.data, max_samples=2000, max_length=256)
    print(f"  Loaded {len(all_samples)} samples")

    # Baseline loss with gates open but memory untrained
    print("\n  Measuring baseline loss (gates open, memory untrained)...")
    model.eval()
    baseline_losses = []
    with torch.no_grad():
        for i in range(0, min(20, len(all_samples)), args.batch_size):
            batch_samples = all_samples[i:i + args.batch_size]
            input_ids, labels = make_batch(batch_samples, args.batch_size)
            input_ids, labels = input_ids.to(device), labels.to(device)
            _, loss = model(input_ids, labels=labels)
            baseline_losses.append(loss.item())
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    baseline_ppl = math.exp(min(baseline_loss, 20))
    print(f"  Baseline loss: {baseline_loss:.4f} (PPL: {baseline_ppl:.1f})")

    # Also measure with gates closed (the "no memory" reference)
    with torch.no_grad():
        model.internal_memory.read_gate.fill_(-5.0)
    nomem_losses = []
    with torch.no_grad():
        for i in range(0, min(20, len(all_samples)), args.batch_size):
            batch_samples = all_samples[i:i + args.batch_size]
            input_ids, labels = make_batch(batch_samples, args.batch_size)
            input_ids, labels = input_ids.to(device), labels.to(device)
            _, loss = model(input_ids, labels=labels)
            nomem_losses.append(loss.item())
    nomem_loss = sum(nomem_losses) / len(nomem_losses)
    nomem_ppl = math.exp(min(nomem_loss, 20))
    print(f"  No-memory loss: {nomem_loss:.4f} (PPL: {nomem_ppl:.1f})")

    # Restore gate
    with torch.no_grad():
        model.internal_memory.read_gate.fill_(args.gate_init)

    print(f"\n  Gap to close: {baseline_loss - nomem_loss:+.4f}")
    print(f"  (negative = memory is currently hurting, training must fix this)")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training memory ({args.steps} steps, lr={args.lr})...")
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)

    best_loss = float("inf")
    best_state = None
    losses_log = []
    gate_log = []
    t0 = time.time()

    for step in range(args.steps):
        # Random batch
        idx = (step * args.batch_size) % len(all_samples)
        batch_samples = all_samples[idx:idx + args.batch_size]
        if len(batch_samples) < 2:
            batch_samples = all_samples[:args.batch_size]
        input_ids, labels = make_batch(batch_samples, args.batch_size)
        input_ids, labels = input_ids.to(device), labels.to(device)

        _, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        losses_log.append(loss.item())
        read_g = torch.sigmoid(model.internal_memory.read_gate).item()
        gate_log.append(read_g)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if step % 100 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            avg_loss = sum(losses_log[-50:]) / len(losses_log[-50:])
            ppl = math.exp(min(avg_loss, 20))
            print(f"  Step {step:5d}/{args.steps}: "
                  f"loss={loss.item():.4f} avg={avg_loss:.4f} PPL={ppl:.1f} "
                  f"read_gate={read_g:.4f} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    final_loss = sum(losses_log[-50:]) / len(losses_log[-50:])
    final_ppl = math.exp(min(final_loss, 20))
    final_gate = torch.sigmoid(model.internal_memory.read_gate).item()

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  No-memory baseline:    loss={nomem_loss:.4f} PPL={nomem_ppl:.1f}")
    print(f"  Open-gate baseline:    loss={baseline_loss:.4f} PPL={baseline_ppl:.1f}")
    print(f"  After training:        loss={final_loss:.4f} PPL={final_ppl:.1f}")
    print(f"  Best loss seen:        {best_loss:.4f}")
    print(f"  Final read_gate:       {final_gate:.4f} ({final_gate*100:.1f}%)")
    print(f"  Training time:         {elapsed:.1f}s")

    if final_loss < nomem_loss:
        improvement = nomem_loss - final_loss
        print(f"\n  MEMORY IS HELPING! Loss improved by {improvement:.4f}")
        print(f"  The living memory architecture CAN learn.")
    elif final_loss < baseline_loss:
        improvement = baseline_loss - final_loss
        print(f"\n  Memory adapted but not better than no-memory.")
        print(f"  Reduced damage by {improvement:.4f} but still net negative.")
    else:
        print(f"\n  Memory is not helping. Architecture may not be viable for this model.")

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "merged_final_GATES_OPEN.pt")

    # Save best state
    print(f"\nSaving best checkpoint to: {args.output}")
    torch.save({
        "model_state_dict": best_state if best_state else model.state_dict(),
        "tag": f"{data.get('tag', 'unknown')}_gates_open",
        "training_log": {
            "nomem_loss": nomem_loss,
            "baseline_loss": baseline_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "final_read_gate": final_gate,
            "steps": args.steps,
            "lr": args.lr,
            "gate_init": args.gate_init,
        },
    }, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
