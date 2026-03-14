"""Surgical repair of the ATLES 300M RecurrentTransformer.

Runs Level 1 (cosmetic) + Level 2 (spectral) statically, then Level 3
(distillation) to revive dead memory banks and collapsed embeddings.

Usage:
    python surgical_repair.py [--checkpoint PATH] [--data PATH] [--device cuda]

Expects:
    - model-clinic installed (pip install model-clinic)
    - The 300M model checkpoint (.pt with model_state_dict key)
    - Calibration data (textbook_rho1.jsonl)
    - The model class (copied inline below to avoid import issues)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Minimal model class for distillation ──────────────────────────────────
# We only need forward() to work so we can capture activations.
# This is a stripped-down version of the RecurrentTransformer.

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
        if self.count.item() == 0:
            return torch.zeros_like(query)
        scores = F.cosine_similarity(
            query.unsqueeze(-2), self.keys.unsqueeze(0).unsqueeze(0), dim=-1
        )
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * self.values.unsqueeze(0).unsqueeze(0)).sum(-2)


class InternalMemory(nn.Module):
    def __init__(self, dim, tier_sizes=(256, 512, 1024, 512, 256)):
        super().__init__()
        self.tiers = nn.ModuleList([MemoryBank(s, dim) for s in tier_sizes])
        self.read_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007
        self.write_gate = nn.Parameter(torch.tensor(-5.0))

    def read(self, query):
        gate = torch.sigmoid(self.read_gate)
        result = torch.zeros_like(query)
        for tier in self.tiers:
            result = result + tier.read(query)
        return result * gate

    def forward(self, x):
        return self.read(x)


class RecurrentTransformer(nn.Module):
    """Minimal 300M model for activation capture during distillation."""

    def __init__(self, vocab_size=32000, dim=1024, n_layers=16, n_heads=8, mlp_dim=4096):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.internal_memory = InternalMemory(dim)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        mem = self.internal_memory.read(x)
        x = x + mem
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Infer dimensions from state dict and load."""
        embed_w = state_dict["embed_tokens.weight"]
        vocab_size, dim = embed_w.shape

        # Count layers
        n_layers = 0
        while f"layers.{n_layers}.attn_norm.weight" in state_dict:
            n_layers += 1

        # Infer heads from q_proj shape
        q_w = state_dict["layers.0.attention.q_proj.weight"]
        # Assume head_dim = 128 for 1024-dim model
        n_heads = dim // 128 if dim >= 128 else dim // 64

        # Infer MLP dim
        mlp_dim = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]

        model = cls(vocab_size, dim, n_layers, n_heads, mlp_dim)

        # Load with strict=False to handle extra keys (drives, ACT, etc.)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (expected): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (drives/ACT/etc.): {len(unexpected)}")

        return model


# ── Calibration data loader ───────────────────────────────────────────────

def load_calibration(path, max_samples=200, max_length=256):
    """Load text from JSONL, tokenize with simple byte-level encoding."""
    # We use a simple tokenizer since the actual tokenizer may not be available.
    # For distillation we just need diverse input patterns, not perfect tokenization.
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("text", obj.get("prompt", obj.get("content", "")))
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    # Try to load the real tokenizer if available
    tokenizer = None
    tokenizer_path = Path(path).parent / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path.parent))
            print(f"  Loaded tokenizer from {tokenizer_path.parent}")
        except Exception:
            pass

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            print("  Using Llama-2 tokenizer (fallback)")
        except Exception:
            pass

    if tokenizer is None:
        # Last resort: character-level encoding
        print("  WARNING: No tokenizer available, using character-level encoding")
        batches = []
        for text in texts:
            ids = [min(ord(c), 31999) for c in text[:max_length]]
            if len(ids) >= 16:
                batches.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
        return batches

    batches = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        if len(tokens) >= 16:
            batches.append(torch.tensor(tokens, dtype=torch.long).unsqueeze(0))

    return batches


# ── Main surgical repair pipeline ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Surgical repair of ATLES 300M model")
    parser.add_argument("--checkpoint",
                        default="/home/conner/atles-prototype/checkpoints/merged_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data",
                        default="/home/conner/atles-prototype/textbook_rho1.jsonl",
                        help="Path to calibration data (JSONL)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None,
                        help="Output path for repaired checkpoint (default: auto)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Distillation steps per module group")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for distillation")
    parser.add_argument("--skip-distill", action="store_true",
                        help="Skip Level 3 (distillation), only do static repair")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"\n[1/6] Loading checkpoint: {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = data["model_state_dict"]
    print(f"  Tag: {data.get('tag', 'unknown')}")
    print(f"  Tensors: {len(sd)}")
    total_params = sum(v.numel() for v in sd.values())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # ── Level 1+2: Static repair (cosmetic + spectral) ───────────────────
    print(f"\n[2/6] Diagnosing...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import model_clinic as mc

    findings_before = mc.diagnose(sd)
    score_before = mc.compute_health_score(findings_before)
    print(f"  Health: {score_before.overall}/100 ({score_before.grade})")
    print(f"  Findings: {len(findings_before)}")

    print(f"\n[3/6] Applying Level 1+2 (cosmetic + spectral surgery)...")
    prescriptions = mc.prescribe(findings_before)
    actionable = [p for p in prescriptions if p.action != "advisory"]
    applied_static = 0
    for rx in actionable:
        result = mc.apply_treatment(sd, rx)
        if result.success:
            applied_static += 1
            print(f"  [OK] {rx.action}: {rx.finding.param_name} -> {result.description}")

    findings_mid = mc.diagnose(sd)
    score_mid = mc.compute_health_score(findings_mid)
    print(f"\n  After static repair: {score_mid.overall}/100 ({score_mid.grade})")
    print(f"  Applied: {applied_static}/{len(actionable)}")

    if args.skip_distill:
        print("\n  Skipping Level 3 (--skip-distill)")
        sd_final = sd
    else:
        # ── Level 3: Distillation repair ──────────────────────────────────
        print(f"\n[4/6] Loading calibration data: {args.data}")
        batches = load_calibration(args.data, max_samples=200, max_length=256)
        print(f"  Loaded {len(batches)} calibration samples")

        print(f"\n[5/6] Building model for distillation...")
        model = RecurrentTransformer.from_state_dict(sd)
        model = model.to(device)
        model.eval()

        # We know from analysis exactly which modules are dead
        dead_modules = ["internal_memory"]
        print(f"  Targeting dead modules: {dead_modules}")

        # Capture teacher activations (from working transformer layers)
        print(f"\n[6/6] Running distillation repair ({args.steps} steps, lr={args.lr})...")
        print(f"  Freezing {total_params - sum(p.numel() for n, p in model.named_parameters() if any(n.startswith(dm) for dm in dead_modules)):,} params")
        print(f"  Training {sum(p.numel() for n, p in model.named_parameters() if any(n.startswith(dm) for dm in dead_modules)):,} params")

        # Freeze non-dead params
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(dm) for dm in dead_modules)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  Trainable: {trainable:,} | Frozen: {frozen:,}")

        # Teacher activations: run all calibration data, capture outputs of final norm
        print("  Capturing teacher signal...")
        teacher_outputs = []
        with torch.no_grad():
            for batch in batches[:50]:  # Use 50 samples for teacher
                batch = batch.to(device)
                x = model.embed_tokens(batch)
                # Skip memory read for teacher (it's broken)
                for layer in model.layers:
                    x = layer(x)
                teacher_outputs.append(model.norm(x).cpu())

        # Reset dead modules
        print("  Resetting dead modules...")
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
                print(f"    Reset: {name} {list(param.shape)}")
            elif param.requires_grad and param.dim() == 1:
                nn.init.zeros_(param)

        # Distillation loop
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

        losses = []
        t0 = time.time()
        for step in range(args.steps):
            idx = step % len(teacher_outputs)
            batch = batches[idx].to(device)
            target = teacher_outputs[idx].to(device)

            # Student forward (with memory)
            logits = model(batch)
            # We want the pre-lm_head representation to match teacher
            x = model.embed_tokens(batch)
            mem = model.internal_memory.read(x)
            x = x + mem
            for layer in model.layers:
                x = layer(x)
            student_repr = model.norm(x)

            # MSE loss between student and teacher representations
            loss = F.mse_loss(student_repr, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())
            if step % 50 == 0 or step == args.steps - 1:
                elapsed = time.time() - t0
                print(f"    Step {step:4d}/{args.steps}: loss={loss.item():.6f} "
                      f"({elapsed:.1f}s)")

        print(f"\n  Distillation complete: loss {losses[0]:.6f} -> {losses[-1]:.6f}")

        # Unfreeze all and extract state dict
        for param in model.parameters():
            param.requires_grad = True
        sd_final = model.state_dict()

    # ── Final diagnosis ───────────────────────────────────────────────────
    print(f"\nFinal diagnosis...")
    findings_after = mc.diagnose(sd_final)
    score_after = mc.compute_health_score(findings_after)

    print(f"\n{'='*60}")
    print(f"  BEFORE:       {score_before.overall}/100 ({score_before.grade})")
    print(f"  AFTER L1+L2:  {score_mid.overall}/100 ({score_mid.grade})")
    print(f"  AFTER L3:     {score_after.overall}/100 ({score_after.grade})")
    print(f"{'='*60}")
    for cat in score_before.categories:
        b = score_before.categories[cat]
        m = score_mid.categories.get(cat, b)
        a = score_after.categories.get(cat, b)
        print(f"  {cat:15s} {b:5.0f} -> {m:5.0f} -> {a:5.0f}")
    print(f"  Findings:       {len(findings_before):3d}  -> {len(findings_mid):3d}  -> {len(findings_after):3d}")

    if findings_after:
        print(f"\nRemaining issues:")
        for f in findings_after:
            print(f"  [{f.severity}] {f.condition}: {f.param_name}")

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "merged_final_REPAIRED.pt")

    print(f"\nSaving to: {args.output}")
    torch.save({
        "model_state_dict": sd_final,
        "tag": f"{data.get('tag', 'unknown')}_repaired",
        "repair_log": {
            "score_before": score_before.overall,
            "score_after_static": score_mid.overall,
            "score_after_distill": score_after.overall,
            "static_treatments": applied_static,
            "distill_steps": args.steps if not args.skip_distill else 0,
        }
    }, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
