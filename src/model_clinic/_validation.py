"""Treatment validation — before/after metric comparison.

Turns "applied 3 fixes" into evidence: shows what the treatment actually did
to the model's health score (always), and to perplexity + coherence (when a
runnable model is available via --test).

The health-score delta is pure static analysis — it works on every checkpoint,
no GPU or tokenizer required. PPL/coherence require a loadable model.
"""

from dataclasses import dataclass, field
from typing import Optional

from model_clinic._types import HealthScore


# ── ANSI ──────────────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

def _pick_arrows():
    """Use unicode arrows when the console can encode them, else ASCII."""
    import sys
    enc = (getattr(sys.stdout, "encoding", None) or "ascii")
    try:
        "▲▼".encode(enc)
        return "▲", "▼", "="
    except (UnicodeEncodeError, LookupError):
        return "^", "v", "="


_UP, _DOWN, _FLAT = _pick_arrows()


@dataclass
class ValidationReport:
    """Before/after comparison of a treatment's effect.

    health_* are always present (static). ppl_* / coherence_* are present only
    when the treatment was run with --test against a loadable model.
    """
    n_applied: int = 0
    n_total: int = 0

    health_before: Optional[HealthScore] = None
    health_after: Optional[HealthScore] = None

    ppl_before: Optional[float] = None
    ppl_after: Optional[float] = None

    # coherence stored as (coherent_count, total)
    coherence_before: Optional[tuple] = None
    coherence_after: Optional[tuple] = None

    rolled_back: bool = False
    rollback_reason: str = ""

    # ── derived metrics ────────────────────────────────────────────────────
    @property
    def health_delta(self) -> Optional[int]:
        if self.health_before is None or self.health_after is None:
            return None
        return self.health_after.overall - self.health_before.overall

    @property
    def ppl_factor(self) -> Optional[float]:
        """How many times better PPL got. >1 means improvement."""
        if self.ppl_before is None or self.ppl_after is None:
            return None
        if self.ppl_after <= 0:
            return None
        return self.ppl_before / self.ppl_after

    @property
    def coherence_ratio_before(self) -> Optional[float]:
        if not self.coherence_before or self.coherence_before[1] == 0:
            return None
        return self.coherence_before[0] / self.coherence_before[1]

    @property
    def coherence_ratio_after(self) -> Optional[float]:
        if not self.coherence_after or self.coherence_after[1] == 0:
            return None
        return self.coherence_after[0] / self.coherence_after[1]

    def verdict(self) -> str:
        """One of: IMPROVED, REGRESSED, NEUTRAL, INCONCLUSIVE.

        Decision precedence: runtime evidence (PPL/coherence) outranks the
        static health score when available, because it measures real behaviour.
        """
        if self.rolled_back:
            return "ROLLED BACK"

        # Runtime signals first (strongest evidence)
        runtime_signals = []
        if self.coherence_ratio_before is not None and self.coherence_ratio_after is not None:
            if self.coherence_ratio_after > self.coherence_ratio_before:
                runtime_signals.append(1)
            elif self.coherence_ratio_after < self.coherence_ratio_before:
                runtime_signals.append(-1)
            else:
                runtime_signals.append(0)
        if self.ppl_factor is not None:
            if self.ppl_factor > 1.02:
                runtime_signals.append(1)
            elif self.ppl_factor < 0.98:
                runtime_signals.append(-1)
            else:
                runtime_signals.append(0)

        if runtime_signals:
            if any(s < 0 for s in runtime_signals):
                return "REGRESSED"
            if any(s > 0 for s in runtime_signals):
                return "IMPROVED"
            # runtime measured but flat — fall through to health
        # Static health
        d = self.health_delta
        if d is None:
            return "INCONCLUSIVE"
        if d > 0:
            return "IMPROVED"
        if d < 0:
            return "REGRESSED"
        return "NEUTRAL"

    def to_dict(self) -> dict:
        def hs(h):
            return None if h is None else {"overall": h.overall, "grade": h.grade}
        return {
            "applied": self.n_applied,
            "total": self.n_total,
            "health_before": hs(self.health_before),
            "health_after": hs(self.health_after),
            "health_delta": self.health_delta,
            "ppl_before": self.ppl_before,
            "ppl_after": self.ppl_after,
            "ppl_factor": round(self.ppl_factor, 2) if self.ppl_factor else None,
            "coherence_before": self.coherence_ratio_before,
            "coherence_after": self.coherence_ratio_after,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "verdict": self.verdict(),
        }


# ── rendering ───────────────────────────────────────────────────────────────

def _arrow(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-9:
        return f"{_DIM}{_FLAT}{_RESET}"
    good = (delta > 0) == higher_is_better
    color = _GREEN if good else _RED
    sym = _UP if delta > 0 else _DOWN
    return f"{color}{sym}{_RESET}"


def _row(label, before, after, change, arrow):
    return f"  {label:<12s}  {before:<12s}  {after:<12s}  {change} {arrow}"


def _emit(s, out):
    """Print a line, falling back to ASCII if the console can't encode it."""
    try:
        print(s, file=out)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"), file=out)


def print_validation_report(report: ValidationReport, file=None):
    """Render the before/after validation table."""
    import sys
    out = file or sys.stdout

    print = lambda s="", file=out: _emit(s, file)  # noqa: E731

    print(f"\n{'='*80}", file=out)
    print(f"{_BOLD}TREATMENT VALIDATION{_RESET}", file=out)
    print(f"{'='*80}", file=out)
    print(f"  Applied {report.n_applied}/{report.n_total} fixes.\n", file=out)

    print(f"  {_BOLD}{'Metric':<12s}  {'Before':<12s}  {'After':<12s}  {'Change':<16s}{_RESET}", file=out)
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*16}", file=out)

    # Health (always present)
    if report.health_before is not None and report.health_after is not None:
        hb, ha = report.health_before, report.health_after
        d = report.health_delta
        before_s = f"{hb.overall}/{hb.grade}"
        after_s = f"{ha.overall}/{ha.grade}"
        change_s = f"{d:+d}"
        print(_row("Health", before_s, after_s, change_s, _arrow(d, higher_is_better=True)), file=out)

    # PPL (only with --test)
    if report.ppl_before is not None and report.ppl_after is not None:
        factor = report.ppl_factor
        before_s = f"{report.ppl_before:.1f}"
        after_s = f"{report.ppl_after:.1f}"
        if factor and factor >= 1.0:
            change_s = f"{factor:.1f}x better"
        elif factor:
            change_s = f"{1/factor:.1f}x worse"
        else:
            change_s = "—"
        delta = report.ppl_before - report.ppl_after  # positive = improved
        print(_row("PPL", before_s, after_s, change_s, _arrow(delta, higher_is_better=True)), file=out)

    # Coherence (only with --test)
    cb, ca = report.coherence_ratio_before, report.coherence_ratio_after
    if cb is not None and ca is not None:
        before_s = f"{cb:.2f}"
        after_s = f"{ca:.2f}"
        change_s = f"{ca - cb:+.2f}"
        print(_row("Coherence", before_s, after_s, change_s, _arrow(ca - cb, higher_is_better=True)), file=out)

    # Verdict
    verdict = report.verdict()
    vcolor = {
        "IMPROVED": _GREEN, "REGRESSED": _RED, "ROLLED BACK": _RED,
        "NEUTRAL": _DIM, "INCONCLUSIVE": _DIM,
    }.get(verdict, "")
    print(file=out)
    if report.rolled_back:
        print(f"  {vcolor}{_BOLD}VERDICT: {verdict}{_RESET} - {report.rollback_reason}", file=out)
    else:
        msg = {
            "IMPROVED": "treatment helped on measured metrics.",
            "REGRESSED": "treatment hurt at least one metric.",
            "NEUTRAL": "no change in health score.",
            "INCONCLUSIVE": "not enough signal to judge.",
        }.get(verdict, "")
        print(f"  {vcolor}{_BOLD}VERDICT: {verdict}{_RESET} - {msg}", file=out)
    print(f"{'='*80}", file=out)
