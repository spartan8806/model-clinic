"""Shared data types for model-clinic."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    """A diagnosed issue in a model."""
    condition: str        # e.g. "dead_neurons", "generation_collapse"
    severity: str         # ERROR, WARN, INFO
    param_name: str       # which parameter (or "model" for runtime findings)
    details: dict = field(default_factory=dict)

    def __str__(self):
        return f"[{self.severity}] {self.condition}: {self.param_name}"


@dataclass
class Prescription:
    """A treatment for a finding."""
    name: str
    description: str
    risk: str             # low, medium, high
    finding: Finding
    action: str           # treatment action identifier
    params: dict = field(default_factory=dict)


@dataclass
class TreatmentResult:
    """Result of applying a prescription."""
    prescription: Prescription
    success: bool
    description: str
    backup: Any = None    # cloned tensor for rollback


@dataclass
class ModelMeta:
    """Metadata about a loaded model."""
    source: str = "unknown"        # "huggingface", "checkpoint", "safetensors"
    num_params: int = 0
    num_tensors: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    dtype: str = "float32"
    extra: dict = field(default_factory=dict)


@dataclass
class ExamReport:
    """Full examination report."""
    model_path: str
    meta: ModelMeta
    findings: list = field(default_factory=list)
    prescriptions: list = field(default_factory=list)
    treatments: list = field(default_factory=list)
    before_score: float = None
    before_ppl: float = None
    after_score: float = None
    after_ppl: float = None

    def to_dict(self):
        return {
            "model": self.model_path,
            "meta": {
                "source": self.meta.source,
                "num_params": self.meta.num_params,
                "num_tensors": self.meta.num_tensors,
                "hidden_size": self.meta.hidden_size,
                "num_layers": self.meta.num_layers,
            },
            "findings": [
                {"condition": f.condition, "severity": f.severity,
                 "param": f.param_name, "details": f.details}
                for f in self.findings
            ],
            "prescriptions": [
                {"name": rx.name, "risk": rx.risk, "action": rx.action,
                 "description": rx.description, "param": rx.finding.param_name}
                for rx in self.prescriptions
            ],
            "treatments": [
                {"name": t.prescription.name, "success": t.success,
                 "description": t.description}
                for t in self.treatments
            ],
            "before_score": self.before_score,
            "before_ppl": self.before_ppl,
            "after_score": self.after_score,
            "after_ppl": self.after_ppl,
        }
