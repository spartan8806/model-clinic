"""model-clinic: Diagnose, treat, and understand neural network models."""

__version__ = "0.2.0"

from model_clinic._types import Finding, Prescription, TreatmentResult, ExamReport, ModelMeta
from model_clinic._loader import load_state_dict, load_model, build_meta, save_state_dict
from model_clinic._eval import generate, eval_coherence, eval_perplexity, eval_logit_entropy, eval_diversity
from model_clinic.clinic import diagnose, prescribe, apply_treatment, rollback_treatment

__all__ = [
    # Types
    "Finding",
    "Prescription",
    "TreatmentResult",
    "ExamReport",
    "ModelMeta",
    # Loader
    "load_state_dict",
    "load_model",
    "build_meta",
    "save_state_dict",
    # Clinic
    "diagnose",
    "prescribe",
    "apply_treatment",
    "rollback_treatment",
    # Eval
    "generate",
    "eval_coherence",
    "eval_perplexity",
    "eval_logit_entropy",
    "eval_diversity",
]
