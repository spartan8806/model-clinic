"""model-clinic: Diagnose, treat, and understand neural network models."""

__version__ = "0.3.0"

from model_clinic._types import (
    Finding, Prescription, TreatmentResult, ExamReport, ModelMeta, HealthScore,
    ExamResult, PipelineResult, MonitorAlert, MonitorSummary,
)
from model_clinic._health_score import compute_health_score, print_health_score
from model_clinic._monitor import ClinicMonitor
from model_clinic._hf_callback import ClinicTrainerCallback
from model_clinic._integrations import (
    WandbCallback,
    MLflowCallback,
    TensorBoardCallback,
    log_health_to_wandb,
    log_health_to_mlflow,
)
from model_clinic._loader import load_state_dict, load_model, build_meta, save_state_dict
from model_clinic._eval import generate, eval_coherence, eval_perplexity, eval_logit_entropy, eval_diversity
from model_clinic._manifest import TreatmentManifest
from model_clinic.clinic import (
    diagnose, prescribe, apply_treatment, rollback_treatment,
    examine_batch, create_pipeline, TreatmentPipeline,
    causal_rank,
)
from model_clinic._synthetic import SYNTHETIC_MODELS, make_healthy_mlp, make_everything_broken
from model_clinic._plugins import load_plugins, list_plugins
from model_clinic._tools.prune_suggest import prune_suggestions
from model_clinic._tools.autopsy import autopsy
from model_clinic._mri import model_mri, mri_summary, LayerMRI
from model_clinic._profiles import ArchProfile, get_profile, auto_detect_profile, list_profiles
from model_clinic._badge import (
    generate_badge_svg,
    generate_badge_url,
    generate_model_card_snippet,
    save_badge_svg,
)

__all__ = [
    # Types
    "Finding",
    "Prescription",
    "TreatmentResult",
    "ExamReport",
    "ExamResult",
    "PipelineResult",
    "ModelMeta",
    "HealthScore",
    "MonitorAlert",
    "MonitorSummary",
    # Monitor
    "ClinicMonitor",
    "ClinicTrainerCallback",
    # Integrations
    "WandbCallback",
    "MLflowCallback",
    "TensorBoardCallback",
    "log_health_to_wandb",
    "log_health_to_mlflow",
    # Manifest
    "TreatmentManifest",
    # Health score
    "compute_health_score",
    "print_health_score",
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
    "examine_batch",
    "create_pipeline",
    "TreatmentPipeline",
    "causal_rank",
    # Eval
    "generate",
    "eval_coherence",
    "eval_perplexity",
    "eval_logit_entropy",
    "eval_diversity",
    # Synthetic models
    "SYNTHETIC_MODELS",
    "make_healthy_mlp",
    "make_everything_broken",
    # Profiles
    "ArchProfile",
    "get_profile",
    "auto_detect_profile",
    "list_profiles",
    # Badge generation
    "generate_badge_svg",
    "generate_badge_url",
    "generate_model_card_snippet",
    "save_badge_svg",
    # Plugins
    "load_plugins",
    "list_plugins",
    # MRI
    "model_mri",
    "mri_summary",
    "LayerMRI",
    # Autopsy and pruning
    "autopsy",
    "prune_suggestions",
]
