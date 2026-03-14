"""Repair modules for model-clinic: deep repair beyond cosmetic treatment."""

from model_clinic._repair.spectral import (
    spectral_denoise,
    spectral_analysis,
    SpectralReport,
)

from model_clinic._repair.graft import (
    graft,
    graft_modules,
    score_parameter,
    GraftManifest,
)

__all__ = [
    "spectral_denoise",
    "spectral_analysis",
    "SpectralReport",
    "graft",
    "graft_modules",
    "score_parameter",
    "GraftManifest",
]

# Optional imports — only available if distill/calibration modules exist
try:
    from model_clinic._repair.distill import (
        identify_dead_modules,
        reset_module_params,
        distill_repair,
        DistillReport,
    )
    from model_clinic._repair.calibration import (
        load_calibration_data,
        generate_random_calibration,
    )
    __all__ += [
        "identify_dead_modules",
        "reset_module_params",
        "distill_repair",
        "DistillReport",
        "load_calibration_data",
        "generate_random_calibration",
    ]
except ImportError:
    pass

from model_clinic._repair.activation import (
    activation_audit,
    activation_repair,
    find_destructive_layers,
    effective_rank,
    token_entropy,
    ActivationReport,
    LayerStats,
)
__all__ += [
    "activation_audit",
    "activation_repair",
    "find_destructive_layers",
    "effective_rank",
    "token_entropy",
    "ActivationReport",
    "LayerStats",
]
