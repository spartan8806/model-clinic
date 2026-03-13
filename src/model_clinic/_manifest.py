"""Treatment Manifest — audit log for applied treatments.

Records what treatments were applied to a model, including checksums
of affected tensors before and after modification.

Usage:
    from model_clinic._manifest import TreatmentManifest

    manifest = TreatmentManifest()
    manifest.record(treatment_result, state_dict)

    manifest.save("treatment_manifest.json")
    manifest.print_summary()
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import torch

from model_clinic import __version__
from model_clinic._types import TreatmentResult


def _tensor_checksum(tensor):
    """Compute first 8 chars of sha256 of tensor bytes."""
    data = tensor.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:8]


class TreatmentManifest:
    """Audit log of treatments applied to a model."""

    def __init__(self):
        self.treatments = []
        self._start_time = datetime.now(timezone.utc)

    def record(self, result, state_dict):
        """Record a treatment result with before/after checksums.

        Args:
            result: TreatmentResult from apply_treatment().
            state_dict: Current state dict (after treatment was applied).
        """
        rx = result.prescription
        target = rx.finding.param_name

        # Compute checksums
        checksum_before = None
        checksum_after = None

        if result.backup is not None and isinstance(result.backup, torch.Tensor):
            checksum_before = _tensor_checksum(result.backup)

        if target in state_dict and isinstance(state_dict[target], torch.Tensor):
            checksum_after = _tensor_checksum(state_dict[target])

        entry = {
            "prescription": rx.name,
            "action": rx.action,
            "target": target,
            "risk": rx.risk,
            "success": result.success,
            "description": result.description,
            "checksum_before": checksum_before,
            "checksum_after": checksum_after,
        }
        self.treatments.append(entry)

    def to_dict(self):
        """Serialize manifest to dict."""
        succeeded = [t for t in self.treatments if t["success"]]
        failed = [t for t in self.treatments if not t["success"]]

        risk_breakdown = {}
        for t in succeeded:
            risk_breakdown[t["risk"]] = risk_breakdown.get(t["risk"], 0) + 1

        return {
            "timestamp": self._start_time.isoformat(),
            "model_clinic_version": __version__,
            "treatments": self.treatments,
            "summary": {
                "total_applied": len(succeeded),
                "total_failed": len(failed),
                "risk_breakdown": risk_breakdown,
            },
        }

    def save(self, path):
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def print_summary(self, file=None):
        """Print a human-readable summary."""
        out = file or sys.stdout
        data = self.to_dict()
        summary = data["summary"]

        print(f"\nTreatment Manifest", file=out)
        print(f"{'-' * 45}", file=out)
        print(f"  Version:  {data['model_clinic_version']}", file=out)
        print(f"  Time:     {data['timestamp']}", file=out)
        print(f"  Applied:  {summary['total_applied']}", file=out)
        print(f"  Failed:   {summary['total_failed']}", file=out)

        if summary["risk_breakdown"]:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(summary["risk_breakdown"].items()))
            print(f"  Risk:     {parts}", file=out)

        print(file=out)

        for i, t in enumerate(self.treatments):
            status = "OK" if t["success"] else "FAIL"
            print(f"  [{status}] {t['prescription']} -> {t['target']}", file=out)
            if t["checksum_before"] and t["checksum_after"]:
                print(f"         {t['checksum_before']} -> {t['checksum_after']}", file=out)

        print(file=out)
