"""Fail-closed output authorization for reserved-test captures."""

from __future__ import annotations

from pathlib import Path

from audit_evaluation_protocol import audit, load_json, reserved_seeds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/research/final_evaluation_protocol.json"


def validate_output_authorization(
    *,
    seed: int,
    output_dir: Path,
    final_evaluation_authorized: bool,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> Path:
    """Return the permitted output root or reject the requested write."""
    if not final_evaluation_authorized:
        allowed = (ROOT / "outputs/live_pipeline").resolve()
        if not output_dir.is_relative_to(allowed):
            raise ValueError(f"Output directory must be under {allowed}: {output_dir}")
        return allowed

    protocol = load_json(protocol_path.resolve())
    preflight = audit(protocol)
    if preflight["status"] != "passed":
        raise RuntimeError(f"Reserved-test authorization failed: {preflight}")
    if seed not in reserved_seeds(protocol):
        raise ValueError(f"Final-evaluation seed is not reserved: {seed}")

    allowed = (ROOT / protocol["final_output_root"]).resolve()
    if not output_dir.is_relative_to(allowed):
        raise ValueError(f"Final output directory must be under {allowed}: {output_dir}")
    return allowed
