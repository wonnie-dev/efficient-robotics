"""Capture a small GPU-5-only validation batch for objective occlusion labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from run_live_single_gpu_pipeline import ROOT, write_json_atomic
from run_single_gpu_pilot import configured_physical_gpu, require_single_gpu_policy


ISAAC_PYTHON = Path("/data/wonheekoh/isaacsim_venv/bin/python")
VALIDATION_VARIANT_PATTERN = (
    "inside_clear",
    "outside",
    "rim_occluded",
    "covered_unknown",
    "behind_ambiguous",
    "inside_clear",
    "outside",
    "rim_occluded",
    "covered_unknown",
)
RESERVED_TEST_SEEDS = frozenset(range(200, 210))
VIEWS = ("center", "close_high", "right")
ROW_FIELDS = (
    "seed",
    "variant",
    "view_id",
    "valid",
    "visible_target_pixels",
    "amodal_target_pixels",
    "visible_fraction_of_amodal",
    "occlusion_fraction",
    "severity",
    "fully_hidden",
    "reference_valid",
    "reference_revealed_target_pixels",
    "reference_occlusion_fraction",
    "reference_severity",
)


def validation_scenes(
    seed_start: int,
    variants: tuple[str, ...] = VALIDATION_VARIANT_PATTERN,
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (seed_start + offset, variant)
        for offset, variant in enumerate(variants)
    )


def single_gpu_environment() -> dict[str, str]:
    physical_gpu = configured_physical_gpu()
    if physical_gpu != 5:
        raise RuntimeError(
            f"Objective occlusion validation requires physical GPU 5, got {physical_gpu}"
        )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "5",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "5",
            "PHYSICAL_GPU": "5",
        }
    )
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    return environment


def capture_complete(capture_root: Path) -> bool:
    result_path = capture_root / "smoke_result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        result.get("status") == "completed"
        and result.get("gpu_policy", {}).get("physical_gpu") == 5
        and all(
            (
                capture_root
                / "observations"
                / view_id
                / filename
            ).is_file()
            for view_id in VIEWS
            for filename in (
                "objective_occlusion.json",
                "objective_reference_occlusion.json",
            )
        )
    )


def run_capture(
    output_root: Path,
    seed: int,
    variant: str,
    environment: dict[str, str],
) -> dict:
    seed_root = output_root / variant / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    run_indices = []
    for candidate in sorted(seed_root.glob("run*")):
        try:
            run_indices.append(int(candidate.name.removeprefix("run")))
        except ValueError:
            continue
        if capture_complete(candidate):
            return {
                "seed": seed,
                "variant": variant,
                "status": "reused",
                "capture_root": str(candidate),
                "runtime_seconds": 0.0,
            }
    capture_root = seed_root / f"run{max(run_indices, default=0) + 1:03d}"
    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "run_actual_view_motion_smoke.py"),
        "--seed",
        str(seed),
        "--scanned-basket-perception-pilot",
        "--calibration-scene-variant",
        variant,
        "--requested-views",
        "close_high",
        "right",
        "--output-dir",
        str(capture_root),
    ]
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"seed{seed:03d}_{variant}.stdout.log"
    stderr_path = log_root / f"seed{seed:03d}_{variant}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=420.0,
            check=False,
        )
    result = {
        "seed": seed,
        "variant": variant,
        "status": (
            "completed"
            if completed.returncode == 0 and capture_complete(capture_root)
            else "failed"
        ),
        "capture_root": str(capture_root),
        "runtime_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    return result


def collect_rows(capture_results: list[dict]) -> list[dict]:
    rows = []
    for capture in capture_results:
        if capture["status"] not in ("completed", "reused"):
            continue
        capture_root = Path(capture["capture_root"])
        for view_id in VIEWS:
            measurement = json.loads(
                (
                    capture_root
                    / "observations"
                    / view_id
                    / "objective_occlusion.json"
                ).read_text(encoding="utf-8")
            )
            reference_measurement = json.loads(
                (
                    capture_root
                    / "observations"
                    / view_id
                    / "objective_reference_occlusion.json"
                ).read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "seed": capture["seed"],
                    "variant": capture["variant"],
                    "view_id": view_id,
                    "valid": measurement["valid"],
                    "visible_target_pixels": measurement[
                        "visible_target_pixels"
                    ],
                    "amodal_target_pixels": measurement[
                        "amodal_target_pixels"
                    ],
                    "visible_fraction_of_amodal": measurement[
                        "visible_fraction_of_amodal"
                    ],
                    "occlusion_fraction": measurement[
                        "occlusion_fraction"
                    ],
                    "severity": measurement["severity"],
                    "fully_hidden": measurement["fully_hidden"],
                    "reference_valid": reference_measurement["valid"],
                    "reference_revealed_target_pixels": (
                        reference_measurement[
                            "reference_revealed_target_pixels"
                        ]
                    ),
                    "reference_occlusion_fraction": (
                        reference_measurement[
                            "reference_occlusion_fraction"
                        ]
                    ),
                    "reference_severity": reference_measurement[
                        "severity"
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
    )
    parser.add_argument("--seed-start", type=int, default=156)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "inside_clear",
            "outside",
            "rim_occluded",
            "covered_unknown",
            "behind_ambiguous",
            "close_high_only",
            "right_only",
            "either_view",
            "cover_removal_required",
        ),
    )
    parser.add_argument("--capture-offset", type=int, default=0)
    parser.add_argument("--capture-limit", type=int)
    args = parser.parse_args()
    require_single_gpu_policy()
    scenes = validation_scenes(
        args.seed_start,
        tuple(args.variants) if args.variants else VALIDATION_VARIANT_PATTERN,
    )
    if args.capture_offset < 0 or args.capture_offset >= len(scenes):
        raise ValueError("capture-offset escapes the validation scene list")
    if args.capture_limit is not None and args.capture_limit < 1:
        raise ValueError("capture-limit must be positive")
    requested_scenes = scenes[
        args.capture_offset : (
            None
            if args.capture_limit is None
            else args.capture_offset + args.capture_limit
        )
    ]
    if any(seed in RESERVED_TEST_SEEDS for seed, _ in scenes):
        raise RuntimeError("Reserved test seeds must not be used for validation")
    environment = single_gpu_environment()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            ROOT
            / "outputs"
            / "live_pipeline"
            / (
                "objective_reference_occlusion_validation_"
                f"seed{scenes[0][0]}_{scenes[-1][0]}_gpu5"
            )
        ).resolve()
    )
    allowed_root = (ROOT / "outputs" / "live_pipeline").resolve()
    if not output_root.is_relative_to(allowed_root):
        raise ValueError(f"Output root must be under {allowed_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    capture_results = []
    for seed, variant in requested_scenes:
        capture_results.append(
            run_capture(output_root, seed, variant, environment)
        )
        write_json_atomic(
            output_root / "progress.json",
            {
                "schema_version": "objective-occlusion-validation-progress-v1",
                "capture_results": capture_results,
            },
        )
    rows = collect_rows(capture_results)
    csv_path = output_root / "objective_occlusion_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    completed = [
        result
        for result in capture_results
        if result["status"] in ("completed", "reused")
    ]
    failed = [
        result for result in capture_results if result["status"] == "failed"
    ]
    summary = {
        "schema_version": "objective-reference-occlusion-validation-summary-v1",
        "status": "completed" if not failed else "partial_failure",
        "purpose": (
            "pilot validation of simulator-only total and reference-attributed "
            "amodal occlusion ground truth"
        ),
        "valid_for_final_evaluation": False,
        "training_performed": False,
        "calibration_performed": False,
        "reserved_test_seeds_used": False,
        "gpu_policy": {
            "physical_gpu": 5,
            "renderer_active_gpu": 5,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "runtime_seconds": time.perf_counter() - started,
        "requested_episode_count": len(requested_scenes),
        "full_validation_episode_count": len(scenes),
        "capture_offset": args.capture_offset,
        "completed_episode_count": len(completed),
        "failed_episode_count": len(failed),
        "capture_results": capture_results,
        "observation_count": len(rows),
        "severity_counts": {
            severity: sum(row["severity"] == severity for row in rows)
            for severity in ("no", "partial", "severe", "unknown")
        },
        "reference_severity_counts": {
            severity: sum(
                row["reference_severity"] == severity for row in rows
            )
            for severity in ("no", "partial", "severe", "unknown")
        },
        "rows_file": csv_path.name,
        "definition": (
            "1 - pixels(actual target-ID mask intersect target-only amodal "
            "support) / target-only amodal pixels under the same camera and "
            "target pose"
        ),
        "reference_definition": (
            "pixels newly revealed by hiding only /World/OpenContainer, "
            "divided by target-only amodal pixels at the same camera and "
            "target pose"
        ),
        "thresholds": {
            "no": "[0.0, 0.10)",
            "partial": "[0.10, 0.60)",
            "severe": "[0.60, 1.0]",
        },
    }
    write_json_atomic(
        output_root / "capture_manifest.json",
        {
            "schema_version": "existing-calibration-capture-manifest-v1",
            "purpose": (
                "reuse objective-occlusion validation captures for "
                "single-model-at-a-time learned perception validation"
            ),
            "episodes": [
                {
                    "seed": result["seed"],
                    "variant": result["variant"],
                    "capture_dir": result["capture_root"],
                }
                for result in completed
            ],
            "reserved_test_seeds_used": False,
            "valid_for_final_evaluation": False,
        },
    )
    write_json_atomic(output_root / "summary.json", summary)
    if failed:
        raise RuntimeError(f"Objective occlusion validation had failures: {failed}")
    print(f"OBJECTIVE_OCCLUSION_VALIDATION={output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
