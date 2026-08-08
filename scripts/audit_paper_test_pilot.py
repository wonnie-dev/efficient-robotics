"""Summarize the reserved-seed live pilot without calling it final evaluation."""
import glob, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_test_live_pilot_audit_seed014_030.json"
rows = []
missing = []
for seed in range(14, 31):
    paths = sorted(glob.glob(str(ROOT / "outputs/live_pipeline/learned_scanned_basket_e2e" / f"seed{seed:03d}" / "run*" / "pipeline_result.json")))
    if not paths:
        missing.append(seed)
        continue
    data = json.loads(Path(paths[-1]).read_text())
    rows.append({"seed": seed, "status": data.get("status"), "terminal_action": data.get("terminal_action"), "grasp_executed": data.get("grasp_executed"), "runtime_seconds": data.get("runtime_seconds"), "actual_mpc_solver": data.get("actual_mpc_solver"), "training_performed": data.get("training_performed"), "calibration_performed": data.get("calibration_performed"), "valid_for_final_evaluation": data.get("valid_for_final_evaluation")})
report = {"schema_version":"paper-test-live-pilot-audit-v1", "seed_range":[14,30], "completed_count":len(rows), "missing_seeds":missing, "grasp_count":sum(r["grasp_executed"] is True for r in rows), "terminal_grasp_count":sum(r["terminal_action"]=="grasp" for r in rows), "mean_runtime_seconds":statistics.mean(r["runtime_seconds"] for r in rows), "rows":rows, "interpretation":"Reserved-seed proposed-method pilot only. Same-input baseline/ablation, calibration, and formal task-risk-aware belief-space MPC are not included; do not report as final paper evidence.", "training":"not performed", "calibration":"not performed"}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("completed_count","missing_seeds","grasp_count","terminal_grasp_count","mean_runtime_seconds")}, indent=2))
