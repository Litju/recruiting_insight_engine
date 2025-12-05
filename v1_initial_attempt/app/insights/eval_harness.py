"""
Evaluation Harness -- Recruiting Insight Engine (Phase 2, Component 4)

Purpose:
    - Run a fixed set of canonical test profiles through InsightEngine.
    - Generate a deterministic baseline JSON of insights for regression tests.
    - Compare current engine outputs vs the stored baseline and detect drift.

Usage (from project root):
    # 1) Generate baseline (first time or after re-benchmark)
    python app/insights/eval_harness.py baseline

    # 2) Compare current engine vs baseline
    python app/insights/eval_harness.py compare

Outputs:
    - tests/baseline_insights.json
    - tests/drift_report.json

Constraints:
    - Deterministic (no randomness, no timestamps in outputs)
    - No external dependencies
    - Windows-friendly paths
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.insights.engine import InsightEngine

# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
BASELINE_FILE = TESTS_DIR / "baseline_insights.json"
DRIFT_REPORT_FILE = TESTS_DIR / "drift_report.json"

TESTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# FIXED TEST PROFILES (canonical, deterministic)
# -------------------------------------------------------------------------
TEST_PROFILES: Dict[str, Dict[str, Any]] = {
    "mid_dev_bachelor_male": {
        "Age": 32,
        "Gender": "Male",
        "Education Level": "Bachelor's",
        "Job Title": "Software Engineer",
        "Years of Experience": 5,
    },
    "junior_data_analyst_female": {
        "Age": 28,
        "Gender": "Female",
        "Education Level": "Master's",
        "Job Title": "Data Analyst",
        "Years of Experience": 3,
    },
    "senior_manager_phd_male": {
        "Age": 45,
        "Gender": "Male",
        "Education Level": "PhD",
        "Job Title": "Senior Manager",
        "Years of Experience": 15,
    },
    "mid_sales_associate_female": {
        "Age": 36,
        "Gender": "Female",
        "Education Level": "Bachelor's",
        "Job Title": "Sales Associate",
        "Years of Experience": 7,
    },
    "director_master_male": {
        "Age": 52,
        "Gender": "Male",
        "Education Level": "Master's",
        "Job Title": "Director",
        "Years of Experience": 20,
    },
}

# -------------------------------------------------------------------------
# DRIFT CONFIG
# -------------------------------------------------------------------------
SALARY_ATOL = 0.01  # absolute tolerance on salary prediction (USD)
SALARY_RTOL = 0.01  # relative tolerance (1% drift ignored)
FAIRNESS_ATOL = 0.1
FAIRNESS_RTOL = 0.01


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def _run_engine_on_profiles(engine: InsightEngine) -> Dict[str, Dict[str, Any]]:
    """
    Run InsightEngine on all TEST_PROFILES and return raw insight bundles.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for profile_id, features in TEST_PROFILES.items():
        bundle = engine.generate_insights(features)
        results[profile_id] = bundle
    return results


def _compact_view(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact, comparison-friendly view from a full insight bundle.
    Focus on core drift signals.
    """
    salary_pred = insight.get("salary_prediction")

    salary_insights = insight.get("salary_insights", {})
    global_market = salary_insights.get("global_market", {})
    market_band = global_market.get("market_band")
    fairness_score = global_market.get("fairness_score")

    flags = insight.get("flags", [])
    bias_audit = insight.get("bias_audit", {})
    bias_flags = bias_audit.get("bias_flags", [])

    return {
        "salary_prediction": salary_pred,
        "market_band": market_band,
        "fairness_score": fairness_score,
        "flags": sorted(flags) if isinstance(flags, list) else flags,
        "bias_flags": sorted(bias_flags) if isinstance(bias_flags, list) else bias_flags,
    }


def _num_diff(
    base_val: Any,
    cur_val: Any,
    atol: float,
    rtol: float,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two numeric values with absolute and relative tolerances.
    Returns (drift_detected, details).
    """
    try:
        b = float(base_val)
        c = float(cur_val)
    except (TypeError, ValueError):
        return True, {"baseline": base_val, "current": cur_val, "note": "non-numeric"}

    delta = c - b
    rel = abs(delta) / (abs(b) + 1e-9)
    drift = not (abs(delta) <= atol or rel <= rtol)

    details = {
        "baseline": round(b, 4),
        "current": round(c, 4),
        "delta": round(delta, 4),
        "delta_pct": round(rel * 100.0, 4),
        "tolerance": {"atol": atol, "rtol": rtol},
    }
    return drift, details


def _severity(delta_pct: float) -> str:
    """
    Map percentage delta to severity bucket.
    """
    if delta_pct is None:
        return "unknown"
    if abs(delta_pct) < 1:
        return "minor"
    if abs(delta_pct) < 5:
        return "moderate"
    return "major"


def _recommendation(key: str, delta_pct: Optional[float], flags: List[str]) -> str:
    """
    Simple deterministic recommendation text for drifted fields.
    """
    drift_dir = "increase" if (delta_pct or 0) > 0 else "decrease"
    if key == "salary_prediction":
        return f"Investigate model pipeline changes causing a {drift_dir} in salary prediction."
    if key == "fairness_score":
        return "Review fairness scoring and cohort statistics for potential distribution shifts."
    if key == "market_band":
        return "Re-run market benchmarks and verify percentile thresholds."
    if key in {"flags", "bias_flags"}:
        return "Review rules/thresholds that trigger flags; confirm data integrity."
    # default
    return "Review recent code/model changes impacting this signal."


def _compare_views(
    baseline_view: Dict[str, Any],
    current_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two compact views and produce a drift-aware diff structure.
    Includes severity and recommendation per drifted field.
    """
    diffs: Dict[str, Any] = {}
    keys = set(baseline_view.keys()) | set(current_view.keys())

    for key in keys:
        base_val = baseline_view.get(key)
        cur_val = current_view.get(key)

        if key in {"salary_prediction"}:
            drift, details = _num_diff(base_val, cur_val, SALARY_ATOL, SALARY_RTOL)
            if drift:
                details["severity"] = _severity(details.get("delta_pct"))
                details["recommendation"] = _recommendation(key, details.get("delta_pct"), [])
                diffs[key] = details

        elif key in {"fairness_score"}:
            drift, details = _num_diff(base_val, cur_val, FAIRNESS_ATOL, FAIRNESS_RTOL)
            if drift:
                details["severity"] = _severity(details.get("delta_pct"))
                details["recommendation"] = _recommendation(key, details.get("delta_pct"), [])
                diffs[key] = details

        elif key in {"market_band"}:
            if base_val != cur_val:
                diffs[key] = {
                    "baseline": base_val,
                    "current": cur_val,
                    "severity": "moderate",
                    "recommendation": _recommendation(key, None, []),
                }

        elif key in {"flags", "bias_flags"}:
            base_list = sorted(base_val) if isinstance(base_val, list) else base_val
            cur_list = sorted(cur_val) if isinstance(cur_val, list) else cur_val
            if base_list != cur_list:
                diffs[key] = {
                    "baseline": base_list,
                    "current": cur_list,
                    "severity": "moderate",
                    "recommendation": _recommendation(key, None, cur_list or []),
                }

        else:
            if base_val != cur_val:
                diffs[key] = {
                    "baseline": base_val,
                    "current": cur_val,
                    "severity": "minor",
                    "recommendation": _recommendation(key, None, []),
                }

    return diffs


def _validate_bundle(bundle: Dict[str, Any]) -> bool:
    """
    Basic sanity checks to ensure required keys exist in an insight bundle.
    """
    required = ["salary_prediction", "salary_insights", "bias_audit"]
    return all(k in bundle for k in required)


# -------------------------------------------------------------------------
# MODES
# -------------------------------------------------------------------------
def generate_baseline() -> None:
    """
    Generate baseline_insights.json from current InsightEngine.
    """
    print(">> Initializing InsightEngine...")
    engine = InsightEngine()

    print(">> Running engine on test profiles...")
    raw_results = _run_engine_on_profiles(engine)

    # Validate bundles before saving
    invalid = [pid for pid, b in raw_results.items() if not _validate_bundle(b)]
    if invalid:
        print("!! Aborting: invalid insight bundles for profiles:", ", ".join(invalid))
        return

    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_FILE.open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, sort_keys=True)

    print(f">> Baseline saved -> {BASELINE_FILE}")


def compare_against_baseline() -> None:
    """
    Compare current engine insights vs stored baseline.
    Produces tests/drift_report.json with per-profile diffs and summary.
    """
    if not BASELINE_FILE.exists():
        print(f"!! No baseline found at {BASELINE_FILE}")
        print("   Run first: python app/insights/eval_harness.py baseline")
        return

    print(">> Loading baseline insights...")
    with BASELINE_FILE.open("r", encoding="utf-8") as f:
        baseline_data: Dict[str, Any] = json.load(f)

    print(">> Initializing InsightEngine (current version)...")
    engine = InsightEngine()

    print(">> Running engine on test profiles (current version)...")
    current_data = _run_engine_on_profiles(engine)

    drift_report: Dict[str, Any] = {
        "summary": {
            "profiles_tested": len(TEST_PROFILES),
            "profiles_with_drift": 0,
            "categories": {
                "prediction_drift": 0,
                "fairness_drift": 0,
                "structural_drift": 0,
                "flags_drift": 0,
            },
        },
        "profiles": {},
    }

    profiles_with_drift: List[str] = []

    for profile_id in TEST_PROFILES.keys():
        base_insight = baseline_data.get(profile_id)
        cur_insight = current_data.get(profile_id)

        if base_insight is None or cur_insight is None:
            drift_report["profiles"][profile_id] = {
                "drift": True,
                "error": "Missing profile in baseline or current run.",
                "baseline_present": base_insight is not None,
                "current_present": cur_insight is not None,
            }
            profiles_with_drift.append(profile_id)
            drift_report["summary"]["categories"]["structural_drift"] += 1
            continue

        if not (_validate_bundle(base_insight) and _validate_bundle(cur_insight)):
            drift_report["profiles"][profile_id] = {
                "drift": True,
                "error": "Invalid insight bundle structure.",
            }
            profiles_with_drift.append(profile_id)
            drift_report["summary"]["categories"]["structural_drift"] += 1
            continue

        base_view = _compact_view(base_insight)
        cur_view = _compact_view(cur_insight)
        diffs = _compare_views(base_view, cur_view)

        if diffs:
            profiles_with_drift.append(profile_id)
            # categorize drift
            if "salary_prediction" in diffs:
                drift_report["summary"]["categories"]["prediction_drift"] += 1
            if "fairness_score" in diffs:
                drift_report["summary"]["categories"]["fairness_drift"] += 1
            if ("flags" in diffs) or ("bias_flags" in diffs):
                drift_report["summary"]["categories"]["flags_drift"] += 1

            drift_report["summary"]["categories"]["structural_drift"] += int(
                any(k not in {"salary_prediction", "fairness_score", "market_band", "flags", "bias_flags"} for k in diffs)
            )

            drift_report["profiles"][profile_id] = {
                "drift": True,
                "diffs": diffs,
                "recommendation": "Review drifted fields and re-run baseline if intentional.",
            }
        else:
            drift_report["profiles"][profile_id] = {
                "drift": False,
                "diffs": {},
            }

    drift_report["summary"]["profiles_with_drift"] = len(profiles_with_drift)
    drift_report["summary"]["profile_ids_with_drift"] = profiles_with_drift

    with DRIFT_REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2, sort_keys=True)

    # Console summary (deterministic text)
    print(">> Drift Report")
    print(f"   Profiles tested     : {drift_report['summary']['profiles_tested']}")
    print(f"   Profiles with drift : {drift_report['summary']['profiles_with_drift']}")
    print(f"   Prediction drift    : {drift_report['summary']['categories']['prediction_drift']}")
    print(f"   Fairness drift      : {drift_report['summary']['categories']['fairness_drift']}")
    print(f"   Flags drift         : {drift_report['summary']['categories']['flags_drift']}")
    print(f"   Structural drift    : {drift_report['summary']['categories']['structural_drift']}")
    if profiles_with_drift:
        print(f"   IDs with drift      : {', '.join(profiles_with_drift)}")
    print(f">> Drift report saved -> {DRIFT_REPORT_FILE}")


# -------------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------------
def main() -> None:
    """
    CLI entrypoint.
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"

    if mode == "baseline":
        print("=== Generating baseline insights ===")
        generate_baseline()
    elif mode == "compare":
        print("=== Comparing current engine vs baseline ===")
        compare_against_baseline()
    else:
        print("Usage:")
        print("  python app/insights/eval_harness.py baseline   # generate baseline")
        print("  python app/insights/eval_harness.py compare    # compare vs baseline")


if __name__ == "__main__":
    main()
