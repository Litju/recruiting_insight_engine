"""
Insight Engine -- Phase 2 (Component 1 + Component 2: Bias & Fairness Audit)
Production-grade, deterministic, rule-based HR analytics on top of the salary
predictor. No LLMs, no randomness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from app.ml.inference import SalaryPredictor

# -------------------------------------------------------------------------
# PATHS & CONSTANTS
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MERGED_FILE = PROCESSED_DIR / "merged.csv"

FEATURE_KEYS = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
]

REQUIRED_COLUMNS = set(FEATURE_KEYS + ["Salary"])

GAP_RATIO_THRESHOLD = 0.80
ABS_DIFF_THRESHOLD = 15000.0


class InsightEngine:
    """
    Expert HR analytics on top of the salary model.
    Deterministic, defensive, and aligned with the 5-feature schema.
    """

    def __init__(self) -> None:
        # Load prediction pipeline (already enforces strict 5 columns internally)
        self.predictor = SalaryPredictor()

        # Load processed dataset
        if not MERGED_FILE.exists():
            raise FileNotFoundError(
                f"Missing processed dataset: {MERGED_FILE}\n"
                "Run: python app/ml/train_pipeline.py"
            )

        self.df = pd.read_csv(MERGED_FILE)

        # Validate required columns
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Merged dataset missing columns: {missing}")

        # Pre-compute global stats once
        self.global_stats = self._compute_global_stats()
        # Pre-compute dataset-level bias audit once (reused per request)
        self.bias_audit = self._compute_bias_audit()

    # ------------------------------------------------------------------
    # FEATURE NORMALIZATION
    # ------------------------------------------------------------------
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure we only keep the 5 expected keys and preserve order.
        Missing keys are filled with None (preprocessor will impute).
        Extra keys are dropped to prevent feature explosion.
        """
        normalized = {}
        for key in FEATURE_KEYS:
            normalized[key] = features.get(key)
        return normalized

    # ------------------------------------------------------------------
    # GLOBAL STATS
    # ------------------------------------------------------------------
    def _compute_global_stats(self) -> Dict[str, float]:
        salary = self.df["Salary"].dropna()
        return {
            "mean_salary": float(salary.mean()),
            "median_salary": float(salary.median()),
            "p25": float(salary.quantile(0.25)),
            "p75": float(salary.quantile(0.75)),
            "min_salary": float(salary.min()),
            "max_salary": float(salary.max()),
        }

    # ------------------------------------------------------------------
    # COHORT HELPERS
    # ------------------------------------------------------------------
    def _cohort_stats(self, col: str, value: Any) -> Optional[Dict[str, float]]:
        """
        Return cohort statistics for a given categorical value, or None if absent.
        """
        subset = self.df[self.df[col] == value]
        if subset.empty:
            return None

        salary = subset["Salary"].dropna()
        if salary.empty:
            return None

        stats = {
            "count": int(len(salary)),
            "mean_salary": float(salary.mean()),
            "median_salary": float(salary.median()),
            "p25": float(salary.quantile(0.25)),
            "p75": float(salary.quantile(0.75)),
        }

        if "Years of Experience" in subset.columns:
            exp = subset["Years of Experience"].dropna()
            if not exp.empty:
                stats["mean_experience"] = float(exp.mean())

        return stats

    @staticmethod
    def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    @staticmethod
    def _safe_pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """
        Percentage difference: (a - b) / b * 100
        Returns None if denominator is invalid.
        """
        if a is None or b is None or b == 0:
            return None
        return float((a - b) / b * 100.0)

    # ------------------------------------------------------------------
    # BIAS & FAIRNESS (Component 2)
    # ------------------------------------------------------------------
    def _age_group(self, age: Any) -> str:
        """
        Map raw age to a stable, interpretable age bucket.
        """
        try:
            a = float(age)
        except (TypeError, ValueError):
            return "unknown"

        if a < 20:
            return "<20"
        if a < 30:
            return "20-29"
        if a < 40:
            return "30-39"
        if a < 50:
            return "40-49"
        if a < 60:
            return "50-59"
        return "60+"

    def _compute_bias_for_categorical(
        self,
        series: pd.Series,
        population_mean: float,
    ) -> Tuple[Dict[str, float], Dict[str, int], Optional[float], Optional[float], Dict[str, float]]:
        """
        Generic bias computation for a categorical series.
        Returns subgroup means, counts, gap ratio, abs diff, parity diffs.
        """
        valid = series.dropna()
        if valid.empty:
            return {}, {}, None, None, {}

        grouped = self.df.loc[valid.index].groupby(valid).agg(
            mean_salary=("Salary", "mean"),
            count=("Salary", "count"),
        )

        means = grouped["mean_salary"].to_dict()
        counts = grouped["count"].astype(int).to_dict()

        if len(means) < 2:
            return means, counts, None, None, {
                k: float(v - population_mean) for k, v in means.items()
            }

        max_mean = max(means.values())
        min_mean = min(means.values())

        gap_ratio = None
        if max_mean != 0:
            gap_ratio = float(min_mean / max_mean)

        abs_diff = float(abs(max_mean - min_mean))

        parity_diffs = {k: float(v - population_mean) for k, v in means.items()}

        return means, counts, gap_ratio, abs_diff, parity_diffs

    def _bias_flags(self, label: str, gap_ratio: Optional[float], abs_diff: Optional[float]) -> list:
        """
        Generate bias flags based on configurable thresholds.
        """
        flags = []
        if gap_ratio is not None and gap_ratio < GAP_RATIO_THRESHOLD:
            flags.append(f"{label}_bias_high_gap")
        if abs_diff is not None and abs_diff > ABS_DIFF_THRESHOLD:
            flags.append(f"{label}_bias_high_abs_diff")
        return flags

    def _compute_bias_audit(self) -> Dict[str, Any]:
        """
        Compute dataset-level bias metrics for Gender, Education Level, Age Groups.
        """
        population_mean = float(self.df["Salary"].mean())

        # Gender
        gender_means, gender_counts, gender_gap_ratio, gender_abs_diff, gender_parity = (
            self._compute_bias_for_categorical(self.df["Gender"], population_mean)
        )

        # Education
        edu_means, edu_counts, edu_gap_ratio, edu_abs_diff, edu_parity = (
            self._compute_bias_for_categorical(self.df["Education Level"], population_mean)
        )

        # Age groups (derived)
        age_groups_series = self.df["Age"].map(self._age_group)
        age_means, age_counts, age_gap_ratio, age_abs_diff, age_parity = (
            self._compute_bias_for_categorical(age_groups_series, population_mean)
        )

        # Flags (aggregate)
        flags = []
        flags.extend(self._bias_flags("gender", gender_gap_ratio, gender_abs_diff))
        flags.extend(self._bias_flags("education", edu_gap_ratio, edu_abs_diff))
        flags.extend(self._bias_flags("age", age_gap_ratio, age_abs_diff))

        return {
            "gender": {
                "group_means": gender_means,
                "group_counts": gender_counts,
                "gap_ratio": gender_gap_ratio,
                "abs_diff": gender_abs_diff,
                "parity_diff_vs_population": gender_parity,
            },
            "education": {
                "group_means": edu_means,
                "group_counts": edu_counts,
                "gap_ratio": edu_gap_ratio,
                "abs_diff": edu_abs_diff,
                "parity_diff_vs_population": edu_parity,
            },
            "age_groups": {
                "group_means": age_means,
                "group_counts": age_counts,
                "gap_ratio": age_gap_ratio,
                "abs_diff": age_abs_diff,
                "parity_diff_vs_population": age_parity,
            },
            "bias_flags": flags,
            "population_mean_salary": population_mean,
            "thresholds": {
                "gap_ratio": GAP_RATIO_THRESHOLD,
                "abs_diff": ABS_DIFF_THRESHOLD,
            },
        }

    # ------------------------------------------------------------------
    # INTERPRETABILITY (Component 3)
    # ------------------------------------------------------------------
    def _normalized_feature_importance(self) -> Dict[str, float]:
        """
        Aggregate model feature importances back to the 5 raw features.
        Falls back to equal weights if importances are unavailable.
        """
        # Default equal weights
        default = {k: 1.0 / len(FEATURE_KEYS) for k in FEATURE_KEYS}

        pipeline = getattr(self.predictor, "model", None)
        if pipeline is None or not hasattr(pipeline, "named_steps"):
            return default

        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")
        if model is None or preprocessor is None or not hasattr(model, "feature_importances_"):
            return default

        try:
            transformed_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_
        except Exception:
            return default

        agg: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}

        for name, imp in zip(transformed_names, importances):
            base = name
            if "__" in name:
                base = name.split("__", 1)[1]
            if base.startswith("Age"):
                agg["Age"] += float(imp)
            elif base.startswith("Years of Experience"):
                agg["Years of Experience"] += float(imp)
            elif base.startswith("Gender"):
                agg["Gender"] += float(imp)
            elif base.startswith("Education Level"):
                agg["Education Level"] += float(imp)
            elif base.startswith("Job Title"):
                agg["Job Title"] += float(imp)

        total = sum(agg.values())
        if total <= 0:
            return default

        return {k: round(v / total, 6) for k, v in agg.items()}

    def _feature_deltas(self, features: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Compute directional deltas vs reference means for each feature.
        """
        deltas: Dict[str, Optional[float]] = {k: None for k in FEATURE_KEYS}

        # Numeric deltas vs global means
        mean_age = float(self.df["Age"].dropna().mean()) if not self.df["Age"].dropna().empty else 0.0
        mean_exp = float(self.df["Years of Experience"].dropna().mean()) if not self.df["Years of Experience"].dropna().empty else 0.0

        age_val = features.get("Age")
        try:
            deltas["Age"] = float(age_val) - mean_age if age_val is not None else None
        except (TypeError, ValueError):
            deltas["Age"] = None

        exp_val = features.get("Years of Experience")
        try:
            deltas["Years of Experience"] = float(exp_val) - mean_exp if exp_val is not None else None
        except (TypeError, ValueError):
            deltas["Years of Experience"] = None

        # Categorical deltas vs global salary mean
        baseline_salary = self.global_stats["mean_salary"]

        job_stats = self._cohort_stats("Job Title", features.get("Job Title"))
        if job_stats and "mean_salary" in job_stats:
            deltas["Job Title"] = float(job_stats["mean_salary"] - baseline_salary)

        edu_stats = self._cohort_stats("Education Level", features.get("Education Level"))
        if edu_stats and "mean_salary" in edu_stats:
            deltas["Education Level"] = float(edu_stats["mean_salary"] - baseline_salary)

        gender_stats = self._cohort_stats("Gender", features.get("Gender"))
        if gender_stats and "mean_salary" in gender_stats:
            deltas["Gender"] = float(gender_stats["mean_salary"] - baseline_salary)

        return deltas

    def _compute_contributions(
        self,
        features: Dict[str, Any],
        prediction: float,
        importances: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Approximate contribution per feature based on:
            - normalized global importances
            - directional deltas vs reference means
            - proportion of prediction lift over global mean
        Deterministic and lightweight (no SHAP).
        """
        baseline = self.global_stats["mean_salary"]
        total_lift = prediction - baseline
        deltas = self._feature_deltas(features)

        contributions: Dict[str, float] = {}
        for feat in FEATURE_KEYS:
            imp = importances.get(feat, 0.0)
            delta = deltas.get(feat)
            if delta is None:
                contributions[feat] = 0.0
                continue

            # Scale factor based on relative delta magnitude; capped for stability
            ref = baseline if feat in {"Job Title", "Education Level", "Gender"} else (abs(delta) + abs(baseline))
            ref = max(ref, 1e-9)
            magnitude = min(2.0, abs(delta) / ref)
            direction = 1.0 if delta >= 0 else -1.0

            contrib = total_lift * imp * direction * (0.5 + 0.5 * magnitude)
            contributions[feat] = float(round(contrib, 2))

        return contributions

    def _top_drivers(self, contributions: Dict[str, float], top_n: int = 3) -> list:
        """
        Rank features by absolute contribution.
        """
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [
            {"feature": feat, "contribution": contrib}
            for feat, contrib in ranked[:top_n]
        ]

    def _prediction_explanation(
        self,
        importances: Dict[str, float],
        contributions: Dict[str, float],
        prediction: float,
    ) -> str:
        """
        Deterministic, short textual explanation of main drivers.
        """
        sorted_imps = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        primary = [f for f, w in sorted_imps if w > 0][:3] or FEATURE_KEYS[:3]

        top_contrib = self._top_drivers(contributions, top_n=2)
        driver_names = [d["feature"] for d in top_contrib if abs(d["contribution"]) > 0] or primary[:2]

        direction = "above" if prediction >= self.global_stats["mean_salary"] else "below"

        if len(driver_names) >= 2:
            lead = f"The prediction is primarily influenced by {driver_names[0]} and {driver_names[1]}"
        elif len(driver_names) == 1:
            lead = f"The prediction is primarily influenced by {driver_names[0]}"
        else:
            lead = "The prediction is primarily influenced by the core features"

        secondary = ""
        if len(primary) >= 3:
            secondary = f", with secondary impact from {primary[2]}"

        closing = (
            f". Based on the model's learned patterns, these factors drive the estimated salary "
            f"{direction} the overall market average."
        )

        return lead + secondary + closing

    def _compute_interpretability(
        self,
        features: Dict[str, Any],
        prediction: float,
    ) -> Dict[str, Any]:
        """
        Compute lightweight interpretability signals for the current prediction.
        """
        importances = self._normalized_feature_importance()
        contributions = self._compute_contributions(features, prediction, importances)
        top_drivers = self._top_drivers(contributions, top_n=3)
        explanation = self._prediction_explanation(importances, contributions, prediction)

        return {
            "feature_importance": importances,
            "top_drivers": top_drivers,
            "contribution_breakdown": contributions,
            "prediction_explanation": explanation,
        }

    # ------------------------------------------------------------------
    # MARKET BAND & FAIRNESS
    # ------------------------------------------------------------------
    def _market_band(self, pred: float) -> str:
        p25 = self.global_stats["p25"]
        p75 = self.global_stats["p75"]

        if pred < p25:
            return "below_market"
        if pred > p75:
            return "above_market"
        return "at_market"

    def _fairness_score(self, pred: float) -> float:
        """
        Simple fairness score based on deviation from global mean.
        100 = perfectly aligned with mean
        0   = >50% deviation
        """
        mean_salary = self.global_stats["mean_salary"]
        if mean_salary <= 0:
            return 50.0

        pct_dev = abs(pred - mean_salary) / mean_salary * 100.0
        penalty = min(pct_dev * 2.0, 100.0)  # cap influence after ~50% deviation
        score = max(0.0, 100.0 - penalty)
        return float(round(score, 1))

    @staticmethod
    def _confidence_from_count(n: int) -> Dict[str, Any]:
        """
        Map cohort sample size to qualitative + numeric confidence.
        """
        if n >= 50:
            return {"level": "high", "score": 0.9, "samples": n}
        if n >= 20:
            return {"level": "medium", "score": 0.7, "samples": n}
        if n >= 5:
            return {"level": "low", "score": 0.5, "samples": n}
        return {"level": "very_low", "score": 0.3, "samples": n}

    # ------------------------------------------------------------------
    # OFFER RANGE
    # ------------------------------------------------------------------
    def _recommended_offer_range(self, pred: float) -> Dict[str, float]:
        """
        Narrow offer band around the model prediction (+/-5%), clamped to global min/max.
        """
        low = float(round(pred * 0.95, 2))
        high = float(round(pred * 1.05, 2))

        global_min = self.global_stats["min_salary"]
        global_max = self.global_stats["max_salary"]

        low = max(low, global_min)
        high = min(high, global_max)

        if low > high:
            low, high = high, low

        return {"recommended_min": low, "recommended_max": high}

    # ------------------------------------------------------------------
    # FLAGS
    # ------------------------------------------------------------------
    def _generate_flags(
        self,
        pred: float,
        features: Dict[str, Any],
        job_stats: Optional[Dict[str, float]],
        edu_stats: Optional[Dict[str, float]],
        gender_stats: Optional[Dict[str, float]],
    ) -> list:
        flags = []

        p25 = self.global_stats["p25"]
        p75 = self.global_stats["p75"]

        if pred < p25:
            flags.append("below_market_salary")

        if pred > p75:
            flags.append("above_market_salary")

        if job_stats is not None:
            job_mean = job_stats["mean_salary"]
            pct_dev_job = self._safe_pct_diff(pred, job_mean)
            if pct_dev_job is not None and abs(pct_dev_job) > 30:
                flags.append("outlier_vs_job_title")

        years = features.get("Years of Experience")
        if job_stats is not None and "mean_experience" in job_stats and years is not None:
            try:
                years_float = float(years)
                mean_exp = job_stats["mean_experience"]
                if mean_exp > 0 and years_float < 0.5 * mean_exp:
                    flags.append("low_experience_for_job")
            except (TypeError, ValueError):
                pass

        return flags

    # ------------------------------------------------------------------
    # NARRATIVE
    # ------------------------------------------------------------------
    def _build_narrative(
        self,
        pred: float,
        market_band: str,
        delta_global_pct: Optional[float],
        fairness_score: float,
        offer_range: Dict[str, float],
        job_title: Optional[str],
        job_stats: Optional[Dict[str, float]],
        flags: list,
    ) -> str:
        """
        Compact, corporate-style narrative (no LLM).
        """
        parts = []

        mb_label = {
            "below_market": "below the estimated market level",
            "at_market": "aligned with the estimated market level",
            "above_market": "above the estimated market level",
        }.get(market_band, "close to the estimated market level")

        parts.append(
            f"The model estimates a target salary of approximately {round(pred):,.0f} USD, "
            f"which is {mb_label}."
        )

        if delta_global_pct is not None:
            sign = "higher than" if delta_global_pct > 0 else "lower than"
            parts.append(
                f"This is about {abs(delta_global_pct):.1f}% {sign} the overall market average "
                f"observed in the historical dataset."
            )

        if job_title and job_stats is not None:
            parts.append(
                f"For the job title '{job_title}', the dataset contains {job_stats['count']} "
                f"comparable profiles with an average salary of {job_stats['mean_salary']:,.0f} USD."
            )

        parts.append(
            f"The internal salary fairness score for this profile is {fairness_score:.1f}/100."
        )

        parts.append(
            "A reasonable offer band, balancing internal fairness and market consistency, "
            f"would be in the range of {offer_range['recommended_min']:,.0f}-"
            f"{offer_range['recommended_max']:,.0f} USD."
        )

        if flags:
            flags_str = ", ".join(flags)
            parts.append(f"Key risk/attention flags raised for this profile: {flags_str}.")
        else:
            parts.append("No major risk flags were raised for this profile.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # PUBLIC: MAIN ENTRYPOINT
    # ------------------------------------------------------------------
    def generate_insights(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the full insight bundle for a candidate profile.
        """
        clean_features = self._normalize_features(features)

        # 1) Prediction
        salary_pred = float(self.predictor.predict_one(clean_features))

        # 2) Cohorts
        job_title = clean_features.get("Job Title")
        edu_level = clean_features.get("Education Level")
        gender = clean_features.get("Gender")

        job_stats = self._cohort_stats("Job Title", job_title)
        edu_stats = self._cohort_stats("Education Level", edu_level)
        gender_stats = self._cohort_stats("Gender", gender)

        # 3) Market & fairness
        market_band = self._market_band(salary_pred)
        fairness_score = self._fairness_score(salary_pred)
        delta_global_pct = self._safe_pct_diff(
            salary_pred,
            self.global_stats["mean_salary"],
        )

        # 4) Offer range
        offer_range = self._recommended_offer_range(salary_pred)

        # 5) Flags
        flags = self._generate_flags(
            salary_pred,
            clean_features,
            job_stats,
            edu_stats,
            gender_stats,
        )

        # 6) Interpretability
        interpretability = self._compute_interpretability(clean_features, salary_pred)

        # 7) Narrative
        narrative = self._build_narrative(
            pred=salary_pred,
            market_band=market_band,
            delta_global_pct=delta_global_pct,
            fairness_score=fairness_score,
            offer_range=offer_range,
            job_title=job_title,
            job_stats=job_stats,
            flags=flags,
        )

        # 8) Build structured output
        result: Dict[str, Any] = {
            "input_features": clean_features,
            "salary_prediction": salary_pred,
            "salary_insights": {
                "global_market": {
                    **self.global_stats,
                    "market_band": market_band,
                    "delta_vs_mean_pct": delta_global_pct,
                    "fairness_score": fairness_score,
                    "recommended_offer_range": offer_range,
                },
                "by_job_title": {
                    "job_title": job_title,
                    "cohort": job_stats,
                    "delta_vs_job_mean": self._safe_diff(
                        salary_pred,
                        job_stats["mean_salary"] if job_stats else None,
                    ),
                    "delta_vs_job_mean_pct": self._safe_pct_diff(
                        salary_pred,
                        job_stats["mean_salary"] if job_stats else None,
                    ),
                    "confidence": (
                        self._confidence_from_count(job_stats["count"])
                        if job_stats is not None
                        else None
                    ),
                },
                "by_education_level": {
                    "education_level": edu_level,
                    "cohort": edu_stats,
                    "delta_vs_edu_mean": self._safe_diff(
                        salary_pred,
                        edu_stats["mean_salary"] if edu_stats else None,
                    ),
                    "delta_vs_edu_mean_pct": self._safe_pct_diff(
                        salary_pred,
                        edu_stats["mean_salary"] if edu_stats else None,
                    ),
                    "confidence": (
                        self._confidence_from_count(edu_stats["count"])
                        if edu_stats is not None
                        else None
                    ),
                },
                "by_gender": {
                    "gender": gender,
                    "cohort": gender_stats,
                    "delta_vs_gender_mean": self._safe_diff(
                        salary_pred,
                        gender_stats["mean_salary"] if gender_stats else None,
                    ),
                    "delta_vs_gender_mean_pct": self._safe_pct_diff(
                        salary_pred,
                        gender_stats["mean_salary"] if gender_stats else None,
                    ),
                    "confidence": (
                        self._confidence_from_count(gender_stats["count"])
                        if gender_stats is not None
                        else None
                    ),
                },
            },
            "bias_audit": self.bias_audit,
            "interpretability": interpretability,
            "flags": flags,
            "narrative": narrative,
        }

        return result


# -------------------------------------------------------------------------
# CLI TEST
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("-> Running InsightEngine Phase 2 test...")

    engine = InsightEngine()

    example = {
        "Age": 32,
        "Gender": "Male",
        "Education Level": "Bachelor's",
        "Job Title": "Software Engineer",
        "Years of Experience": 5,
    }

    from pprint import pprint

    bundle = engine.generate_insights(example)
    pprint(bundle)
