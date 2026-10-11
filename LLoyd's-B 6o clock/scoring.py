from __future__ import annotations

import math

import numpy as np
import pandas as pd


COMPONENTS = (
    ("Energy efficiency", "energy_efficiency_subscore", 25),
    ("Energy consumption intensity", "consumption_intensity_subscore", 20),
    ("Insulation", "insulation_subscore", 15),
    ("Heating system", "heating_subscore", 15),
    ("Windows and glazing", "glazing_subscore", 10),
    ("Renewable energy", "solar_subscore", 10),
    ("Smart energy management", "smart_controls_subscore", 5),
)
WEIGHTS = {name: weight for name, _, weight in COMPONENTS}

ENERGY_RATING_LOOKUP = {
    "A": 100,
    "B": 85,
    "C": 70,
    "D": 55,
    "E": 40,
    "F": 25,
    "G": 10,
}

# Each row is an inclusive upper bound in kWh/m²/year and its sub-score.
ENERGY_INTENSITY_LOOKUP = (
    (50, 100),
    (100, 85),
    (150, 70),
    (200, 55),
    (250, 40),
    (300, 25),
    (math.inf, 0),
)

INSULATION_LOOKUP = {
    "Poor": 15,
    "Fair": 45,
    "Good": 75,
    "Excellent": 100,
}
HEATING_LOOKUP = {
    "Direct electric": 25,
    "Electric boiler": 35,
    "Gas boiler": 50,
    "Air-source heat pump": 100,
}
GLAZING_LOOKUP = {
    "Single": 30,
    "Double": 75,
    "Triple": 100,
}
YES_NO_LOOKUP = {"No": 0, "Yes": 100}

CLASSIFICATION_BANDS = (
    (80, 100, "Highly efficient"),
    (65, 79, "Good"),
    (50, 64, "Moderate"),
    (30, 49, "Poor"),
    (0, 29, "Very poor"),
)


def energy_intensity_subscore(intensity: float | None) -> int | None:
    if intensity is None or pd.isna(intensity):
        return None
    if intensity < 0:
        raise ValueError("Energy consumption intensity cannot be negative.")
    for upper_bound, subscore in ENERGY_INTENSITY_LOOKUP:
        if intensity <= upper_bound:
            return subscore
    return 0


def classify_score(score: int | float | None) -> str:
    if score is None or pd.isna(score):
        return "Not scored"
    score_value = int(score)
    for lower_bound, upper_bound, label in CLASSIFICATION_BANDS:
        if lower_bound <= score_value <= upper_bound:
            return label
    raise ValueError(f"Score must be between 0 and 100; got {score_value}.")


def _category_subscores(
    values: pd.Series, lookup: dict[str, int]
) -> pd.Series:
    normalized_lookup = {
        str(label).strip().casefold(): score for label, score in lookup.items()
    }
    normalized_values = values.astype("string").str.strip().str.casefold()
    return normalized_values.map(normalized_lookup).fillna(0).astype(int)


def _yes_no_subscores(values: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        if pd.isna(value):
            return "Unknown"
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        normalized = str(value).strip().casefold()
        if normalized in {"yes", "true", "1", "y"}:
            return "Yes"
        if normalized in {"no", "false", "0", "n"}:
            return "No"
        return "Unknown"

    return values.map(normalize).map(YES_NO_LOOKUP).fillna(0).astype(int)


def _annualized_energy(homes: pd.DataFrame, monthly_energy: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "property_id",
        "month",
        "electricity_kwh",
        "gas_kwh",
    }
    missing_columns = required_columns.difference(monthly_energy.columns)
    if missing_columns:
        raise ValueError(
            "Monthly energy data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    records = monthly_energy[list(required_columns)].copy()
    records["month_period"] = pd.to_datetime(
        records["month"], errors="coerce"
    ).dt.to_period("M")
    records["electricity_kwh"] = pd.to_numeric(
        records["electricity_kwh"], errors="coerce"
    )
    records["gas_kwh"] = pd.to_numeric(records["gas_kwh"], errors="coerce")

    valid_month_keys = records["month_period"].notna()
    if records.loc[
        valid_month_keys, ["property_id", "month_period"]
    ].duplicated().any():
        raise ValueError("There must be no more than one row per home per month.")

    for fuel_column in ("electricity_kwh", "gas_kwh"):
        if records[fuel_column].dropna().lt(0).any():
            raise ValueError(f"{fuel_column} cannot contain negative values.")

    # A month counts only when both readings are present. This prevents an
    # unknown fuel reading from being silently counted as zero.
    complete_months = records.loc[
        valid_month_keys
        & records["electricity_kwh"].notna()
        & records["gas_kwh"].notna()
    ]
    usage_by_home = complete_months.groupby("property_id").agg(
        months_observed=("month_period", "nunique"),
        mean_monthly_electricity_kwh=("electricity_kwh", "mean"),
        mean_monthly_gas_kwh=("gas_kwh", "mean"),
    )
    usage_by_home["annualized_electricity_kwh"] = (
        usage_by_home["mean_monthly_electricity_kwh"] * 12
    )
    usage_by_home["annualized_gas_kwh"] = (
        usage_by_home["mean_monthly_gas_kwh"] * 12
    )

    result = homes.merge(
        usage_by_home[
            [
                "months_observed",
                "annualized_electricity_kwh",
                "annualized_gas_kwh",
            ]
        ],
        left_on="property_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    result["months_observed"] = result["months_observed"].fillna(0).astype(int)
    result["missing_month_count"] = (12 - result["months_observed"]).clip(lower=0)
    result["has_missing_months"] = result["months_observed"] < 12
    result["annualized_total_kwh"] = (
        result["annualized_electricity_kwh"] + result["annualized_gas_kwh"]
    )

    floor_area = pd.to_numeric(result["floor_area"], errors="coerce")
    if floor_area.isna().any() or floor_area.le(0).any():
        raise ValueError("Every home must have a positive floor_area.")
    result["energy_intensity_kwh_m2_year"] = (
        result["annualized_total_kwh"] / floor_area
    )
    return result


def score_homes(homes: pd.DataFrame, monthly_energy: pd.DataFrame) -> pd.DataFrame:
    """Score each home using fixed lookup tables and annualized energy data."""
    required_home_columns = {
        "property_id",
        "floor_area",
        "energy_rating",
        "insulation",
        "heating",
        "glazing",
        "solar",
        "smart_controls",
    }
    missing_columns = required_home_columns.difference(homes.columns)
    if missing_columns:
        raise ValueError(
            "Home data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if homes["property_id"].duplicated().any():
        raise ValueError("Each home must have a unique property_id.")

    scored = _annualized_energy(homes, monthly_energy)
    scored["energy_efficiency_subscore"] = _category_subscores(
        scored["energy_rating"], ENERGY_RATING_LOOKUP
    )
    scored["consumption_intensity_subscore"] = pd.Series(
        [
            energy_intensity_subscore(value)
            for value in scored["energy_intensity_kwh_m2_year"]
        ],
        index=scored.index,
        dtype="Float64",
    )
    scored["insulation_subscore"] = _category_subscores(
        scored["insulation"], INSULATION_LOOKUP
    )
    scored["heating_subscore"] = _category_subscores(
        scored["heating"], HEATING_LOOKUP
    )
    scored["glazing_subscore"] = _category_subscores(
        scored["glazing"], GLAZING_LOOKUP
    )
    scored["solar_subscore"] = _yes_no_subscores(scored["solar"])
    scored["smart_controls_subscore"] = _yes_no_subscores(
        scored["smart_controls"]
    )

    contribution_columns: list[str] = []
    for _, subscore_column, weight in COMPONENTS:
        contribution_column = f"{subscore_column}_contribution"
        scored[contribution_column] = (
            scored[subscore_column].astype("Float64") * weight / 100
        )
        contribution_columns.append(contribution_column)

    weighted_score = scored[contribution_columns].sum(axis=1, min_count=len(COMPONENTS))
    rounded_score = weighted_score.map(
        lambda value: pd.NA if pd.isna(value) else math.floor(value + 0.5)
    )
    scored["score"] = pd.array(rounded_score, dtype="Int64")
    scored["score_class"] = scored["score"].map(classify_score)
    return scored


def build_key_issues(home: pd.Series) -> str:
    issues: list[str] = []
    intensity = home.get("energy_intensity_kwh_m2_year")

    if pd.notna(intensity) and intensity > 200:
        issues.append(f"High energy use ({intensity:.0f} kWh/m²/year)")
    if home.get("insulation") == "Poor":
        issues.append("Poor insulation")
    elif home.get("insulation") == "Fair":
        issues.append("Fair insulation")
    if home.get("heating_subscore", 0) <= 50:
        issues.append(f"Lower-scoring heating type ({home.get('heating', 'Unknown')})")
    if home.get("glazing") == "Single":
        issues.append("Single glazing")
    if home.get("solar_subscore", 0) == 0:
        issues.append("No solar")
    if home.get("smart_controls_subscore", 0) == 0:
        issues.append("No smart controls")
    if home.get("energy_rating") in {"F", "G"}:
        issues.append(f"Low energy rating ({home.get('energy_rating')})")
    if pd.notna(home.get("age")) and home.get("age") >= 80:
        issues.append("Older home (80+ years)")
    missing_month_count = int(home.get("missing_month_count", 0))
    if missing_month_count > 0:
        issues.append(f"{missing_month_count} missing month(s); usage annualized")

    return "; ".join(issues) if issues else "No specific issue identified"
