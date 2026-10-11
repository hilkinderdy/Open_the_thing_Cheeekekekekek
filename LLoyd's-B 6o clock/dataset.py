from __future__ import annotations

import numpy as np
import pandas as pd


HOME_COUNT = 2_000
DATA_YEAR = 2025
DATASET_VERSION = 3
RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

REGIONS = [
    "East Midlands",
    "East of England",
    "London",
    "North East",
    "North West",
    "Northern Ireland",
    "Scotland",
    "South East",
    "South West",
    "Wales",
    "West Midlands",
    "Yorkshire and the Humber",
]

REGION_WEIGHTS = [
    0.08,
    0.09,
    0.09,
    0.04,
    0.11,
    0.03,
    0.09,
    0.13,
    0.09,
    0.05,
    0.09,
    0.11,
]

CLIMATE_FACTOR = {
    "East Midlands": 1.03,
    "East of England": 0.98,
    "London": 0.92,
    "North East": 1.14,
    "North West": 1.10,
    "Northern Ireland": 1.10,
    "Scotland": 1.17,
    "South East": 0.94,
    "South West": 0.98,
    "Wales": 1.08,
    "West Midlands": 1.02,
    "Yorkshire and the Humber": 1.08,
}

REGION_PRICE_PER_SQM = {
    "East Midlands": 2_450,
    "East of England": 3_750,
    "London": 7_500,
    "North East": 1_850,
    "North West": 2_450,
    "Northern Ireland": 1_950,
    "Scotland": 2_500,
    "South East": 4_400,
    "South West": 3_350,
    "Wales": 2_350,
    "West Midlands": 2_500,
    "Yorkshire and the Humber": 2_250,
}

APPLIANCE_ELECTRICITY_PROFILE = np.array(
    [1.08, 1.06, 1.03, 1.00, 0.98, 0.95, 0.94, 0.95, 0.98, 1.00, 1.08, 1.10],
    dtype=float,
)
APPLIANCE_ELECTRICITY_PROFILE /= APPLIANCE_ELECTRICITY_PROFILE.mean()

ELECTRIC_HEATING_PROFILE = np.array(
    [2.10, 1.95, 1.70, 1.15, 0.58, 0.18, 0.10, 0.15, 0.55, 1.15, 1.75, 2.00],
    dtype=float,
)
ELECTRIC_HEATING_PROFILE /= ELECTRIC_HEATING_PROFILE.mean()

GAS_HEATING_PROFILE = np.array(
    [2.20, 2.00, 1.70, 1.15, 0.58, 0.20, 0.10, 0.12, 0.50, 1.10, 1.85, 2.20],
    dtype=float,
)
GAS_HEATING_PROFILE /= GAS_HEATING_PROFILE.mean()
GAS_BASELOAD_SHARE = 0.18

SOLAR_MONTHLY_OFFSET = np.array(
    [0.04, 0.05, 0.08, 0.12, 0.15, 0.18, 0.18, 0.16, 0.12, 0.08, 0.05, 0.04],
    dtype=float,
)


def _normalized_month_weights(
    seasonal_profile: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
) -> np.ndarray:
    noise = rng.lognormal(
        mean=-0.5 * noise_sigma**2,
        sigma=noise_sigma,
        size=12,
    )
    weights = seasonal_profile * noise
    return weights / weights.sum()


def _age_based_features(
    ages: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    insulation = np.empty(len(ages), dtype=object)
    glazing = np.empty(len(ages), dtype=object)

    for index, age in enumerate(ages):
        if age <= 15:
            insulation[index] = rng.choice(
                ["Poor", "Fair", "Good", "Excellent"],
                p=[0.04, 0.15, 0.48, 0.33],
            )
            glazing[index] = rng.choice(
                ["Single", "Double", "Triple"], p=[0.01, 0.66, 0.33]
            )
        elif age <= 40:
            insulation[index] = rng.choice(
                ["Poor", "Fair", "Good", "Excellent"],
                p=[0.09, 0.25, 0.45, 0.21],
            )
            glazing[index] = rng.choice(
                ["Single", "Double", "Triple"], p=[0.04, 0.80, 0.16]
            )
        elif age <= 75:
            insulation[index] = rng.choice(
                ["Poor", "Fair", "Good", "Excellent"],
                p=[0.23, 0.38, 0.30, 0.09],
            )
            glazing[index] = rng.choice(
                ["Single", "Double", "Triple"], p=[0.15, 0.78, 0.07]
            )
        else:
            insulation[index] = rng.choice(
                ["Poor", "Fair", "Good", "Excellent"],
                p=[0.42, 0.36, 0.18, 0.04],
            )
            glazing[index] = rng.choice(
                ["Single", "Double", "Triple"], p=[0.34, 0.62, 0.04]
            )

    return insulation, glazing


def generate_synthetic_dataset(
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 2,000 fictional UK homes and 12 months of energy use for each."""
    rng = np.random.default_rng(seed)

    property_types = rng.choice(
        ["Detached", "Semi-detached", "Terraced", "Flat", "Bungalow"],
        size=HOME_COUNT,
        p=[0.21, 0.27, 0.23, 0.19, 0.10],
    )
    region = rng.choice(REGIONS, size=HOME_COUNT, p=REGION_WEIGHTS)
    ages = np.clip(rng.gamma(shape=2.0, scale=21.0, size=HOME_COUNT), 0, 150)
    ages = np.rint(ages).astype(int)

    area_parameters = {
        "Detached": (144, 38),
        "Semi-detached": (96, 22),
        "Terraced": (82, 20),
        "Flat": (61, 17),
        "Bungalow": (91, 24),
    }
    floor_area = np.array(
        [
            rng.normal(*area_parameters[property_type])
            for property_type in property_types
        ]
    )
    floor_area = np.round(np.clip(floor_area, 30, 320), 1)

    insulation, glazing = _age_based_features(ages, rng)
    heating = rng.choice(
        [
            "Gas boiler",
            "Direct electric",
            "Air-source heat pump",
            "Electric boiler",
        ],
        size=HOME_COUNT,
        p=[0.66, 0.12, 0.15, 0.07],
    )
    solar = rng.random(HOME_COUNT) < 0.14
    smart_controls = rng.random(HOME_COUNT) < 0.42

    insulation_penalty = pd.Series(insulation).map(
        {"Poor": 28, "Fair": 15, "Good": 6, "Excellent": 0}
    ).to_numpy()
    glazing_penalty = pd.Series(glazing).map(
        {"Single": 19, "Double": 7, "Triple": 0}
    ).to_numpy()
    heating_penalty = pd.Series(heating).map(
        {
            "Gas boiler": 9,
            "Direct electric": 13,
            "Air-source heat pump": 0,
            "Electric boiler": 15,
        }
    ).to_numpy()
    age_penalty = np.minimum(np.maximum(ages - 10, 0) * 0.22, 20)

    rating_score = (
        96
        - insulation_penalty
        - glazing_penalty
        - heating_penalty
        - age_penalty
        + solar.astype(int) * 7
        + smart_controls.astype(int) * 4
        + rng.normal(0, 4, HOME_COUNT)
    )
    energy_rating = np.select(
        [
            rating_score >= 86,
            rating_score >= 75,
            rating_score >= 64,
            rating_score >= 53,
            rating_score >= 42,
            rating_score >= 31,
        ],
        ["A", "B", "C", "D", "E", "F"],
        default="G",
    )

    type_value_factor = pd.Series(property_types).map(
        {
            "Detached": 1.12,
            "Semi-detached": 1.0,
            "Terraced": 0.91,
            "Flat": 0.88,
            "Bungalow": 0.98,
        }
    ).to_numpy()
    regional_value = pd.Series(region).map(REGION_PRICE_PER_SQM).to_numpy()
    property_value = (
        floor_area
        * regional_value
        * type_value_factor
        * rng.lognormal(mean=0, sigma=0.22, size=HOME_COUNT)
    )

    homes = pd.DataFrame(
        {
            "property_id": [f"UKH{number:06d}" for number in range(1, HOME_COUNT + 1)],
            "property_type": property_types,
            "floor_area": floor_area,
            "age": ages,
            "region": region,
            "energy_rating": energy_rating,
            "heating": heating,
            "insulation": insulation,
            "glazing": glazing,
            "solar": solar,
            "smart_controls": smart_controls,
            "property_value": np.rint(property_value).astype(int),
        }
    )

    insulation_load = pd.Series(insulation).map(
        {"Poor": 75, "Fair": 43, "Good": 19, "Excellent": 4}
    ).to_numpy()
    glazing_load = pd.Series(glazing).map(
        {"Single": 43, "Double": 15, "Triple": 3}
    ).to_numpy()
    rating_load = pd.Series(energy_rating).map(
        {"A": -14, "B": -8, "C": 0, "D": 13, "E": 27, "F": 43, "G": 60}
    ).to_numpy()
    age_load = np.minimum(ages, 100) * 0.19
    climate = pd.Series(region).map(CLIMATE_FACTOR).to_numpy()
    control_saving = smart_controls.astype(int)

    appliance_electricity = 27 + floor_area * 0.015
    electric_heating_intensity = np.select(
        [
            heating == "Gas boiler",
            heating == "Direct electric",
            heating == "Air-source heat pump",
            heating == "Electric boiler",
        ],
        [
            17 + insulation_load * 0.08 + glazing_load * 0.06,
            94 + insulation_load + glazing_load * 0.70 + rating_load * 0.55,
            62 + insulation_load * 0.52 + glazing_load * 0.32 + rating_load * 0.40,
            120 + insulation_load * 0.92 + glazing_load * 0.70 + rating_load * 0.62,
        ],
        default=0,
    )
    electric_intensity = (
        appliance_electricity
        + electric_heating_intensity * climate
        + age_load * 0.14
        - control_saving * 4
    )
    solar_electricity_factor = solar.astype(float) * 0.13
    electric_intensity = np.maximum(
        electric_intensity * (1 - solar_electricity_factor), 12
    )

    gas_home = heating == "Gas boiler"
    gas_intensity = np.where(
        gas_home,
        (
            78
            + insulation_load * 0.92
            + glazing_load * 0.62
            + rating_load * 0.57
            + age_load * 0.80
            - control_saving * 8
        )
        * climate,
        0,
    )
    gas_intensity = np.maximum(gas_intensity, 0)

    # Retain the earlier solar-related yearly reduction while changing only its
    # distribution across months.
    previous_electricity_profile = np.array(
        [0.94, 0.91, 0.90, 0.88, 0.90, 0.93, 0.96, 0.96, 0.95, 0.98, 1.08, 1.15],
        dtype=float,
    )
    previous_electricity_profile /= previous_electricity_profile.mean()
    solar_annual_factor = np.average(
        1 - SOLAR_MONTHLY_OFFSET,
        weights=previous_electricity_profile,
    )

    months = pd.date_range(f"{DATA_YEAR}-01-01", periods=12, freq="MS")
    gas_profile = (
        GAS_BASELOAD_SHARE
        + (1 - GAS_BASELOAD_SHARE) * GAS_HEATING_PROFILE
    )
    # Hidden behaviour the features cannot see (occupancy, habits, appliances).
    shared_behaviour = rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15, size=HOME_COUNT)
    elec_behaviour = shared_behaviour * rng.lognormal(mean=-0.5 * 0.10**2, sigma=0.10, size=HOME_COUNT)
    gas_behaviour = shared_behaviour * rng.lognormal(mean=-0.5 * 0.10**2, sigma=0.10, size=HOME_COUNT)
    has_extra_load = rng.random(HOME_COUNT) < 0.12  # e.g. EV charger, hot tub, home working
    extra_electricity = np.where(has_extra_load, rng.uniform(600, 2500, HOME_COUNT), 0.0)

    # Conflicting signals: 20 homes with a good rating (A or B) but unusually high use.
    good_rating = np.flatnonzero(np.isin(energy_rating, ["A", "B"]))
    conflict_idx = rng.choice(good_rating, size=20, replace=False)
    elec_behaviour[conflict_idx] *= 1.8
    gas_behaviour[conflict_idx] *= 1.8

    monthly_rows: list[dict[str, object]] = []

    for home_index, home in homes.iterrows():
        appliance_load = max(
            appliance_electricity[home_index]
            + age_load[home_index] * 0.14
            - control_saving[home_index] * 4,
            8,
        )
        heating_load = max(
            electric_heating_intensity[home_index] * climate[home_index],
            0,
        )

        if heating[home_index] == "Gas boiler":
            electricity_profile = APPLIANCE_ELECTRICITY_PROFILE.copy()
        else:
            combined_load = appliance_load + heating_load
            electricity_profile = (
                appliance_load * APPLIANCE_ELECTRICITY_PROFILE
                + heating_load * ELECTRIC_HEATING_PROFILE
            ) / combined_load

        if solar[home_index]:
            electricity_profile *= 1 - SOLAR_MONTHLY_OFFSET

        electricity_weights = _normalized_month_weights(
            electricity_profile,
            rng,
            noise_sigma=0.055,
        )
        annual_electricity = (
            floor_area[home_index] * electric_intensity[home_index] * elec_behaviour[home_index]
            + extra_electricity[home_index]
        )
        if solar[home_index]:
            annual_electricity *= solar_annual_factor
        month_electricity = annual_electricity * electricity_weights

        gas_weights = _normalized_month_weights(gas_profile, rng, noise_sigma=0.06)
        annual_gas = floor_area[home_index] * gas_intensity[home_index] * gas_behaviour[home_index]
        month_gas = annual_gas * gas_weights

        for month_index, month in enumerate(months):
            monthly_rows.append(
                {
                    "property_id": home["property_id"],
                    "month": month.strftime("%Y-%m"),
                    "electricity_kwh": round(float(month_electricity[month_index]), 1),
                    "gas_kwh": round(float(month_gas[month_index]), 1),
                }
            )

    monthly_energy = pd.DataFrame(monthly_rows)

    # Missing data (documented): gaps are blank values, never zeros.
    ids = homes["property_id"].to_numpy()
    order = rng.permutation(HOME_COUNT)
    no_reading_ids = ids[order[:20]]      # 20 homes with no readings at all
    gap_ids = ids[order[20:80]]           # 60 homes missing 1 to 4 months of both fuels
    month_labels = monthly_energy["month"].unique()

    monthly_energy.loc[
        monthly_energy["property_id"].isin(no_reading_ids), ["electricity_kwh", "gas_kwh"]
    ] = np.nan
    for pid in gap_ids:
        dropped = rng.choice(month_labels, size=int(rng.integers(1, 5)), replace=False)
        mask = (monthly_energy["property_id"] == pid) & monthly_energy["month"].isin(dropped)
        monthly_energy.loc[mask, ["electricity_kwh", "gas_kwh"]] = np.nan

    # 20 gas-boiler homes with a few missing GAS readings only (partial fields).
    gas_pool = ids[(heating == "Gas boiler") & ~np.isin(ids, np.concatenate([no_reading_ids, gap_ids]))]
    for pid in rng.choice(gas_pool, size=20, replace=False):
        dropped = rng.choice(month_labels, size=int(rng.integers(1, 4)), replace=False)
        mask = (monthly_energy["property_id"] == pid) & monthly_energy["month"].isin(dropped)
        monthly_energy.loc[mask, "gas_kwh"] = np.nan

    return homes, monthly_energy
