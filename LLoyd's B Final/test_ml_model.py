"""Tests for the ML pipeline. Run: python test_ml_model.py"""
import numpy as np
import pandas as pd

import ml_model as M


def synthetic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    homes = pd.DataFrame({
        "property_id": [f"P{i}" for i in range(n)],
        "property_type": rng.choice(["Flat", "Detached", "Terraced"], n),
        "floor_area": rng.uniform(35, 180, n).round(1), "age": rng.integers(1, 90, n),
        "region": rng.choice(["Scotland", "South East"], n),
        "energy_rating": rng.choice(list("ABCDEFG"), n),
        "heating": rng.choice(["Gas boiler", "Direct electric"], n),
        "insulation": rng.choice(["Poor", "Good"], n), "glazing": rng.choice(["Single", "Double"], n),
        "solar": rng.integers(0, 2, n).astype(bool), "smart_controls": rng.integers(0, 2, n).astype(bool),
        "property_value": 200000})
    per_m2 = 80 + 90 * (homes.insulation == "Poor") + 40 * (homes.glazing == "Single") + homes.age * 0.5
    total = per_m2 * homes.floor_area * rng.normal(1, 0.05, n)
    rows = []
    for pid, t, h in zip(homes.property_id, total, homes.heating):
        for m in range(12):
            rows.append({"property_id": pid, "month": f"2025-{m+1:02d}",
                         "electricity_kwh": t * 0.3 / 12, "gas_kwh": t * 0.7 / 12 if h == "Gas boiler" else 0.0})
    return homes, pd.DataFrame(rows)


def test_splits_are_disjoint_and_complete():
    ids = [f"P{i}" for i in range(100)]
    a, b, c = M.split_by_property(ids)
    assert not (a & b) and not (a & c) and not (b & c)
    assert a | b | c == set(ids)


def test_split_is_reproducible():
    ids = [f"P{i}" for i in range(100)]
    assert M.split_by_property(ids) == M.split_by_property(ids)


def test_features_exclude_leaky_fields():
    for bad in ("property_value", "elec_kwh", "gas_kwh", "purchased_kwh", "kwh_per_m2", "property_id"):
        assert bad not in M.FEATURES


def test_missing_months_not_zero_filled():
    homes, monthly = synthetic(20)
    monthly.loc[monthly.property_id == "P0", "electricity_kwh"] = np.nan
    monthly.loc[monthly.property_id == "P0", "gas_kwh"] = np.nan
    df = M.prepare(homes, monthly)
    assert "P0" not in set(df.property_id)  # dropped, not treated as zero


def test_model_beats_baseline_on_validation():
    homes, monthly = synthetic()
    r = M.train(homes, monthly, save=False)
    assert r["model"]["validation"]["mae_kwh_per_m2"] < r["baseline"]["validation"]["mae_kwh_per_m2"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")