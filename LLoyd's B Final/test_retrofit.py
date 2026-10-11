"""Simple tests for the retrofit engine. Run:  python test_retrofit.py"""
import pandas as pd

import retrofit as R

EF = {"electricity": 0.177, "gas": 0.18296}


def home(**kw):
    base = dict(property_id="T1", property_type="Semi-detached", floor_area=90, age=60,
                region="South East", energy_rating="E", heating="Gas boiler", insulation="Poor",
                glazing="Single", solar=False, smart_controls=False, property_value=300000)
    base.update(kw)
    return pd.Series(base)


def test_heat_pump_raises_electricity_and_cuts_gas():
    h = home(insulation="Good", glazing="Double")
    s = R.savings(h, 3000, 12000, ["heat_pump"])
    assert s["gas_saving"] > 0, "gas should fall"
    assert s["elec_saving"] < 0, "electricity should rise (negative saving)"
    # net purchased-energy cut equals the catalogue percentage (20% mid)
    net = s["elec_saving"] + s["gas_saving"]
    assert abs(net - 0.20 * 15000) < 1


def test_no_infeasible_recommendations():
    names = lambda h: {r["retrofit"] for r in R.recommend(h, 4000)}
    assert "solar_panels" not in names(home(solar=True))
    assert "double_glazing" not in names(home(glazing="Double"))
    assert "double_glazing" not in names(home(glazing="Triple"))
    assert "loft_insulation" not in names(home(insulation="Excellent"))
    assert "heat_pump" not in names(home(heating="Air-source heat pump"))
    assert "smart_thermostat" not in names(home(smart_controls=True))


def test_insulation_before_heat_pump():
    order = [r["retrofit"] for r in R.recommend(home(), 4000)]
    assert order.index("loft_insulation") < order.index("heat_pump")


def test_carbon_matches_per_fuel_formula():
    h = home()
    s = R.savings(h, 3000, 12000, ["loft_insulation", "heat_pump"])
    expected = s["elec_saving"] * 0.177 + s["gas_saving"] * 0.18296
    assert abs(R.carbon_kg(s["elec_saving"], s["gas_saving"], EF) - expected) < 1e-9


def test_package_accounts_for_interactions():
    h = home()
    trio = ["loft_insulation", "double_glazing", "smart_thermostat"]
    singles = sum(R.savings(h, 3000, 12000, [r])["gas_saving"] for r in trio)
    package = R.savings(h, 3000, 12000, trio)["gas_saving"]
    assert package < singles, "package must be less than the simple sum"


def test_low_mid_high_ordering():
    h = home()
    lo = R.savings(h, 3000, 12000, ["loft_insulation"], "low")["gas_saving"]
    mid = R.savings(h, 3000, 12000, ["loft_insulation"], "mid")["gas_saving"]
    hi = R.savings(h, 3000, 12000, ["loft_insulation"], "high")["gas_saving"]
    assert lo < mid < hi


def test_improvement_potential_unavailable_when_baseline_zero_or_missing():
    assert R.improvement_potential(0, 0, 0, 0) is None
    assert R.improvement_potential(float("nan"), 100, 50, 50) is None
    assert abs(R.improvement_potential(4800, 12000, 4080, 10200) - 15.0) < 1e-9


def test_annualise_does_not_treat_missing_months_as_zero():
    months = [{"property_id": "A", "month": f"2025-{m:02d}", "electricity_kwh": 100.0, "gas_kwh": 50.0}
              for m in range(1, 7)]
    out = R.annualise_energy(pd.DataFrame(months)).iloc[0]
    assert abs(out["elec_kwh"] - 1200) < 1e-9 and out["elec_months"] == 6


def test_missing_energy_gives_no_estimate():
    res = R.analyse(home(), float("nan"), float("nan"), None, EF)
    assert res["available"] is False


def test_value_range_is_a_range_and_labelled_by_rating():
    v = R.value_range(home(energy_rating="E"), 30.0)
    assert v["available"] and v["uplift_pct_high"] > v["uplift_pct_low"] > 0
    assert R.value_range(home(), 2.0)["uplift_pct_high"] == 0  # too small to change band


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")