"""Tests for the LLM layer that need no API key. Run: python test_llm_explain.py"""
import pandas as pd

import llm_explain as L
import retrofit as R


def home(**kw):
    b = dict(property_id="T1", property_type="Semi-detached", floor_area=90, age=60, region="South East",
             energy_rating="E", heating="Gas boiler", insulation="Poor", glazing="Single",
             solar=False, smart_controls=False, property_value=300000)
    b.update(kw)
    return pd.Series(b)


def facts_for(h, chosen=None):
    recs = [r["retrofit"] for r in R.recommend(h, 3000)]
    res = R.analyse(h, 3000, 12000, chosen if chosen is not None else recs, {"electricity": .177, "gas": .18296})
    return L.build_facts(h, 3000, 12000, res), res


def test_parse_whatif():
    p = L.parse_whatif("What if I only install solar and loft insulation?")
    assert p["include"] == {"solar_panels", "loft_insulation"}
    p = L.parse_whatif("Do everything except the heat pump")
    assert p["all"] and p["exclude"] == {"heat_pump"} and "heat_pump" not in p["include"]
    p = L.parse_whatif("What about solar without insulation?")
    assert "solar_panels" in p["include"] and "loft_insulation" in p["exclude"]


def test_infeasible_requests_are_dropped_with_reason():
    h = home(solar=True, glazing="Double")
    recs = [r["retrofit"] for r in R.recommend(h, 3000)]
    chosen, na = L.resolve_scenario(h, recs, L.parse_whatif("what if I add solar and double glazing"))
    assert chosen == [] and {n["retrofit"] for n in na} == {"Solar panels", "Double glazing"}
    assert all(n["reason"] for n in na)


def test_grounding_check_catches_invented_numbers():
    f, _ = facts_for(home())
    g = f["package_estimate"]["central"]["gas_saving_kwh_per_year"]
    good = f"You could save about {g:,} kWh of gas per year (an estimate)."
    assert L.check_numbers(good, f)["grounded"]
    bad = "You could save 9,876 kWh of gas per year."
    c = L.check_numbers(bad, f)
    assert not c["grounded"] and "9,876" in c["unsupported"]


def test_grounding_check_accepts_rounding_at_displayed_precision():
    f = {"x": 2771.2, "t": 0.3197}
    assert L.check_numbers("About 2,771 kWh and 0.32 tonnes", f)["grounded"]
    assert not L.check_numbers("About 2,800 kWh", f)["grounded"]


def test_facts_report_missing_data_instead_of_numbers():
    h = home()
    res = R.analyse(h, float("nan"), float("nan"), None, {"electricity": .177, "gas": .18296})
    f = L.build_facts(h, float("nan"), float("nan"), res)
    assert "data_status" in f and "package_estimate" not in f


def test_heat_pump_negative_electricity_is_in_facts():
    f, _ = facts_for(home(insulation="Good", glazing="Double"), ["heat_pump"])
    assert f["package_estimate"]["central"]["electricity_saving_kwh_per_year"] < 0


def test_template_fallback_runs_without_llm():
    f, _ = facts_for(home())
    assert "estimates" in L.template_explanation(f)


def test_explain_with_mock_llm_flags_hallucination():
    f, _ = facts_for(home())
    L.call_llm = lambda system, user: "You will save 123,456 kWh per year."
    out = L.explain("how much?", f)
    assert out["source"] == "llm" and not out["check"]["grounded"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")