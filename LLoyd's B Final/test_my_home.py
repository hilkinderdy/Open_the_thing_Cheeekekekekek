"""Tests for the 'check your own home' form logic (no API key needed). Run: python test_my_home.py"""
import os

import my_home as M
import llm_explain as L


def inp(**kw):
    b = dict(property_type="Semi-detached", floor_area=95.0, age=60, region="Scotland", energy_rating="E",
             heating="Gas boiler", insulation="Poor", glazing="Single", solar=False, smart_controls=False,
             property_value=250000.0, electricity_kwh=3200.0, gas_kwh=13000.0)
    b.update(kw)
    return b


def test_valid_home_gets_score_and_recommendations():
    ev = M.evaluate(inp())
    assert ev["score_row"] is not None and 0 <= int(ev["score_row"]["score"]) <= 100
    assert ev["res"]["available"] and ev["res"]["recommendations"]


def test_poor_home_scores_lower_than_good_home():
    poor = int(M.evaluate(inp())["score_row"]["score"])
    good = int(M.evaluate(inp(energy_rating="A", heating="Air-source heat pump", insulation="Excellent",
                              glazing="Triple", solar=True, smart_controls=True,
                              electricity_kwh=2500.0, gas_kwh=0.0))["score_row"]["score"])
    assert good > poor + 30


def test_unknown_rating_means_no_score_and_no_guess():
    ev = M.evaluate(inp(energy_rating=None))
    assert ev["score_row"] is None
    assert ev["res"]["available"]  # savings still work
    assert not ev["res"]["value"]["available"]  # value needs a rating


def test_unknown_energy_means_insufficient_data():
    ev = M.evaluate(inp(electricity_kwh=None))
    assert ev["score_row"] is None and ev["res"]["available"] is False


def test_validation_blocks_impossible_values_and_warns_on_odd_ones():
    errors, _ = M.validate(inp(floor_area=5.0))
    assert errors
    errors, _ = M.validate(inp(gas_kwh=-1.0))
    assert errors
    errors, warns = M.validate(inp(electricity_kwh=100.0, gas_kwh=100.0))
    assert not errors and any("very low" in w for w in warns)
    errors, warns = M.validate(inp(gas_kwh=0.0))
    assert any("no gas use" in w for w in warns)


def test_no_logging_and_no_ai_without_consent():
    log = L.REPORT_DIR / "llm_log.jsonl"
    before = log.stat().st_size if log.exists() else None
    calls = []
    L.call_llm = lambda s, u: calls.append(1) or "You could save 999,999 kWh."
    ev = M.evaluate(inp())
    facts = M.make_facts(ev, "What should I improve first?")
    out = M.explain_private("What should I improve first?", facts, allow_ai=False)
    assert out["source"] == "template" and not calls
    after = log.stat().st_size if log.exists() else None
    assert before == after, "form must not write to the LLM log"


def test_what_if_reruns_calculations_and_flags_made_up_numbers():
    os.environ["LLM_API_KEY"] = "x"
    L.call_llm = lambda s, u: "You could save 999,999 kWh."
    ev = M.evaluate(inp())
    facts = M.make_facts(ev, "What if I only install solar?")
    assert [r["retrofit"] for r in facts["retrofits_in_scenario"]] == ["Solar panels"]
    out = M.explain_private("What if I only install solar?", facts, allow_ai=True)
    assert out["source"] == "llm" and not out["check"]["grounded"]


def test_already_has_solar_is_reported_not_applicable():
    ev = M.evaluate(inp(solar=True))
    facts = M.make_facts(ev, "What if I add solar panels?")
    assert facts["not_applicable"] and "already" in facts["not_applicable"][0]["reason"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")