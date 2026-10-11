"""Check your own home: a form that runs the SAME score, retrofit and explanation
code on one home that a person types in.

Privacy design:
- No name, address or postcode is asked for.
- Nothing the person types is saved to disk or logged (llm_explain.explain() logs,
  so this module deliberately does not use it).
- The AI explanation is only requested after the person ticks a consent box,
  because the calculated results are sent to an outside AI service.

All numbers still come from scoring.py and retrofit.py. The AI only explains them.
"""
from __future__ import annotations

import json

import pandas as pd

import llm_explain as L
import retrofit as R
import scoring as S

try:
    from dataset import REGIONS
except Exception:  # pragma: no cover
    REGIONS = ["East Midlands", "East of England", "London", "North East", "North West",
               "Northern Ireland", "Scotland", "South East", "South West", "Wales",
               "West Midlands", "Yorkshire and the Humber"]

PROPERTY_TYPES = ["Detached", "Semi-detached", "Terraced", "Flat", "Bungalow"]
RATINGS = list("ABCDEFG")
HEATING = ["Gas boiler", "Electric boiler", "Direct electric", "Air-source heat pump"]
INSULATION = ["Poor", "Fair", "Good", "Excellent"]
GLAZING = ["Single", "Double", "Triple"]
HOME_ID = "YOUR_HOME"

AREA_LIMITS = (15.0, 600.0)
AGE_LIMITS = (0, 400)
KWH_MAX = 250_000.0
VALUE_MAX = 50_000_000.0


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def validate(inp: dict):
    """Return (errors, warnings). Errors block the calculation; warnings do not."""
    errors, warnings = [], []
    area, age = inp.get("floor_area"), inp.get("age")
    if area is None or not (AREA_LIMITS[0] <= area <= AREA_LIMITS[1]):
        errors.append(f"Floor area must be between {AREA_LIMITS[0]:.0f} and {AREA_LIMITS[1]:.0f} m2.")
    if age is None or not (AGE_LIMITS[0] <= age <= AGE_LIMITS[1]):
        errors.append(f"Age must be between {AGE_LIMITS[0]} and {AGE_LIMITS[1]} years.")
    for key, label in (("electricity_kwh", "Electricity"), ("gas_kwh", "Gas")):
        v = _num(inp.get(key))
        if v is not None and not (0 <= v <= KWH_MAX):
            errors.append(f"{label} use must be between 0 and {KWH_MAX:,.0f} kWh a year.")
    v = _num(inp.get("property_value"))
    if v is not None and not (0 < v <= VALUE_MAX):
        errors.append("Property value must be a positive amount (or left as not sure).")
    if errors:
        return errors, warnings

    e, g = _num(inp.get("electricity_kwh")), _num(inp.get("gas_kwh"))
    if e is not None and g is not None and area:
        intensity = (e + g) / area
        if intensity < 20:
            warnings.append(f"That is only {intensity:.0f} kWh per m2 a year, which is very low. Please check your figures.")
        elif intensity > 600:
            warnings.append(f"That is {intensity:.0f} kWh per m2 a year, which is very high. Please check your figures.")
    heating = inp.get("heating")
    if g is not None and heating == "Gas boiler" and g == 0:
        warnings.append("You chose a gas boiler but entered no gas use. Please check.")
    if g is not None and heating not in (None, "Gas boiler") and g > 0:
        warnings.append("You chose non-gas heating but entered gas use (maybe a cooker or hot water). "
                        "Results assume your heating runs on electricity.")
    if inp.get("energy_rating") is None:
        warnings.append("No energy rating given, so the Green Home Score and value range cannot be worked out. "
                        "Your EPC certificate shows it.")
    if e is None or g is None:
        warnings.append("Without both yearly energy figures the score and savings cannot be estimated.")
    return errors, warnings


# ----------------------------------------------------------------------------
# Build tables in the same format as homes.csv / monthly_energy.csv
# ----------------------------------------------------------------------------
def build_tables(inp: dict):
    value = _num(inp.get("property_value"))
    homes = pd.DataFrame([{
        "property_id": HOME_ID, "property_type": inp["property_type"], "floor_area": float(inp["floor_area"]),
        "age": int(inp["age"]), "region": inp["region"], "energy_rating": inp.get("energy_rating") or "Unknown",
        "heating": inp["heating"], "insulation": inp["insulation"], "glazing": inp["glazing"],
        "solar": bool(inp["solar"]), "smart_controls": bool(inp["smart_controls"]),
        "property_value": float(value) if value is not None else float("nan"),
    }])
    e, g = _num(inp.get("electricity_kwh")), _num(inp.get("gas_kwh"))
    rows = [{"property_id": HOME_ID, "month": f"2025-{m:02d}",
             "electricity_kwh": float("nan") if e is None else e / 12,
             "gas_kwh": float("nan") if g is None else g / 12} for m in range(1, 13)]
    return homes, pd.DataFrame(rows)


def evaluate(inp: dict) -> dict:
    """Run the existing score and retrofit code for one home."""
    homes, monthly = build_tables(inp)
    home = homes.iloc[0]
    annual = R.annualise_energy(monthly).set_index("property_id")
    e0, g0 = float(annual.loc[HOME_ID, "elec_kwh"]), float(annual.loc[HOME_ID, "gas_kwh"])
    energy_known = e0 == e0 and g0 == g0
    scored, score_row = None, None
    if inp.get("energy_rating") and energy_known:
        scored = S.score_homes(homes, monthly)
        score_row = scored.iloc[0]
    res = R.analyse(home, e0, g0, None, R.load_emission_factors())
    return {"home": home, "e0": e0, "g0": g0, "scored": scored, "score_row": score_row, "res": res}


# ----------------------------------------------------------------------------
# AI explanation without any logging
# ----------------------------------------------------------------------------
def explain_private(question: str, facts: dict, allow_ai: bool) -> dict:
    """Like llm_explain.explain() but never writes to disk, and only calls the AI with consent."""
    error, source = None, "template"
    if allow_ai and L.llm_configured():
        user = f"QUESTION: {question}\n\nFACTS (the only information you may use):\n{json.dumps(facts, indent=1, default=str)}"
        try:
            answer, source = L.call_llm(L.SYSTEM_PROMPT, user), "llm"
        except Exception as e:  # show the plain template instead
            answer, source, error = L.template_explanation(facts), "template_fallback", str(e)
    else:
        answer = L.template_explanation(facts)
    return {"answer": answer, "source": source, "error": error, "check": L.check_numbers(answer, facts)}


def make_facts(ev: dict, question: str):
    home, e0, g0, res = ev["home"], ev["e0"], ev["g0"], ev["res"]
    recs = [r["retrofit"] for r in res["recommendations"]]
    parsed = L.parse_whatif(question)
    is_whatif = bool(parsed["include"] or parsed["exclude"] or parsed["all"])
    chosen, na = L.resolve_scenario(home, recs, parsed)
    scen = R.analyse(home, e0, g0, chosen, R.load_emission_factors())
    score_info = None
    if ev["scored"] is not None:
        try:
            score_info = L._score_info(ev["scored"], HOME_ID)
        except Exception:
            score_info = None
    label = "what-if scenario" if is_whatif else "recommended package"
    return L.build_facts(home, e0, g0, scen, label, na, score_info)


# ----------------------------------------------------------------------------
# Streamlit section
# ----------------------------------------------------------------------------
def _show_package(st, r: dict):
    mid, lo, hi = r["package"]["mid"], r["package"]["low"], r["package"]["high"]
    a, b, c, d = st.columns(4)
    a.metric("Electricity change", f"{mid['elec_saving']:,.0f} kWh/yr")
    b.metric("Gas change", f"{mid['gas_saving']:,.0f} kWh/yr")
    c.metric("Carbon reduction", f"{mid['carbon_kg'] / 1000:,.2f} t CO2e/yr")
    pot = mid["improvement_potential_pct"]
    d.metric("Improvement potential", "unavailable" if pot is None else f"{pot:.1f}%")
    st.caption(f"Range from low to high impact: electricity {lo['elec_saving']:,.0f} to {hi['elec_saving']:,.0f} kWh, "
               f"gas {lo['gas_saving']:,.0f} to {hi['gas_saving']:,.0f} kWh a year. A negative number means "
               "consumption goes up. These are estimates, not guarantees.")
    v = r["value"]
    if v.get("available") and v["uplift_pct_high"] > 0:
        st.info(f"Estimated potential property-value impact (indicative, not a proven effect): "
                f"+{v['uplift_pct_low']:.1f}% to +{v['uplift_pct_high']:.1f}% "
                f"(GBP {v['value_low']:,.0f} to {v['value_high']:,.0f}).")
    else:
        st.info("Property-value impact: not enough information, or no rating change expected.")


def render():
    import streamlit as st

    st.header("Check your own home")
    st.caption("Enter your home's details to get a Green Home Score and upgrade ideas. Demo only: we do not ask "
               "for your name or address, nothing you enter is saved, and results are estimates, not advice.")

    with st.form("my_home_form"):
        c1, c2, c3 = st.columns(3)
        ptype = c1.selectbox("Property type", PROPERTY_TYPES, key="my_type")
        area = c2.number_input("Floor area (m2)", 0.0, 2000.0, 85.0, step=1.0, key="my_area")
        age = c3.number_input("Age of home (years)", 0, 1000, 40, step=1, key="my_age")
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Region", REGIONS, key="my_region")
        rating = c2.selectbox("Energy rating (from your EPC)", ["Not sure"] + RATINGS, key="my_rating")
        heating = c3.selectbox("Heating", HEATING, key="my_heating")
        c1, c2, c3 = st.columns(3)
        insulation = c1.selectbox("Insulation", INSULATION, index=2, key="my_insulation")
        glazing = c2.selectbox("Windows", GLAZING, index=1, key="my_glazing")
        value = c3.number_input("Approx. property value (GBP, 0 = not sure)", 0, 100_000_000, 0, step=5000, key="my_value")
        c1, c2 = st.columns(2)
        solar = c1.checkbox("Has solar panels", key="my_solar")
        smart = c2.checkbox("Has smart heating controls", key="my_smart")
        c1, c2 = st.columns(2)
        elec = c1.number_input("Electricity use (kWh per year)", 0, 1_000_000, 3200, step=100, key="my_elec")
        elec_unknown = c1.checkbox("Not sure about electricity", key="my_elec_unknown")
        gas = c2.number_input("Gas use (kWh per year, 0 = none)", 0, 1_000_000, 11000, step=100, key="my_gas")
        gas_unknown = c2.checkbox("Not sure about gas", key="my_gas_unknown")
        submitted = st.form_submit_button("Check my home")

    if submitted:
        st.session_state["my_inputs"] = {
            "property_type": ptype, "floor_area": float(area), "age": int(age), "region": region,
            "energy_rating": None if rating == "Not sure" else rating, "heating": heating,
            "insulation": insulation, "glazing": glazing, "solar": solar, "smart_controls": smart,
            "property_value": None if value == 0 else float(value),
            "electricity_kwh": None if elec_unknown else float(elec),
            "gas_kwh": None if gas_unknown else float(gas)}
    inp = st.session_state.get("my_inputs")
    if not inp:
        return

    errors, warnings = validate(inp)
    for e in errors:
        st.error(e)
    if errors:
        return
    for w in warnings:
        st.warning(w)

    ev = evaluate(inp)
    row, res = ev["score_row"], ev["res"]

    st.subheader("Your Green Home Score")
    if row is None:
        st.info("A score needs both an energy rating and your yearly electricity and gas use.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", f"{int(row['score'])}/100")
        m2.metric("Class", row["score_class"])
        m3.metric("Energy use", f"{float(row['energy_intensity_kwh_m2_year']):.0f} kWh/m2/year")
        comp = pd.DataFrame([{"Component": n, "Weight": f"{w}%", "Sub-score": int(row[c]),
                              "Contribution (points)": round(float(row[f"{c}_contribution"]), 2)}
                             for n, c, w in S.COMPONENTS])
        st.bar_chart(comp.set_index("Component")[["Contribution (points)"]], horizontal=True)
        st.dataframe(comp, hide_index=True)
        st.caption("Main issues: " + S.build_key_issues(row))

    st.subheader("Suggested upgrades")
    if not res["recommendations"]:
        st.info("No feasible upgrades found for this home.")
    else:
        singles = {s["retrofit"]: s for s in res.get("individual", [])}
        table = []
        for r in res["recommendations"]:
            s = singles.get(r["retrofit"])
            table.append({"Upgrade": R.LABELS[r["retrofit"]], "Suitability": r["suitability"], "Reason": r["reason"],
                          "Electricity saving (kWh/yr)": None if s is None else round(s["elec_saving"]),
                          "Gas saving (kWh/yr)": None if s is None else round(s["gas_saving"])})
        st.dataframe(pd.DataFrame(table), hide_index=True)
        if res["available"]:
            _show_package(st, res)
        else:
            st.info("Savings, carbon and value need your yearly electricity and gas use.")

    st.subheader("Ask about your results")
    consent = st.checkbox("I agree that my home's details and calculated results are sent to an outside AI "
                          "service to write the explanation. Nothing is saved by this app.", key="my_consent")
    q = st.text_input("Ask a question or a what-if", "What should I improve first?", key="my_question")
    if st.button("Explain", key="my_explain"):
        facts = make_facts(ev, q)
        with st.spinner("Explaining..."):
            out = explain_private(q, facts, allow_ai=consent)
        st.markdown(out["answer"])
        if out["source"] == "template" and not consent:
            st.caption("Plain summary shown. Tick the box above to get an AI explanation.")
        elif out["source"] == "template" and not L.llm_configured():
            st.caption("No AI key is set on this app, so a plain summary is shown.")
        elif out["source"] == "template_fallback":
            st.info(f"The AI service was unavailable, so a plain summary is shown. ({out['error']})")
        if out["source"] == "llm":
            chk = out["check"]
            if chk["grounded"]:
                st.success("Grounding check passed: every number matches the calculated results.")
            else:
                st.error("Grounding check failed. These numbers are not in the calculated results: "
                         + ", ".join(chk["unsupported"]) + ". Do not trust them.")