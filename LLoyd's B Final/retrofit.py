"""Retrofit engine for the Green Home project (Step 3).

Everything here is deterministic code: feasibility rules, per-fuel savings,
carbon, improvement potential and an indicative property-value range.
No AI model produces any number. All outputs are ESTIMATES, not guarantees.
Assumptions are documented in RETROFIT_METHOD.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# ----------------------------------------------------------------------------
# Assumptions (see RETROFIT_METHOD.md)
# ----------------------------------------------------------------------------
LEVELS = {"low": 0, "mid": 1, "high": 2}  # low/mid/high end of each impact range

# (low, mid, high) impacts. Ranges come from the problem statement.
IMPACT = {
    "loft_insulation": (0.12, 0.15, 0.18),    # of space-heating energy
    "double_glazing": (0.08, 0.10, 0.12),     # of space-heating energy
    "smart_thermostat": (0.05, 0.075, 0.10),  # of heating energy use
    "heat_pump": (0.15, 0.20, 0.25),          # of purchased energy
    "solar_panels": (0.20, 0.275, 0.35),      # of grid electricity imports
}
ORDER = ["loft_insulation", "double_glazing", "smart_thermostat", "heat_pump", "solar_panels"]
LABELS = {
    "loft_insulation": "Loft / roof insulation",
    "double_glazing": "Double glazing",
    "smart_thermostat": "Smart thermostat",
    "heat_pump": "Heat pump",
    "solar_panels": "Solar panels",
}

# Share of the heating fuel that is space heating (rest: hot water, cooking, appliances).
HEATING_SHARE = {"gas": 0.80, "elec": 0.60}

RATINGS = list("ABCDEFG")  # A best
UPLIFT_PER_BAND = (0.010, 0.025)  # indicative +1.0% to +2.5% per rating band
REGION_MULT = {"London": 1.10, "South East": 1.10, "East of England": 1.05,
               "Scotland": 0.90, "North East": 0.90, "Northern Ireland": 0.90, "Wales": 0.90}
TYPE_MULT = {"detached": 1.10, "semi-detached": 1.00, "terraced": 1.00,
             "bungalow": 1.00, "flat": 0.90}

DEFAULT_EF = {"electricity": 0.177, "gas": 0.18296}  # fallback only; CSV is the source


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _flag(v) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "t")
    return bool(v)


def _txt(v) -> str:
    return str(v).strip().lower()


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def load_emission_factors() -> dict:
    path = DATA_DIR / "emissions_factors.csv"
    if not path.exists():
        path = HERE / "emissions_factors.csv"
    if path.exists():
        df = pd.read_csv(path)
        return {str(r["fuel"]).strip().lower(): float(r["kg_co2e_per_kwh"]) for _, r in df.iterrows()}
    return dict(DEFAULT_EF)


def annualise_energy(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annual kWh per fuel = mean of the available months x 12.

    Missing months are NOT treated as zero. months_* columns show how many
    months were used so incomplete homes can be flagged. A real 0 stays 0.
    """
    m = monthly.copy()
    m["electricity_kwh"] = pd.to_numeric(m["electricity_kwh"], errors="coerce")
    m["gas_kwh"] = pd.to_numeric(m["gas_kwh"], errors="coerce")
    agg = m.groupby("property_id").agg(
        elec_kwh=("electricity_kwh", "mean"), gas_kwh=("gas_kwh", "mean"),
        elec_months=("electricity_kwh", "count"), gas_months=("gas_kwh", "count"),
    ).reset_index()
    agg["elec_kwh"] = agg["elec_kwh"] * 12
    agg["gas_kwh"] = agg["gas_kwh"] * 12
    return agg


# ----------------------------------------------------------------------------
# Feasibility and ordering
# ----------------------------------------------------------------------------
_LEVELS3 = ["Low", "Medium", "High"]


def _down(s: str) -> str:
    return _LEVELS3[max(0, _LEVELS3.index(s) - 1)]


def recommend(home, elec_kwh: float | None = None) -> list[dict]:
    """Feasible retrofits for one home, in a sensible order, with suitability and reason."""
    heating = _txt(home["heating"])
    ins = _txt(home["insulation"])
    glazing = _txt(home["glazing"])
    ptype = _txt(home["property_type"])
    has_solar = _flag(home["solar"])
    has_smart = _flag(home["smart_controls"])
    is_flat = ptype == "flat"
    recs: list[dict] = []

    # 1. Loft / roof insulation
    if ins in ("poor", "fair"):
        s = "High" if ins == "poor" else "Medium"
        why = f"Insulation is rated {ins.title()}, so heat loss is high."
        if is_flat:
            s = _down(s)
            why += " Flats often have no private loft, so check access and consent."
        recs.append({"retrofit": "loft_insulation", "suitability": s, "reason": why})

    # 2. Double glazing
    if glazing == "single":
        recs.append({"retrofit": "double_glazing", "suitability": "High",
                     "reason": "Single glazing loses a lot of heat."})

    # 3. Smart thermostat
    if not has_smart and heating != "direct electric":
        s = "High" if heating == "gas boiler" else "Medium"
        recs.append({"retrofit": "smart_thermostat", "suitability": s,
                     "reason": "No smart controls fitted; better scheduling cuts heating use."})

    # 4. Heat pump (after insulation, because it interacts with it)
    if heating != "air-source heat pump":
        planned_loft = any(r["retrofit"] == "loft_insulation" for r in recs)
        eff = ins
        if planned_loft:
            eff = {"poor": "fair", "fair": "good"}.get(ins, ins)
        s = {"excellent": "High", "good": "High", "fair": "Medium", "poor": "Low"}.get(eff, "Medium")
        if heating != "gas boiler" and s == "High":
            s = "Medium"  # no gas-to-electric switch, so a smaller carbon case
        why = f"Currently {heating}. "
        if planned_loft:
            why += "Do the loft insulation first so the heat pump can be smaller and work well."
        elif eff in ("poor", "fair"):
            why += "Insulation is weak, so heat pump performance would be limited."
        else:
            why += "Insulation is good enough for a heat pump to work well."
        if is_flat:
            s = _down(s)
            why += " Flats may lack outdoor space for a unit."
        recs.append({"retrofit": "heat_pump", "suitability": s, "reason": why})

    # 5. Solar panels
    if not has_solar:
        high_use = _ok(elec_kwh) and elec_kwh >= 3000
        s = "High" if high_use else "Medium"
        why = "No solar fitted; it cuts grid electricity imports."
        if any(r["retrofit"] == "heat_pump" for r in recs):
            why += " Worth more after a heat pump, because electricity use will rise."
        if is_flat:
            s = "Low"
            why += " Flats often lack their own roof, so check access and consent."
        recs.append({"retrofit": "solar_panels", "suitability": s, "reason": why})

    recs.sort(key=lambda r: ORDER.index(r["retrofit"]))
    return recs


# ----------------------------------------------------------------------------
# Energy model
# ----------------------------------------------------------------------------
def simulate(home, elec_kwh: float, gas_kwh: float, retrofits, level: str = "mid"):
    """Annual electricity and gas AFTER the given retrofits (kWh/year).

    Order of effects, which is how interactions are handled:
      1. Heating-demand cuts (insulation, glazing, thermostat) are multiplied
         together on the same heating energy, not added.
      2. A heat pump then applies its purchased-energy cut to what is left.
      3. Solar cuts grid imports last, on the electricity after the heat pump.
    """
    i = LEVELS[level]
    retrofits = set(retrofits)
    heating = _txt(home["heating"])
    gas_heated = heating == "gas boiler"
    E, G = float(elec_kwh), float(gas_kwh)

    H = HEATING_SHARE["gas"] * G if gas_heated else HEATING_SHARE["elec"] * E
    keep = 1.0
    for r in ("loft_insulation", "double_glazing", "smart_thermostat"):
        if r in retrofits:
            keep *= 1 - IMPACT[r][i]
    cut = H * (1 - keep)
    H_after = H - cut
    if gas_heated:
        G -= cut
    else:
        E -= cut

    if "heat_pump" in retrofits and heating != "air-source heat pump":
        target = IMPACT["heat_pump"][i] * (E + G)  # net purchased-energy reduction
        if gas_heated:
            removed = H_after            # heating gas goes away
            G -= removed
            if removed >= target:
                E += removed - target    # electricity rises; net saving = target
        else:
            E -= min(target, H_after)

    if "solar_panels" in retrofits:
        E -= IMPACT["solar_panels"][i] * E

    return max(E, 0.0), max(G, 0.0)


def savings(home, elec0, gas0, retrofits, level="mid") -> dict:
    e1, g1 = simulate(home, elec0, gas0, retrofits, level)
    return {"elec_saving": elec0 - e1, "gas_saving": gas0 - g1, "elec_after": e1, "gas_after": g1}


def carbon_kg(elec_saving: float, gas_saving: float, ef: dict) -> float:
    """Delta C = sum over fuels of (baseline - after) x emission factor. Positive = reduction."""
    return elec_saving * ef["electricity"] + gas_saving * ef["gas"]


def improvement_potential(e0, g0, e1, g1):
    """100 x (baseline - after) / baseline, purchased energy = grid electricity + gas."""
    if not (_ok(e0) and _ok(g0) and _ok(e1) and _ok(g1)):
        return None
    base = e0 + g0
    if base <= 0:
        return None  # unavailable: baseline energy is zero or missing
    return 100 * (base - (e1 + g1)) / base


def value_range(home, potential_pct) -> dict:
    """Indicative value range from rating bands gained. Not a proven causal effect."""
    rating = str(home["energy_rating"]).strip().upper()
    if potential_pct is None or rating not in RATINGS or not _ok(home.get("property_value")):
        return {"available": False}
    bands = 0 if potential_pct < 5 else 1 if potential_pct < 15 else 2 if potential_pct < 30 else 3
    cur = RATINGS.index(rating)
    after = max(0, cur - bands)
    gained = cur - after
    region = str(home.get("region", "")).strip()
    mult = REGION_MULT.get(region, 1.0) * TYPE_MULT.get(_txt(home["property_type"]), 1.0)
    lo = gained * UPLIFT_PER_BAND[0] * mult
    hi = gained * UPLIFT_PER_BAND[1] * mult
    v = float(home["property_value"])
    return {"available": True, "rating_before": rating, "rating_after": RATINGS[after],
            "uplift_pct_low": lo * 100, "uplift_pct_high": hi * 100,
            "value_before": v, "value_low": v * (1 + lo), "value_high": v * (1 + hi)}


# ----------------------------------------------------------------------------
# One-home analysis and whole-dataset table
# ----------------------------------------------------------------------------
def analyse(home, elec0, gas0, retrofits=None, ef=None) -> dict:
    """Full analysis for one home. retrofits=None uses the recommended set."""
    ef = ef or load_emission_factors()
    recs = recommend(home, elec0)
    chosen = [r["retrofit"] for r in recs] if retrofits is None else list(retrofits)
    out = {"recommendations": recs, "retrofits_used": chosen, "baseline_elec": elec0,
           "baseline_gas": gas0, "available": _ok(elec0) and _ok(gas0)}
    if not out["available"]:
        out["note"] = "Insufficient energy data to estimate savings."
        return out

    singles = []
    for r in chosen:
        s = savings(home, elec0, gas0, [r])
        singles.append({"retrofit": r, "elec_saving": s["elec_saving"], "gas_saving": s["gas_saving"],
                        "carbon_kg": carbon_kg(s["elec_saving"], s["gas_saving"], ef)})
    out["individual"] = singles

    pkg = {}
    for lvl in ("low", "mid", "high"):
        s = savings(home, elec0, gas0, chosen, lvl)
        s["carbon_kg"] = carbon_kg(s["elec_saving"], s["gas_saving"], ef)
        s["improvement_potential_pct"] = improvement_potential(elec0, gas0, s["elec_after"], s["gas_after"])
        pkg[lvl] = s
    out["package"] = pkg
    out["value"] = value_range(home, pkg["mid"]["improvement_potential_pct"])
    return out


def build_all(homes: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """One summary row per home using its recommended package."""
    ef = load_emission_factors()
    annual = annualise_energy(monthly).set_index("property_id")
    rows = []
    for _, h in homes.iterrows():
        pid = h["property_id"]
        if pid in annual.index:
            a = annual.loc[pid]
            e0, g0 = a["elec_kwh"], a["gas_kwh"]
            months = int(min(a["elec_months"], a["gas_months"]))
        else:
            e0 = g0 = float("nan")
            months = 0
        res = analyse(h, e0, g0, None, ef)
        row = {"property_id": pid, "energy_rating": h["energy_rating"], "heating": h["heating"],
               "baseline_elec_kwh": e0, "baseline_gas_kwh": g0, "months_used": months,
               "n_recommendations": len(res["recommendations"]),
               "recommended": ", ".join(LABELS[r["retrofit"]] for r in res["recommendations"])}
        if res["available"]:
            mid = res["package"]["mid"]
            v = res["value"]
            floor = float(h["floor_area"]) if _ok(h.get("floor_area")) else float("nan")
            row.update({
                "kwh_per_m2": (e0 + g0) / floor if floor and floor > 0 else float("nan"),
                "elec_saving_kwh": mid["elec_saving"], "gas_saving_kwh": mid["gas_saving"],
                "carbon_saving_t": mid["carbon_kg"] / 1000,
                "improvement_potential_pct": mid["improvement_potential_pct"],
                "value_uplift_low_pct": v.get("uplift_pct_low"), "value_uplift_high_pct": v.get("uplift_pct_high"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def load_data():
    homes = pd.read_csv(DATA_DIR / "homes.csv")
    monthly = pd.read_csv(DATA_DIR / "monthly_energy.csv")
    return homes, monthly


# ----------------------------------------------------------------------------
# Streamlit section (call retrofit.render() from app.py)
# ----------------------------------------------------------------------------
def render():
    import streamlit as st

    homes, monthly = load_data()
    if "retrofit_all" not in st.session_state:
        st.session_state["retrofit_all"] = build_all(homes, monthly)
    table = st.session_state["retrofit_all"]
    ef = load_emission_factors()
    annual = annualise_energy(monthly).set_index("property_id")

    st.header("Retrofit recommendations")
    st.caption("All figures are estimates from documented rules, not guarantees. "
               "Savings of different retrofits are not additive.")

    pid = st.selectbox("Choose a home", homes["property_id"].tolist(), key="retrofit_home")
    home = homes[homes["property_id"] == pid].iloc[0]
    e0 = annual.loc[pid, "elec_kwh"] if pid in annual.index else float("nan")
    g0 = annual.loc[pid, "gas_kwh"] if pid in annual.index else float("nan")

    st.write(f"**{home['property_type']}**, {home['floor_area']} m2, rating **{home['energy_rating']}**, "
             f"heating: {home['heating']}, insulation: {home['insulation']}, glazing: {home['glazing']}.")
    if not (_ok(e0) and _ok(g0)):
        st.warning("Insufficient energy data to estimate savings for this home.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Baseline electricity", f"{e0:,.0f} kWh/yr")
    c2.metric("Baseline gas", f"{g0:,.0f} kWh/yr")

    res = analyse(home, e0, g0, None, ef)
    if not res["recommendations"]:
        st.info("No feasible retrofits found for this home.")
        return
    singles = {s["retrofit"]: s for s in res["individual"]}
    st.subheader("Recommended retrofits (in order)")
    st.dataframe(pd.DataFrame([{
        "Retrofit": LABELS[r["retrofit"]], "Suitability": r["suitability"], "Reason": r["reason"],
        "Electricity saving (kWh/yr)": round(singles[r["retrofit"]]["elec_saving"]),
        "Gas saving (kWh/yr)": round(singles[r["retrofit"]]["gas_saving"]),
        "Carbon (kg CO2e/yr)": round(singles[r["retrofit"]]["carbon_kg"]),
    } for r in res["recommendations"]]), use_container_width=True, hide_index=True)
    st.caption("Negative saving = consumption goes up (for example electricity after a heat pump). "
               "Each row is the retrofit on its own against the baseline.")

    def show_package(r):
        mid, lo, hi = r["package"]["mid"], r["package"]["low"], r["package"]["high"]
        a, b, c, d = st.columns(4)
        a.metric("Electricity change", f"{mid['elec_saving']:,.0f} kWh/yr")
        b.metric("Gas change", f"{mid['gas_saving']:,.0f} kWh/yr")
        c.metric("Carbon reduction", f"{mid['carbon_kg'] / 1000:,.2f} t CO2e/yr")
        pot = mid["improvement_potential_pct"]
        d.metric("Improvement potential", "unavailable" if pot is None else f"{pot:.1f}%")
        st.caption(f"Package range (low to high impact): electricity {lo['elec_saving']:,.0f} to "
                   f"{hi['elec_saving']:,.0f} kWh, gas {lo['gas_saving']:,.0f} to {hi['gas_saving']:,.0f} kWh. "
                   "Interactions between retrofits are included, so the package is not the sum of the rows.")
        v = r["value"]
        if v.get("available") and v["uplift_pct_high"] > 0:
            st.info(f"Estimated potential property-value impact (indicative, not a proven causal effect): "
                    f"+{v['uplift_pct_low']:.1f}% to +{v['uplift_pct_high']:.1f}% "
                    f"(GBP {v['value_low']:,.0f} to {v['value_high']:,.0f}), "
                    f"rating {v['rating_before']} to about {v['rating_after']}.")
        else:
            st.info("No rating change expected, or not enough information to estimate a property-value impact.")

    st.subheader("Full recommended package")
    show_package(res)

    st.subheader("What-if: choose your own package")
    options = list(LABELS)
    pick = st.multiselect("Retrofits to include", options, default=res["retrofits_used"],
                          format_func=lambda k: LABELS[k], key="retrofit_whatif")
    feasible = {r["retrofit"] for r in res["recommendations"]}
    for k in pick:
        if k not in feasible:
            st.warning(f"{LABELS[k]} is not recommended for this home; shown for comparison only.")
    if pick:
        show_package(analyse(home, e0, g0, pick, ef))

    st.subheader("Homes ranked by improvement potential")
    only_low = st.checkbox("Only energy ratings E, F or G (proxy for Poor / Very poor)", value=True,
                           key="retrofit_filter")
    t = table.dropna(subset=["improvement_potential_pct"])
    if only_low:
        t = t[t["energy_rating"].isin(["E", "F", "G"])]
    t = t.sort_values("improvement_potential_pct", ascending=False).head(50)
    st.dataframe(t[["property_id", "energy_rating", "heating", "kwh_per_m2", "improvement_potential_pct",
                    "elec_saving_kwh", "gas_saving_kwh", "carbon_saving_t", "recommended"]].round(1),
                 use_container_width=True, hide_index=True)
    st.caption("Ranking uses percentage reduction in purchased energy, so large homes are not "
               "ranked high just because they use more energy.")