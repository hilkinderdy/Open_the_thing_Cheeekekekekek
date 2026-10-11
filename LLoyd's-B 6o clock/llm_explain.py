"""Grounded LLM explanations and what-if answers (Step 5).

Core rule: the LLM never calculates anything. Code computes every number
(retrofit.py). This module (1) packages those results as a "facts" object,
(2) asks the LLM to explain ONLY those facts, (3) checks that every number in
the answer appears in the facts, and (4) logs everything.

What-if flow: parse the request with plain rules -> drop infeasible retrofits
-> RERUN the savings / carbon / value calculations -> pass the new results to
the LLM to explain.

Setup (Replit Secrets):  LLM_PROVIDER = gemini | anthropic
                         LLM_API_KEY  = your key
                         LLM_MODEL    = optional model name override
"""
from __future__ import annotations

import hashlib
import json
import pandas as pd
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import retrofit as R

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "reports"

DEFAULT_MODELS = {"gemini": "gemini-2.0-flash", "anthropic": "claude-haiku-5-5"}

SYSTEM_PROMPT = """You explain home energy results to a homeowner in plain, friendly English.

STRICT RULES
1. Use ONLY the numbers and facts in the FACTS JSON. Never calculate, estimate, round to a different value, convert units, or invent any figure. If you need a number that is not in FACTS, say it is not available.
2. Copy numbers exactly as given (you may add thousands separators).
3. A negative saving means consumption goes UP. Say so plainly (for example, a heat pump can raise electricity use while cutting gas use).
4. Call every saving, carbon and property-value figure an estimate, not a guarantee. The property-value figure is an indicative range, not a proven effect.
5. Only discuss retrofits that appear in FACTS. If FACTS lists a requested retrofit under not_applicable, explain the reason given and do not give savings for it.
6. If FACTS says data is missing or insufficient, say what cannot be estimated. Do not guess.
7. Do not give financial, legal or investment advice. Do not promise bill savings in money.
8. Keep it to about 120-180 words. Short paragraphs or a few bullets. End with one clear next step."""

ALL_RETROFITS = list(R.LABELS)

KEYWORDS = {
    "loft_insulation": ["insulation", "insulate", "loft", "roof"],
    "double_glazing": ["glazing", "windows", "window"],
    "smart_thermostat": ["thermostat", "smart control", "smart"],
    "heat_pump": ["heat pump", "heatpump"],
    "solar_panels": ["solar", "panels", "photovoltaic"],
}
NEGATORS = r"(?:without|skip|skipping|except|excluding|exclude|not|no|minus|don't|dont|rather than|instead of)"


# ----------------------------------------------------------------------------
# What-if parsing (rules, not the LLM)
# ----------------------------------------------------------------------------
def parse_whatif(text: str) -> dict:
    t = text.lower()
    include, exclude = set(), set()
    for r, words in KEYWORDS.items():
        for w in words:
            for m in re.finditer(re.escape(w), t):
                before = t[max(0, m.start() - 30):m.start()]
                if re.search(NEGATORS + r"\W+(?:\w+\W+){0,3}$", before):
                    exclude.add(r)
                else:
                    include.add(r)
    include -= exclude
    everything = bool(re.search(r"\b(everything|all of them|all retrofits|full package|all upgrades)\b", t))
    return {"include": include, "exclude": exclude, "all": everything}


def infeasible_reason(home, retrofit: str):
    heating, ins, gl = R._txt(home["heating"]), R._txt(home["insulation"]), R._txt(home["glazing"])
    if retrofit == "solar_panels" and R._flag(home["solar"]):
        return "this home already has solar panels"
    if retrofit == "double_glazing" and gl in ("double", "triple"):
        return f"this home already has {gl} glazing"
    if retrofit == "loft_insulation" and ins in ("good", "excellent"):
        return f"insulation is already rated {ins}"
    if retrofit == "heat_pump" and heating == "air-source heat pump":
        return "this home already has a heat pump"
    if retrofit == "smart_thermostat":
        if R._flag(home["smart_controls"]):
            return "this home already has smart controls"
        if heating == "direct electric":
            return "a smart thermostat is not suitable for direct electric heating"
    return None


def resolve_scenario(home, recommended: list[str], parsed: dict | None):
    """Return (retrofits_to_simulate, not_applicable[{retrofit, reason}])."""
    if parsed is None or not (parsed["include"] or parsed["exclude"] or parsed["all"]):
        return list(recommended), []
    if parsed["include"]:
        wanted = set(parsed["include"])
    elif parsed["all"]:
        wanted = set(ALL_RETROFITS)
    else:
        wanted = set(recommended)
    wanted -= parsed["exclude"]
    chosen, na = [], []
    for r in ALL_RETROFITS:
        if r in wanted:
            reason = infeasible_reason(home, r)
            if reason or r not in recommended:
                na.append({"retrofit": R.LABELS[r], "reason": reason or "not recommended for this home"})
            else:
                chosen.append(r)
    return chosen, na


# ----------------------------------------------------------------------------
# Facts (everything the LLM may use)
# ----------------------------------------------------------------------------
def _i(x):
    return None if x is None else int(round(x))


def build_facts(home, e0, g0, res: dict, scenario_label: str = "recommended package",
                not_applicable=None, score_info: dict | None = None) -> dict:
    facts = {
        "property_id": str(home["property_id"]),
        "property": {"type": home["property_type"], "floor_area_m2": float(home["floor_area"]),
                     "age_years": int(home["age"]), "region": home["region"],
                     "energy_rating": home["energy_rating"], "heating": home["heating"],
                     "insulation": home["insulation"], "glazing": home["glazing"],
                     "has_solar": R._flag(home["solar"]), "has_smart_controls": R._flag(home["smart_controls"])},
        "scenario": scenario_label,
        "assumed_impact_ranges_percent": {R.LABELS[k]: [round(x * 100, 1) for x in (v[0], v[2])]
                                          for k, v in R.IMPACT.items()},
        "not_applicable": not_applicable or [],
        "notes": ["All savings, carbon and value figures are estimates, not guarantees."],
    }
    if score_info:
        facts["green_home_score"] = score_info
    if not res.get("available"):
        facts["data_status"] = "Insufficient energy data: savings, carbon and value cannot be estimated."
        facts["recommended_retrofits"] = [{"retrofit": R.LABELS[r["retrofit"]], "suitability": r["suitability"],
                                           "reason": r["reason"]} for r in res.get("recommendations", [])]
        return facts

    facts["baseline"] = {"electricity_kwh_per_year": _i(e0), "gas_kwh_per_year": _i(g0),
                         "kwh_per_m2_per_year": round((e0 + g0) / float(home["floor_area"]), 1)}
    suit = {r["retrofit"]: r for r in res["recommendations"]}
    facts["retrofits_in_scenario"] = []
    for s in res.get("individual", []):
        rec = suit.get(s["retrofit"], {})
        facts["retrofits_in_scenario"].append({
            "retrofit": R.LABELS[s["retrofit"]], "suitability": rec.get("suitability"),
            "reason": rec.get("reason"),
            "electricity_saving_kwh_per_year_alone": _i(s["elec_saving"]),
            "gas_saving_kwh_per_year_alone": _i(s["gas_saving"]),
            "carbon_saving_kg_co2e_per_year_alone": _i(s["carbon_kg"]),
        })
    pkg = {}
    for lvl in ("low", "mid", "high"):
        p = res["package"][lvl]
        pot = p["improvement_potential_pct"]
        pkg[lvl] = {"electricity_saving_kwh_per_year": _i(p["elec_saving"]),
                    "gas_saving_kwh_per_year": _i(p["gas_saving"]),
                    "carbon_saving_kg_co2e_per_year": _i(p["carbon_kg"]),
                    "carbon_saving_tonnes_co2e_per_year": round(p["carbon_kg"] / 1000, 2),
                    "improvement_potential_percent": None if pot is None else round(pot, 1)}
    facts["package_estimate"] = {"central": pkg["mid"], "low_impact_case": pkg["low"], "high_impact_case": pkg["high"],
                                 "note": "Package savings include interactions, so they are not the sum of single retrofits."}
    v = res["value"]
    if v.get("available") and v["uplift_pct_high"] > 0:
        facts["property_value_estimate"] = {
            "indicative_uplift_percent_range": [round(v["uplift_pct_low"], 1), round(v["uplift_pct_high"], 1)],
            "value_range_gbp": [_i(v["value_low"]), _i(v["value_high"])], "current_value_gbp": _i(v["value_before"]),
            "rating_before": v["rating_before"], "rating_after_approx": v["rating_after"],
            "note": "Indicative estimate based on assumptions, not a proven causal effect."}
    else:
        facts["property_value_estimate"] = "No rating change expected or not enough information to estimate."
    return facts


# ----------------------------------------------------------------------------
# Grounding check: every number in the answer must come from FACTS
# ----------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _flatten(obj, out):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            try:
                out.append(float(m.group().replace(",", "")))
            except ValueError:
                pass


def check_numbers(answer: str, facts: dict) -> dict:
    """Return numbers in the answer that are not supported by FACTS (rounding allowed)."""
    allowed = []
    _flatten(facts, allowed)
    allowed = allowed + [abs(a) for a in allowed] + [a / 1000 for a in allowed if abs(a) >= 1000]
    unsupported, total = [], 0
    for m in _NUM.finditer(answer):
        raw = m.group()
        txt = raw.replace(",", "")
        try:
            x = abs(float(txt))
        except ValueError:
            continue
        if "." not in txt and x <= 10:
            continue  # small whole numbers (list counts, "3 retrofits") are not treated as claims
        total += 1
        decimals = len(txt.split(".")[1]) if "." in txt else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(x - a) <= tol for a in allowed):
            unsupported.append(raw)
    return {"numbers_checked": total, "unsupported": unsupported, "grounded": not unsupported}


# ----------------------------------------------------------------------------
# LLM call (plain HTTPS, no extra libraries)
# ----------------------------------------------------------------------------
def llm_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM API error {e.code}: {e.read().decode()[:300]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach the LLM API: {e.reason}") from None


def call_llm(system: str, user: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (add it in Replit Secrets).")
    if provider == "gemini":
        data = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                     {"x-goog-api-key": key},
                     {"systemInstruction": {"parts": [{"text": system}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"temperature": 0, "maxOutputTokens": 4000}})
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    if provider == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": key, "anthropic-version": "2023-06-01"},
                     {"model": model, "max_tokens": 600, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use gemini or anthropic.")


def template_explanation(facts: dict) -> str:
    """Plain fallback with no LLM, used if the API fails or no key is set."""
    p = facts["property"]
    lines = [f"Home {facts['property_id']}: a {p['type']} rated {p['energy_rating']} with {p['heating']} heating."]
    if "data_status" in facts:
        return lines[0] + " " + facts["data_status"]
    c = facts["package_estimate"]["central"]
    names = ", ".join(r["retrofit"] for r in facts["retrofits_in_scenario"]) or "none"
    lines.append(f"Scenario ({facts['scenario']}): {names}.")
    lines.append(f"Estimated change: electricity {c['electricity_saving_kwh_per_year']:,} kWh/year, "
                 f"gas {c['gas_saving_kwh_per_year']:,} kWh/year (negative means an increase), "
                 f"carbon {c['carbon_saving_tonnes_co2e_per_year']} tonnes CO2e/year.")
    for n in facts["not_applicable"]:
        lines.append(f"Not applicable: {n['retrofit']} ({n['reason']}).")
    lines.append("These are estimates, not guarantees.")
    return "\n".join(lines)


def explain(question: str, facts: dict, use_cache: dict | None = None) -> dict:
    """Ask the LLM to answer using FACTS only; check and log the answer."""
    facts_json = json.dumps(facts, indent=1, default=str)
    user = f"QUESTION: {question}\n\nFACTS (the only information you may use):\n{facts_json}"
    key = hashlib.sha1(user.encode()).hexdigest()
    if use_cache is not None and key in use_cache:
        return use_cache[key]
    source, error = "llm", None
    try:
        answer = call_llm(SYSTEM_PROMPT, user)
    except Exception as e:  # graceful failure: show the template instead
        answer, source, error = template_explanation(facts), "template_fallback", str(e)
    check = check_numbers(answer, facts)
    out = {"answer": answer, "source": source, "error": error, "check": check, "question": question}
    try:
        REPORT_DIR.mkdir(exist_ok=True)
        with open(REPORT_DIR / "llm_log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "property_id": facts["property_id"],
                                "question": question, "source": source, "error": error,
                                "answer": answer, "unsupported_numbers": check["unsupported"]}) + "\n")
    except OSError:
        pass
    if use_cache is not None:
        use_cache[key] = out
    return out


# ----------------------------------------------------------------------------
# Automated evaluation (a proxy check, not a human review)
# ----------------------------------------------------------------------------
EVAL_QUESTIONS = [
    "Why is my home rated like this and what should I improve first?",
    "What if I only did the loft insulation?",
    "What if I installed solar panels?",
    "What if I add a heat pump? Will my electricity use go up?",
    "How much carbon could I save with the full package?",
    "Could these upgrades increase my property value?",
    "What if I do everything except the heat pump?",
]


def run_eval(homes, annual, n_homes: int = 5, seed: int = 7, pause: float = 1.0) -> dict:
    sample = homes.sample(n=min(n_homes, len(homes)), random_state=seed)
    ef = R.load_emission_factors()
    rows = []
    for _, h in sample.iterrows():
        pid = h["property_id"]
        e0 = annual.loc[pid, "elec_kwh"] if pid in annual.index else float("nan")
        g0 = annual.loc[pid, "gas_kwh"] if pid in annual.index else float("nan")
        recs = [r["retrofit"] for r in R.recommend(h, e0)]
        for q in EVAL_QUESTIONS:
            chosen, na = resolve_scenario(h, recs, parse_whatif(q))
            res = R.analyse(h, e0, g0, chosen, ef)
            facts = build_facts(h, e0, g0, res, "what-if: " + q, na)
            out = explain(q, facts)
            ans = out["answer"].lower()
            rows.append({"property_id": pid, "question": q, "source": out["source"],
                         "grounded": out["check"]["grounded"], "numbers": out["check"]["numbers_checked"],
                         "unsupported": len(out["check"]["unsupported"]),
                         "labels_estimate": ("estimate" in ans) or ("estimated" in ans),
                         "handles_not_applicable": (not na) or any(w in ans for w in ("already", "not suitable", "not applicable", "not recommended"))})
            if out["source"] == "llm":
                time.sleep(pause)
    n = len(rows)
    llm_rows = [r for r in rows if r["source"] == "llm"]
    total_nums = sum(r["numbers"] for r in rows) or 1
    summary = {
        "answers": n, "answers_from_llm": len(llm_rows),
        "groundedness_pct": round(100 * sum(r["grounded"] for r in rows) / max(n, 1), 1),
        "unsupported_number_pct": round(100 * sum(r["unsupported"] for r in rows) / total_nums, 1),
        "labelled_as_estimate_pct": round(100 * sum(r["labels_estimate"] for r in rows) / max(n, 1), 1),
        "not_applicable_handled_pct": round(100 * sum(r["handles_not_applicable"] for r in rows) / max(n, 1), 1),
        "rows": rows,
    }
    try:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "llm_eval.json").write_text(json.dumps(summary, indent=2))
        (REPORT_DIR / "llm_eval.md").write_text(_eval_md(summary))
    except OSError:
        pass
    return summary


def _eval_md(s: dict) -> str:
    return f"""# LLM Explanation Evaluation (automated proxy)

Setup: {s['answers']} answers over sampled homes and {len(EVAL_QUESTIONS)} fixed questions (explanations and what-ifs). Calculations are rerun in code for every what-if before the LLM explains them. {s['answers_from_llm']} answers came from the LLM; the rest used the template fallback.

| Metric | Result | Target |
|---|---|---|
| Groundedness (answers whose numbers all match the computed facts) | {s['groundedness_pct']}% | 90% or higher |
| Unsupported numbers (share of numbers not in the facts) | {s['unsupported_number_pct']}% | under 5% |
| Answers labelling figures as estimates | {s['labelled_as_estimate_pct']}% | 100% |
| Infeasible retrofit requests handled in the answer | {s['not_applicable_handled_pct']}% | 100% |

## Limits of this evaluation
- It checks numbers against the computed facts automatically. It does not check wording, completeness or whether claims without numbers are true; a human review of a sample is still needed.
- Small sample, one model, synthetic data.
- Rounding to the displayed precision is accepted; any other number is flagged.
"""


# ----------------------------------------------------------------------------
# Streamlit section
# ----------------------------------------------------------------------------
def _score_info(table, pid):
    if table is None:
        return None
    try:
        from scoring import COMPONENTS
        row = table.loc[table["property_id"] == pid].iloc[0]
        if pd.isna(row["score"]):
            return None
        info = {"score_out_of_100": int(row["score"]), "class": row["score_class"], "components": []}
        for name, col, weight in COMPONENTS:
            info["components"].append({"component": name, "weight_percent": weight,
                                       "sub_score": int(row[col]),
                                       "contribution_points": round(float(row[f"{col}_contribution"]), 2)})
        return info
    except Exception:
        return None
def render():
    import streamlit as st

    homes, monthly = R.load_data()
    annual = R.annualise_energy(monthly).set_index("property_id")
    ef = R.load_emission_factors()
    st.header("Explain my results (AI)")
    st.caption("The numbers come from the score, retrofit and carbon calculations. The AI only explains them "
               "and cannot change them. Everything is an estimate, not a guarantee.")

    ids = homes["property_id"].tolist()
    default = st.session_state.get("retrofit_home", ids[0])
    pid = st.selectbox("Home", ids, index=ids.index(default) if default in ids else 0, key="llm_home")
    home = homes[homes["property_id"] == pid].iloc[0]
    e0 = annual.loc[pid, "elec_kwh"] if pid in annual.index else float("nan")
    g0 = annual.loc[pid, "gas_kwh"] if pid in annual.index else float("nan")

    if not llm_configured():
        st.warning("No LLM key set, so a plain template explanation is shown. Add LLM_API_KEY (and "
                   "LLM_PROVIDER) in Replit Secrets to turn on AI explanations.")

    q = st.text_input("Ask a question or a what-if",
                      value="Why is my home rated like this and what should I improve first?", key="llm_q")
    st.caption("What-if examples: 'What if I only install solar?', 'Do everything except the heat pump', "
               "'What about a heat pump without insulation?'")
    if st.button("Explain", key="llm_go"):
        recs = [r["retrofit"] for r in R.recommend(home, e0)]
        parsed = parse_whatif(q)
        is_whatif = bool(parsed["include"] or parsed["exclude"] or parsed["all"])
        chosen, na = resolve_scenario(home, recs, parsed)
        res = R.analyse(home, e0, g0, chosen, ef)  # calculations rerun for this scenario
        label = "what-if scenario" if is_whatif else "recommended package"
        facts = build_facts(home, e0, g0, res, label, na, _score_info(st.session_state.get("score_table"), pid))
        cache = st.session_state.setdefault("llm_cache", {})
        with st.spinner("Explaining..."):
            out = explain(q, facts, cache)
        st.markdown(out["answer"])
        if out["source"] != "llm":
            st.info("Showing the plain template because the AI service was unavailable"
                    + (f": {out['error']}" if out["error"] else "."))
        chk = out["check"]
        if chk["grounded"]:
            st.success(f"Grounding check passed: all {chk['numbers_checked']} numbers match the calculated results.")
        else:
            st.error("Grounding check failed. These numbers are not in the calculated results: "
                     + ", ".join(chk["unsupported"]) + ". Do not trust them.")
        with st.expander("Calculated results the AI was given"):
            st.json(facts)

    with st.expander("Evaluate the explanations (runs about 35 AI calls)"):
        n = st.slider("Homes to test", 3, 10, 5, key="llm_eval_n")
        if st.button("Run evaluation", key="llm_eval_go"):
            with st.spinner("Running..."):
                s = run_eval(homes, annual, n_homes=n)
            st.write({k: v for k, v in s.items() if k != "rows"})
            st.dataframe(s["rows"], use_container_width=True)
            st.caption("Saved to reports/llm_eval.md and reports/llm_eval.json")
