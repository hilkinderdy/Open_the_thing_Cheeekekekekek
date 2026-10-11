"""ML energy-use model for the Green Home project (Step 4).

Predicts annual purchased energy intensity (electricity + gas, kWh per m2 per year)
from property characteristics, and compares it with a simple baseline.

Method notes:
- Split BY PROPERTY (70/15/15) so no property appears in more than one split.
- Features are limited to fields available at prediction time (no energy readings,
  no sale prices, no property value).
- Hyperparameters are chosen on the validation set only. The test set is used once.
- Data is synthetic and its energy labels come from documented generation rules,
  so good scores here do NOT prove real-world accuracy.

Run:  python ml_model.py        (or press the button in the app)
Outputs: models/energy_model.joblib, reports/metrics.json,
         reports/evaluation_report.md, reports/test_predictions.csv,
         reports/data_splits.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from retrofit import _flag, annualise_energy

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"
REPORT_DIR = HERE / "reports"
SEED = 42

CATEGORICAL = ["property_type", "region", "energy_rating", "heating", "insulation", "glazing"]
NUMERIC = ["floor_area", "age"]
BOOLEAN = ["solar", "smart_controls"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN
TARGET = "kwh_per_m2"

# Small hyperparameter grid, compared on the VALIDATION set only.
GRID = [
    {"learning_rate": 0.05, "max_iter": 200, "max_depth": 3},
    {"learning_rate": 0.05, "max_iter": 400, "max_depth": 4},
    {"learning_rate": 0.10, "max_iter": 200, "max_depth": 4},
    {"learning_rate": 0.10, "max_iter": 300, "max_depth": 6},
]


def prepare(homes: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per home with the target. Homes with missing energy data are dropped,
    never filled with zero."""
    annual = annualise_energy(monthly)
    df = homes.merge(annual, on="property_id", how="inner")
    df = df[(df["elec_months"] > 0) & (df["gas_months"] > 0)]
    df = df[pd.to_numeric(df["floor_area"], errors="coerce") > 0].copy()
    df["purchased_kwh"] = df["elec_kwh"] + df["gas_kwh"]
    df[TARGET] = df["purchased_kwh"] / df["floor_area"]
    for b in BOOLEAN:
        df[b] = df[b].map(lambda v: 1 if _flag(v) else 0)
    return df.reset_index(drop=True)


def split_by_property(ids, seed: int = SEED, frac=(0.70, 0.15, 0.15)):
    """Property-level split: every property id lands in exactly one split."""
    ids = np.array(sorted(set(ids)))
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    return set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va]), set(ids[n_tr + n_va:])


def make_model(learning_rate, max_iter, max_depth) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC + BOOLEAN),
    ])
    gbm = HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=max_iter,
                                        max_depth=max_depth, random_state=SEED)
    return Pipeline([("pre", pre), ("gbm", gbm)])


class GroupMeanBaseline:
    """Simple baseline: average kWh/m2 of homes of the same property type (train only)."""

    def fit(self, df: pd.DataFrame):
        self.by_type = df.groupby("property_type")[TARGET].mean().to_dict()
        self.overall = float(df[TARGET].mean())
        return self

    def predict(self, df: pd.DataFrame):
        return df["property_type"].map(self.by_type).fillna(self.overall).to_numpy()


def metrics(df: pd.DataFrame, pred) -> dict:
    y = df[TARGET].to_numpy()
    yk = y * df["floor_area"].to_numpy()          # back to kWh/year
    pk = np.asarray(pred) * df["floor_area"].to_numpy()
    return {
        "mae_kwh_per_m2": float(mean_absolute_error(y, pred)),
        "rmse_kwh_per_m2": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "mae_kwh_per_year": float(mean_absolute_error(yk, pk)),
        "n": int(len(df)),
    }


def train(homes: pd.DataFrame | None = None, monthly: pd.DataFrame | None = None,
          save: bool = True) -> dict:
    if homes is None:
        homes = pd.read_csv(DATA_DIR / "homes.csv")
    if monthly is None:
        monthly = pd.read_csv(DATA_DIR / "monthly_energy.csv")
    df = prepare(homes, monthly)

    tr_ids, va_ids, te_ids = split_by_property(df["property_id"])
    assert not (tr_ids & va_ids) and not (tr_ids & te_ids) and not (va_ids & te_ids)
    tr = df[df["property_id"].isin(tr_ids)]
    va = df[df["property_id"].isin(va_ids)]
    te = df[df["property_id"].isin(te_ids)]

    baseline = GroupMeanBaseline().fit(tr)
    base_val, base_test = metrics(va, baseline.predict(va)), metrics(te, baseline.predict(te))

    # Choose hyperparameters on validation only.
    best, best_mae, best_model = None, float("inf"), None
    for params in GRID:
        m = make_model(**params).fit(tr[FEATURES], tr[TARGET])
        mae = mean_absolute_error(va[TARGET], m.predict(va[FEATURES]))
        if mae < best_mae:
            best, best_mae, best_model = params, mae, m
    model = best_model
    model_val = metrics(va, model.predict(va[FEATURES]))
    te_pred = model.predict(te[FEATURES])
    model_test = metrics(te, te_pred)  # test set used once, after choosing the model

    imp = permutation_importance(model, va[FEATURES], va[TARGET], n_repeats=5,
                                 random_state=SEED, scoring="neg_mean_absolute_error")
    importance = sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])

    result = {
        "target": "annual purchased energy intensity (electricity + gas), kWh per m2 per year",
        "features": FEATURES, "seed": SEED, "best_params": best,
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
        "baseline": {"description": "mean kWh/m2 by property type, fitted on train only",
                     "validation": base_val, "test": base_test},
        "model": {"description": "HistGradientBoostingRegressor (scikit-learn)",
                  "validation": model_val, "test": model_test},
        "permutation_importance_validation": [{"feature": f, "mae_increase": float(v)} for f, v in importance],
        "homes_dropped_missing_data": int(len(homes) - len(df)),
    }

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        REPORT_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR / "energy_model.joblib")
        (REPORT_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
        out = te[["property_id", "floor_area", TARGET]].rename(columns={TARGET: "actual_kwh_per_m2"}).copy()
        out["model_pred_kwh_per_m2"] = te_pred
        out["baseline_pred_kwh_per_m2"] = baseline.predict(te)
        out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
        splits = pd.DataFrame(
            [(p, "train") for p in sorted(tr_ids)] + [(p, "validation") for p in sorted(va_ids)]
            + [(p, "test") for p in sorted(te_ids)], columns=["property_id", "split"])
        splits.to_csv(REPORT_DIR / "data_splits.csv", index=False)
        (REPORT_DIR / "evaluation_report.md").write_text(build_report(result))
    return result


def build_report(r: dict) -> str:
    def row(name, d):
        return (f"| {name} | {d['mae_kwh_per_m2']:.1f} | {d['rmse_kwh_per_m2']:.1f} | "
                f"{d['r2']:.3f} | {d['mae_kwh_per_year']:,.0f} |")
    head = "| Model | MAE (kWh/m2) | RMSE (kWh/m2) | R2 | MAE (kWh/year) |\n|---|---|---|---|---|\n"
    imp = "\n".join(f"- {i['feature']}: +{i['mae_increase']:.2f} kWh/m2 MAE when shuffled"
                    for i in r["permutation_importance_validation"])
    s, b, m = r["split_sizes"], r["baseline"], r["model"]
    return f"""# Evaluation Report: energy-use model

**Task:** predict {r['target']}.
**Split:** by property, seed {r['seed']}. Train {s['train']}, validation {s['validation']}, test {s['test']}. No property appears in more than one split.
**Features (available at prediction time):** {', '.join(r['features'])}. Property value, sale prices and energy readings are not used as inputs.
**Missing data:** {r['homes_dropped_missing_data']} homes dropped for missing energy data (missing months are never treated as zero).
**Model:** {m['description']}. Hyperparameters chosen on validation only: {r['best_params']}.
**Baseline:** {b['description']}.

## Validation results
{head}{row('Baseline', b['validation'])}
{row('Model', m['validation'])}

## Held-out test results (used once, after model selection)
{head}{row('Baseline', b['test'])}
{row('Model', m['test'])}

## What drives the predictions (permutation importance, validation set)
{imp}

## Limitations
- The dataset is synthetic. Energy values were generated by documented rules (poorer insulation, older homes, single glazing, gas boilers and lower ratings use more energy per m2, with seasonal patterns and noise). The model learning those rules does not prove it would predict real homes accurately.
- Metrics are only comparable with the baseline on this same dataset.
- The model predicts energy intensity. Retrofit savings, carbon and value are calculated by the rule-based engine, not by this model.
- Predictions are estimates, not guarantees.
"""


def render():
    """Streamlit section: model results plus a button to (re)train."""
    import streamlit as st

    st.header("Energy-use model")
    st.caption("Predicts annual purchased energy per m2 from property features and compares it with a simple "
               "baseline. Synthetic data: results show the method works, not real-world accuracy.")
    if st.button("Train / refresh model", key="ml_train"):
        with st.spinner("Training..."):
            train()
        st.success("Done. Results saved to the reports folder.")
    mp = REPORT_DIR / "metrics.json"
    if not mp.exists():
        st.info("No model yet. Press 'Train / refresh model'.")
        return
    r = json.loads(mp.read_text())
    rows = []
    for name, key in (("Baseline (mean by property type)", "baseline"), ("Gradient boosting model", "model")):
        for split in ("validation", "test"):
            d = r[key][split]
            rows.append({"Model": name, "Split": split, "MAE (kWh/m2)": round(d["mae_kwh_per_m2"], 1),
                         "RMSE (kWh/m2)": round(d["rmse_kwh_per_m2"], 1), "R2": round(d["r2"], 3),
                         "MAE (kWh/year)": round(d["mae_kwh_per_year"])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.write("Split sizes:", r["split_sizes"])
    pp = REPORT_DIR / "test_predictions.csv"
    if pp.exists():
        t = pd.read_csv(pp)
        st.caption("Held-out test homes: actual vs predicted kWh/m2/year (closer to the diagonal is better).")
        st.scatter_chart(t.rename(columns={"actual_kwh_per_m2": "actual", "model_pred_kwh_per_m2": "predicted"}),
                         x="actual", y="predicted")
    with st.expander("What drives the predictions"):
        st.dataframe(pd.DataFrame(r["permutation_importance_validation"]), hide_index=True)


if __name__ == "__main__":
    res = train()
    print(json.dumps({k: res[k] for k in ("split_sizes", "baseline", "model")}, indent=2))
    print("Saved to", REPORT_DIR)