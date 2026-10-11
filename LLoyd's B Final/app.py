from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dataset import (
    DATASET_VERSION,
    DATA_YEAR,
    HOME_COUNT,
    RATING_ORDER,
    generate_synthetic_dataset,
)
from scoring import COMPONENTS, build_key_issues, score_homes


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
HOMES_CSV = DATA_DIR / "homes.csv"
ENERGY_CSV = DATA_DIR / "monthly_energy.csv"

st.set_page_config(
    page_title="Green Home Score and Retrofit Advisor",
    layout="wide",
)

st.title("Green Home Score and Retrofit Advisor")
st.write(
    "Step 1: create a sample of 2,000 fictional UK homes and their monthly "
    "electricity and gas use."
)

st.caption(
    "Data notes: gas use peaks from November to February and is much lower in "
    "summer. Electricity is slightly higher in winter for gas-heated homes and "
    "noticeably higher for homes with electric or heat-pump heating. Seasonality "
    "redistributes use across the year, so annual totals stay about the same."
)

if "active_seed" not in st.session_state:
    st.session_state.active_seed = 42

with st.sidebar.form("dataset_controls"):
    submitted_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999_999,
        value=int(st.session_state.active_seed),
        step=1,
        help="Use the same seed to recreate the same sample.",
    )
    generate_clicked = st.form_submit_button("Generate dataset")

if generate_clicked:
    st.session_state.active_seed = int(submitted_seed)

seed = int(st.session_state.active_seed)
homes, monthly_energy = generate_synthetic_dataset(seed)

DATA_DIR.mkdir(parents=True, exist_ok=True)
if (
    st.session_state.get("saved_seed") != seed
    or st.session_state.get("saved_dataset_version") != DATASET_VERSION
    or not HOMES_CSV.exists()
    or not ENERGY_CSV.exists()
):
    homes.to_csv(HOMES_CSV, index=False)
    monthly_energy.to_csv(ENERGY_CSV, index=False)
    st.session_state.saved_seed = seed
    st.session_state.saved_dataset_version = DATASET_VERSION

home_scores = score_homes(homes, monthly_energy)
average_intensity = home_scores["energy_intensity_kwh_m2_year"].mean()

metric_columns = st.columns(4)
metric_columns[0].metric("Homes", f"{len(homes):,}")
metric_columns[1].metric("Monthly records", f"{len(monthly_energy):,}")
metric_columns[2].metric("Months per home", "12")
metric_columns[3].metric(
    "Avg. annual use",
    f"{average_intensity:.0f}",
    help="Average electricity and gas use per square metre per year (kWh/m²/year).",
)

st.subheader("Download the CSV files")
download_columns = st.columns(2)
download_columns[0].download_button(
    "Download homes.csv",
    data=HOMES_CSV.read_bytes(),
    file_name="homes.csv",
    mime="text/csv",
    width="stretch",
)
download_columns[1].download_button(
    "Download monthly_energy.csv",
    data=ENERGY_CSV.read_bytes(),
    file_name="monthly_energy.csv",
    mime="text/csv",
    width="stretch",
)
st.caption("The same files are saved in the app's data folder.")

preview_size = st.slider("Rows to show in the preview", 5, 50, 10, step=5)
homes_tab, energy_tab = st.tabs(["Home details", "Monthly energy use"])

with homes_tab:
    st.caption("floor_area is measured in square metres; property_value is in GBP.")
    st.caption(f"Showing {preview_size} of {len(homes):,} homes.")
    st.dataframe(homes.head(preview_size), hide_index=True, width="stretch")

with energy_tab:
    st.caption(
        f"Usage covers January–December {DATA_YEAR}. Consumption is measured in kWh."
    )
    st.caption(
        f"Showing {preview_size} of {len(monthly_energy):,} property-month records. "
        "Each home has one row for each month."
    )
    st.dataframe(
        monthly_energy.head(preview_size),
        hide_index=True,
        width="stretch",
    )

rating_counts = (
    homes["energy_rating"]
    .value_counts()
    .reindex(RATING_ORDER, fill_value=0)
    .rename_axis("Energy rating")
    .to_frame("Homes")
)
st.subheader("Sample homes by energy rating")
st.bar_chart(rating_counts)

st.subheader("Score summaries")
summary_columns = st.columns(2)
score_class_order = [
    "Highly efficient",
    "Good",
    "Moderate",
    "Poor",
    "Very poor",
]
class_counts = (
    home_scores["score_class"]
    .value_counts()
    .reindex(score_class_order, fill_value=0)
    .rename_axis("Score class")
    .reset_index(name="Homes")
)
average_score_by_rating = (
    home_scores.groupby("energy_rating")["score"]
    .mean()
    .reindex(RATING_ORDER)
    .round(1)
    .rename_axis("Energy rating")
    .reset_index(name="Average score")
)
with summary_columns[0]:
    st.markdown("**Homes by score class**")
    st.dataframe(class_counts, hide_index=True, height=240, width="stretch")
with summary_columns[1]:
    st.markdown("**Average score by energy rating**")
    st.dataframe(
        average_score_by_rating,
        hide_index=True,
        height=240,
        width="stretch",
    )

st.subheader("Green Home Score")
st.write(
    "Choose a home to see its 0–100 score, classification, and how much each "
    "scoring component contributes."
)

scores_by_id = home_scores.set_index("property_id", drop=False)


def format_home_option(property_id: str) -> str:
    home = scores_by_id.loc[property_id]
    if pd.isna(home["score"]):
        return f"{property_id} — Not scored"
    return f"{property_id} — {int(home['score'])}/100 · {home['score_class']}"


selected_property_id = st.selectbox(
    "Choose a home",
    options=home_scores["property_id"].tolist(),
    format_func=format_home_option,
)
selected_home = scores_by_id.loc[selected_property_id]

score_metrics = st.columns(4)
score_metrics[0].metric(
    "Green Home Score",
    "Not scored" if pd.isna(selected_home["score"]) else f"{int(selected_home['score'])}/100",
)
score_metrics[1].metric("Class", selected_home["score_class"])
intensity = selected_home["energy_intensity_kwh_m2_year"]
score_metrics[2].metric(
    "Annual energy intensity",
    "Unavailable" if pd.isna(intensity) else f"{float(intensity):.1f} kWh/m²/year",
)
score_metrics[3].metric(
    "Complete months",
    f"{int(selected_home['months_observed'])}/12",
)

if selected_home["has_missing_months"]:
    observed = int(selected_home["months_observed"])
    if observed:
        st.warning(
            f"{int(selected_home['missing_month_count'])} month(s) are missing. "
            f"Energy use is annualised from the {observed} complete month(s); "
            "missing months are not counted as zero."
        )
    else:
        st.warning(
            "There are no complete monthly energy readings, so this home cannot "
            "be scored until at least one month has both electricity and gas readings."
        )
if pd.notna(selected_home["score"]):
    st.progress(int(selected_home["score"]) / 100,
                text=f"Green Home Score: {int(selected_home['score'])}/100 ({selected_home['score_class']})")
if pd.notna(selected_home["score"]):
    component_rows = []
    for component_name, subscore_column, weight in COMPONENTS:
        component_rows.append(
            {
                "Component": component_name,
                "Weight": f"{weight}%",
                "Sub-score": int(selected_home[subscore_column]),
                "Contribution (points)": round(
                    float(selected_home[f"{subscore_column}_contribution"]), 2
                ),
            }
        )
    component_breakdown = pd.DataFrame(component_rows)

    st.caption(
        "Each bar is the component sub-score multiplied by its fixed weight. "
        "The contributions add up to the final score before rounding."
    )
    st.bar_chart(
        component_breakdown.set_index("Component")[["Contribution (points)"]],
        horizontal=True,
        x_label="Contribution (score points)",
        y_label="Scoring component",
    )
    st.dataframe(component_breakdown, hide_index=True, width="stretch")

st.subheader("Homes classed Poor or Very poor")
poor_homes = home_scores.loc[
    home_scores["score_class"].isin(["Poor", "Very poor"])
].copy()
if poor_homes.empty:
    st.info("No homes are currently classed Poor or Very poor.")
else:
    poor_homes["key_issues"] = poor_homes.apply(build_key_issues, axis=1)
    st.caption(f"{len(poor_homes):,} homes meet these score classes.")
    st.dataframe(
        poor_homes[
            [
                "property_id",
                "score",
                "score_class",
                "energy_rating",
                "energy_intensity_kwh_m2_year",
                "insulation",
                "heating",
                "glazing",
                "solar",
                "missing_month_count",
                "key_issues",
            ]
        ].rename(
            columns={
                "score": "green_home_score",
                "score_class": "class",
                "energy_intensity_kwh_m2_year": "kWh_per_m2_per_year",
                "missing_month_count": "missing_months",
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.session_state["score_table"] = home_scores

import retrofit
retrofit.render()

try:
    import ml_model
    ml_model.render()
except ImportError as e:
    st.warning(f"Energy-use model needs a library that is not installed yet: {e}")

try:
    import llm_explain
    llm_explain.render()
except ImportError as e:
    st.warning(f"AI explanation section could not load: {e}")

try:
    import my_home
    my_home.render()
except Exception as e:
    st.warning(f"'Check your own home' section could not load: {e}")