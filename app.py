"""
app.py
Streamlit dashboard for:
COMP 3106 – Fuzzy Logic Health Risk Prediction System

Frontend: Streamlit
Backend:  fuzzy_model.py (Tirth’s fuzzy logic)

Run with:
    streamlit run app.py
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fuzzy_model import (
    DATA_PATH,
    BMI_COL,
    BP_COL,
    EXERCISE_COL,
    SMOKING_COL,
    DIET_COL,
    TARGET_COL,
    SKLEARN_AVAILABLE,
    apply_model_to_dataset,
    predict_for_ui,
)

# ---------------------------------------
# Matplotlib styling (premium dark mode)
# ---------------------------------------

matplotlib.rcParams.update(
    {
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
        "figure.facecolor": (0, 0, 0, 0),
        "axes.facecolor": (0, 0, 0, 0),
        "savefig.facecolor": (0, 0, 0, 0),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

# ---------------------------------------
# Page config
# ---------------------------------------

st.set_page_config(
    page_title="Fuzzy Logic Health Risk Dashboard",
    layout="wide",
)

# ---------------------------------------
# Helper: cached loaders
# ---------------------------------------


@st.cache_data
def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_dataset_with_fuzzy() -> pd.DataFrame:
    """
    Ensure we have FuzzyRiskScore / FuzzyRiskLevel columns.
    Uses apply_model_to_dataset() once and caches result.
    """
    # If the file already exists, apply_model_to_dataset() will
    # simply recompute and overwrite; that's fine for this project.
    df_with = apply_model_to_dataset()
    return df_with


# ---------------------------------------
# Helper: paginated dataframe
# ---------------------------------------


def paginate_dataframe(df: pd.DataFrame, rows_per_page: int = 25, key_prefix: str = ""):
    total_rows = len(df)
    total_pages = max(1, int(np.ceil(total_rows / rows_per_page)))

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )

    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    subset = df.iloc[start:end]

    st.dataframe(subset, use_container_width=True, height=350)
    st.caption(f"Showing rows {start + 1}–{min(end, total_rows)} of {total_rows}")


# ---------------------------------------
# Sidebar – patient profile
# ---------------------------------------

st.sidebar.title("🩺 Patient Profile")

# Keep titles simple so they don’t overlap
with st.sidebar.expander("Body & Vitals", expanded=True):
    bmi_val = st.slider(
        "BMI (kg/m²)",
        min_value=18.0,
        max_value=38.0,
        value=25.0,
        step=0.1,
    )

    bp_val = st.slider(
        "Systolic Blood Pressure (mmHg)",
        min_value=100,
        max_value=190,
        value=120,
        step=1,
    )

    waist_val = st.slider(
        "Waist / Abdominal Circumference (cm)",
        min_value=70,
        max_value=120,
        value=90,
        step=1,
    )

with st.sidebar.expander("Lifestyle & Lab markers", expanded=True):
    smoking_label = st.radio(
        "Smoking status",
        options=["Non-smoker", "Smoker"],
        index=0,
        horizontal=True,
    )
    smoking_val = 0 if smoking_label == "Non-smoker" else 1

    chol_val = st.slider(
        "Total Cholesterol (mg/dL)",
        min_value=130,
        max_value=330,
        value=200,
        step=1,
    )

# Build dict with backend column names
input_dict = {
    BMI_COL: bmi_val,
    BP_COL: bp_val,
    EXERCISE_COL: waist_val,
    SMOKING_COL: smoking_val,
    DIET_COL: chol_val,
}

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔮 Predict risk", use_container_width=True)
st.sidebar.caption("Values are for educational demonstration only.")

# ---------------------------------------
# Main layout – header
# ---------------------------------------

st.markdown(
    "<h1 style='font-size: 38px;'>🧠 Fuzzy Logic Health Risk Dashboard</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Interactive, explainable cardiovascular risk demo for COMP 3106 – powered by a "
    "fuzzy rule-based system (backend: `fuzzy_model.py`)."
)

tab_pred, tab_data, tab_eval = st.tabs(
    ["⚡ Single Assessment", "📊 Dataset Analytics", "📈 Model Evaluation"]
)

# ======================================================
# TAB 1 – Single Assessment
# ======================================================

with tab_pred:
    st.subheader("Prediction Result")

    if not predict_clicked:
        st.info(
            "Set the patient profile in the sidebar and click **Predict risk** "
            "to run the fuzzy inference."
        )
        score = None
    else:
        result = predict_for_ui(input_dict)
        score = float(result["score"])
        category = result["category"]

        col_score, col_cat = st.columns(2)
        with col_score:
            st.metric("Risk Score (0–100)", f"{score:.1f}")
        with col_cat:
            st.metric("Risk Category", category)

        # Explanation banner
        if category == "Low":
            st.success("Low Risk — You show strong protective indicators.")
        elif category == "Medium":
            st.warning(
                "Medium Risk — Mixed profile with both protective and risk factors."
            )
        else:
            st.error("High Risk — Several risk factors are active simultaneously.")

    st.markdown("### Input snapshot")

    snap_cols = st.columns(5)
    with snap_cols[0]:
        st.write("**BMI**")
        st.write(f"{bmi_val:.1f}")
    with snap_cols[1]:
        st.write("**Systolic BP (mmHg)**")
        st.write(f"{bp_val:.0f}")
    with snap_cols[2]:
        st.write("**Waist (cm)**")
        st.write(f"{waist_val:.0f}")
    with snap_cols[3]:
        st.write("**Total Cholesterol**")
        st.write(f"{chol_val:.0f} mg/dL")
    with snap_cols[4]:
        st.write("**Smoking**")
        st.write(smoking_label)

    st.caption(
        "This tab mirrors the chosen patient profile and can be referenced in the "
        "report as an example of individual-level fuzzy inference."
    )

# ======================================================
# TAB 2 – Dataset Analytics (premium histograms)
# ======================================================

with tab_data:
    st.subheader("Dataset Analytics")

    df_raw = load_raw_dataset()

    st.markdown(
        f"Loaded **{df_raw.shape[0]}** rows and **{df_raw.shape[1]}** columns "
        f"from `{DATA_PATH}`."
    )

    with st.expander("Preview first 25 rows", expanded=False):
        st.dataframe(df_raw.head(25), use_container_width=True, height=300)

    st.markdown("### Interactive histograms")

    # Choose which variables to visualise
    numeric_options = {
        "BMI": BMI_COL,
        "Blood Pressure (systolic)": BP_COL,
        "Waist / Abdominal Circumference": EXERCISE_COL,
        "Total Cholesterol": DIET_COL,
    }

    selected_labels = st.multiselect(
        "Choose variables to visualise",
        options=list(numeric_options.keys()),
        default=[
            "Waist / Abdominal Circumference",
            "Blood Pressure (systolic)",
            "Total Cholesterol",
            "BMI",
        ],
    )

    if not selected_labels:
        st.info("Select at least one variable above to see histograms.")
    else:
        cols = st.columns(len(selected_labels))
        for idx, label in enumerate(selected_labels):
            col_name = numeric_options[label]
            with cols[idx]:
                st.markdown(f"**{label}**")
                series = df_raw[col_name].dropna()

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(
                    series,
                    bins=15,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.9,
                )
                ax.set_xlabel(label)
                ax.set_ylabel("Count")
                ax.set_title(f"{label} distribution")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlation heatmap (core numeric features)")

    numeric_df = df_raw[[BMI_COL, BP_COL, EXERCISE_COL, DIET_COL]].copy()
    corr = numeric_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    im = ax_corr.imshow(corr, interpolation="nearest")
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr.columns)
    fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("Correlation matrix")
    st.pyplot(fig_corr, use_container_width=True)

# ======================================================
# TAB 3 – Model Evaluation (rich, detailed)
# ======================================================

with tab_eval:
    st.subheader("Model Evaluation")

    df_fuzzy = load_dataset_with_fuzzy()

    st.markdown(
        "This tab uses the **full dataset with fuzzy outputs** "
        "(columns `FuzzyRiskScore` and `FuzzyRiskLevel`)."
    )

    st.markdown("### Dataset with fuzzy outputs (paginated)")

    cols_top = [
        BMI_COL,
        BP_COL,
        EXERCISE_COL,
        DIET_COL,
        "FuzzyRiskScore",
        "FuzzyRiskLevel",
    ]
    df_display = df_fuzzy[cols_top].copy()
    paginate_dataframe(df_display, rows_per_page=25, key_prefix="eval")



    # ---- Group summaries ----
    st.markdown("### Average fuzzy risk by smoking status")

    if SMOKING_COL in df_fuzzy.columns:
        mean_by_smoke = (
            df_fuzzy.groupby(SMOKING_COL)["FuzzyRiskScore"]
            .mean()
            .rename("Mean FuzzyRiskScore")
        )
        st.dataframe(mean_by_smoke, use_container_width=True)
    else:
        st.info(f"Column `{SMOKING_COL}` not found in dataset.")

    st.markdown("### Average fuzzy risk by waist size (≤ 90 vs > 90 cm)")
    if EXERCISE_COL in df_fuzzy.columns:
        small_mask = df_fuzzy[EXERCISE_COL] <= 90
        waist_summary = pd.DataFrame(
            {
                "Group": ["Waist ≤ 90", "Waist > 90"],
                "Mean FuzzyRiskScore": [
                    df_fuzzy.loc[small_mask, "FuzzyRiskScore"].mean(),
                    df_fuzzy.loc[~small_mask, "FuzzyRiskScore"].mean(),
                ],
            }
        )
        st.dataframe(waist_summary, use_container_width=True)
    else:
        st.info(f"Column `{EXERCISE_COL}` not found in dataset.")

    # ---- Visual model behaviour ----
    st.markdown("### How does fuzzy risk behave across the population?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Fuzzy risk by BMI**")
        fig_bmi, ax_bmi = plt.subplots(figsize=(4.5, 3.5))
        ax_bmi.scatter(
            df_fuzzy[BMI_COL],
            df_fuzzy["FuzzyRiskScore"],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_bmi.set_xlabel("BMI")
        ax_bmi.set_ylabel("FuzzyRiskScore")
        st.pyplot(fig_bmi, use_container_width=True)

    with col_b:
        st.markdown("**Fuzzy risk by Blood Pressure**")
        fig_bp, ax_bp = plt.subplots(figsize=(4.5, 3.5))
        ax_bp.scatter(
            df_fuzzy[BP_COL],
            df_fuzzy["FuzzyRiskScore"],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_bp.set_xlabel("Systolic BP (mmHg)")
        ax_bp.set_ylabel("FuzzyRiskScore")
        st.pyplot(fig_bp, use_container_width=True)

    st.markdown("**Distribution of fuzzy risk across risk levels**")
    fig_box, ax_box = plt.subplots(figsize=(5, 3.5))
    df_fuzzy.boxplot(
        column="FuzzyRiskScore",
        by="FuzzyRiskLevel",
        ax=ax_box,
        grid=True,
    )
    ax_box.set_xlabel("FuzzyRiskLevel")
    ax_box.set_ylabel("FuzzyRiskScore")
    ax_box.set_title("FuzzyRiskScore by FuzzyRiskLevel")
    plt.suptitle("")  # remove automatic pandas title
    st.pyplot(fig_box, use_container_width=True)

    st.caption(
        "Detailed console-style diagnostics (including printed summaries) are "
        "also available via `evaluate_fuzzy_model()` inside `fuzzy_model.py` "
        "if you run it in a standard Python environment."
    )
