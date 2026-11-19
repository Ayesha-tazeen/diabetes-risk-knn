"""Diabetes Risk Classification Application
Using K-Nearest Neighbors (KNN) Classifier.

Schema:
    Patient(PatientID, Glucose, BloodPressure, BMI, Insulin, Age, DiabetesStatus)
    0 = Non-Diabetic, 1 = Diabetic
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Classifier (KNN)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-lbl {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .alert-diabetic {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 18px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }
    .alert-nondiabetic {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 18px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data Generation & Loading
# ---------------------------------------------------------
@st.cache_data
def load_or_generate_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """Load existing dataset or generate realistic 500-row synthetic dataset with invalid zeros."""
    csv_file = "diabetes_data.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            expected_cols = [
                "PatientID",
                "Glucose",
                "BloodPressure",
                "BMI",
                "Insulin",
                "Age",
                "DiabetesStatus",
            ]
            if list(df.columns) == expected_cols and len(df) == n_samples:
                return df
        except Exception:
            pass

    # Deterministic generation
    np.random.seed(random_seed)
    patient_ids = [f"PID{1001 + i}" for i in range(n_samples)]
    age = np.random.randint(21, 75, size=n_samples)

    is_diabetic = []
    glucose_vals = []
    bp_vals = []
    bmi_vals = []
    insulin_vals = []

    for i in range(n_samples):
        # Realistic clinical probability curve
        prob = np.clip(0.25 + (age[i] - 30) * 0.006, 0.15, 0.65)
        diab = 1 if np.random.rand() < prob else 0
        is_diabetic.append(diab)

        if diab == 1:
            g = np.random.normal(148, 28)
            bp = np.random.normal(76, 12)
            bmi = np.random.normal(34.5, 6.5)
            ins = np.random.normal(160, 60)
        else:
            g = np.random.normal(108, 18)
            bp = np.random.normal(70, 10)
            bmi = np.random.normal(27.5, 5.0)
            ins = np.random.normal(95, 40)

        glucose_vals.append(max(40.0, round(float(g), 1)))
        bp_vals.append(max(40.0, round(float(bp), 1)))
        bmi_vals.append(max(16.0, round(float(bmi), 1)))
        insulin_vals.append(max(10.0, round(float(ins), 1)))

    df = pd.DataFrame(
        {
            "PatientID": patient_ids,
            "Glucose": glucose_vals,
            "BloodPressure": bp_vals,
            "BMI": bmi_vals,
            "Insulin": insulin_vals,
            "Age": age,
            "DiabetesStatus": is_diabetic,
        }
    )

    # Inject invalid 0 values into physiological columns
    zero_rates = [
        ("Glucose", 0.05),
        ("BloodPressure", 0.07),
        ("BMI", 0.04),
        ("Insulin", 0.18),
    ]
    for col, rate in zero_rates:
        zero_count = int(n_samples * rate)
        zero_indices = np.random.choice(n_samples, size=zero_count, replace=False)
        df.loc[zero_indices, col] = 0.0

    return df


# ---------------------------------------------------------
# Model Training & Pipeline
# ---------------------------------------------------------
@st.cache_resource
def train_diabetes_pipeline(df: pd.DataFrame):
    """
    Executes the ML pipeline:
    1. Replaces invalid 0s with NaN for Glucose, BloodPressure, BMI, Insulin.
    2. Drops PatientID and splits 80/20 stratified.
    3. Imputes missing values with Median.
    4. Scales features with StandardScaler.
    5. Evaluates k in [3, 5, 7, 9] to select the best k.
    6. Fits final KNN classifier and computes metrics.
    """
    zero_cols = ["Glucose", "BloodPressure", "BMI", "Insulin"]
    feature_cols = ["Glucose", "BloodPressure", "BMI", "Insulin", "Age"]

    # Compute raw invalid zero counts
    zero_counts = {col: int((df[col] == 0).sum()) for col in zero_cols}

    # Prepare features and target
    X = df[feature_cols].copy()
    y = df["DiabetesStatus"].copy()

    # Replace invalid 0 with NaN
    X[zero_cols] = X[zero_cols].replace(0, np.nan)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Median Imputation fit only on train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Median values stored for display
    imputed_medians = dict(zip(feature_cols, imputer.statistics_))

    # Standard Scaling fit only on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Test candidate k values in [3, 5, 7, 9]
    candidate_k = [3, 5, 7, 9]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k_comparison = []

    for k in candidate_k:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        knn_temp.fit(X_train_scaled, y_train)
        test_acc = knn_temp.score(X_test_scaled, y_test)
        k_comparison.append(
            {
                "k": k,
                "CV Accuracy": float(cv_scores.mean()),
                "CV Std": float(cv_scores.std()),
                "Test Accuracy": float(test_acc),
            }
        )

    k_df = pd.DataFrame(k_comparison)

    # Pick best k based on cross-validation score
    best_k_idx = k_df["CV Accuracy"].idxmax()
    best_k = int(k_df.loc[best_k_idx, "k"])

    # Final Model Training with best k
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)

    # Evaluation on holdout test set
    y_pred = best_knn.predict(X_test_scaled)
    y_pred_proba = best_knn.predict_proba(X_test_scaled)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cr_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Diabetic", "Diabetic"],
        output_dict=True,
    )

    return {
        "imputer": imputer,
        "scaler": scaler,
        "model": best_knn,
        "best_k": best_k,
        "candidate_k": candidate_k,
        "k_df": k_df,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "cr_dict": cr_dict,
        "zero_counts": zero_counts,
        "imputed_medians": imputed_medians,
        "feature_cols": feature_cols,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Header Banner
    st.title("🩺 Diabetes Risk Classification App")
    st.markdown(
        "A machine learning application using **K-Nearest Neighbors (KNN)** with automated median imputation "
        "and feature standardization to assess diabetes risk."
    )

    # Load Data & Train Pipeline
    df = load_or_generate_dataset()
    results = train_diabetes_pipeline(df)

    # Sidebar Information
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown(
            """
            **Dataset Schema:**
            - `PatientID` (Dropped from features)
            - `Glucose` (mg/dL)
            - `BloodPressure` (mm Hg)
            - `BMI` (kg/m²)
            - `Insulin` (μU/mL)
            - `Age` (Years)
            - `DiabetesStatus` (Target: 0 or 1)
            """
        )
        st.markdown("---")
        st.subheader("⚙️ Preprocessing Pipeline")
        st.markdown(
            """
            1. **Invalid 0s Replacement:**
               `0.0` in Glucose, BP, BMI, Insulin replaced with `NaN`.
            2. **Median Imputation:**
               Imputed using training set medians (`SimpleImputer`).
            3. **Feature Scaling:**
               `StandardScaler` fitted on training set.
            4. **Evaluation:**
               80/20 Stratified Split & 5-fold CV across $k \\in [3, 5, 7, 9]$.
            """
        )

        st.markdown("---")
        st.subheader("💾 Export Data")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Dataset",
            data=csv_data,
            file_name="diabetes_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🔮 Interactive Patient Prediction",
            "📊 Model Performance & Tuning",
            "📋 Dataset & Preprocessing Details",
        ]
    )

    # =========================================================
    # TAB 1: Interactive Prediction Form
    # =========================================================
    with tab1:
        st.subheader("🧪 Individual Patient Risk Assessment")
        st.info(
            "Enter the clinical parameters below. The model will scale inputs and output the predicted risk along with class probabilities."
        )

        with st.form("patient_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_glucose = st.number_input(
                    "Glucose (mg/dL)",
                    min_value=0.0,
                    max_value=300.0,
                    value=148.0,
                    step=1.0,
                    help="Plasma glucose concentration. (Set to 0 if unknown/missing to trigger median imputation)",
                )
                input_bp = st.number_input(
                    "Blood Pressure (mm Hg)",
                    min_value=0.0,
                    max_value=200.0,
                    value=72.0,
                    step=1.0,
                    help="Diastolic blood pressure. (Set to 0 if unknown/missing)",
                )
                input_bmi = st.number_input(
                    "Body Mass Index - BMI (kg/m²)",
                    min_value=0.0,
                    max_value=70.0,
                    value=33.6,
                    step=0.1,
                    help="Body mass index = weight in kg / (height in m)^2",
                )

            with col2:
                input_insulin = st.number_input(
                    "Insulin (μU/mL)",
                    min_value=0.0,
                    max_value=900.0,
                    value=130.0,
                    step=1.0,
                    help="2-Hour serum insulin level. (Set to 0 if unknown/missing)",
                )
                input_age = st.number_input(
                    "Age (Years)",
                    min_value=1.0,
                    max_value=120.0,
                    value=50.0,
                    step=1.0,
                    help="Patient age in years",
                )

            submit_btn = st.form_submit_button("🔍 Run Prediction", use_container_width=True)

        # Execute Prediction
        if submit_btn or True:  # Evaluates automatically with defaults
            # Build input dataframe
            input_df = pd.DataFrame(
                [
                    {
                        "Glucose": np.nan if input_glucose == 0 else input_glucose,
                        "BloodPressure": np.nan if input_bp == 0 else input_bp,
                        "BMI": np.nan if input_bmi == 0 else input_bmi,
                        "Insulin": np.nan if input_insulin == 0 else input_insulin,
                        "Age": input_age,
                    }
                ]
            )

            # Apply Pipeline Transformations
            imputed_input = results["imputer"].transform(input_df)
            scaled_input = results["scaler"].transform(imputed_input)

            # Predict
            pred_class = int(results["model"].predict(scaled_input)[0])
            pred_proba = results["model"].predict_proba(scaled_input)[0]
            prob_nondiabetic = float(pred_proba[0]) * 100.0
            prob_diabetic = float(pred_proba[1]) * 100.0

            st.markdown("### 📋 Prediction Results")
            res_col1, res_col2 = st.columns([1.2, 1.8])

            with res_col1:
                if pred_class == 1:
                    st.markdown(
                        """
                        <div class="alert-diabetic">
                            ⚠️ <strong>Prediction: Diabetic</strong><br>
                            High Risk Detected
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="alert-nondiabetic">
                            ✅ <strong>Prediction: Non-Diabetic</strong><br>
                            Low Risk / Normal
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.markdown(f"**Model Used:** KNN ($k={results['best_k']}$)")

            with res_col2:
                st.markdown("**Confidence & Probability Breakdown:**")
                st.progress(float(pred_proba[1]))
                pcol_a, pcol_b = st.columns(2)
                pcol_a.metric("Non-Diabetic Probability", f"{prob_nondiabetic:.1f}%")
                pcol_b.metric("Diabetic Risk Probability", f"{prob_diabetic:.1f}%")

            # Comparative clinical insights
            with st.expander("🔎 View Clinical Reference & Comparison with Population Medians"):
                comparison_df = pd.DataFrame(
                    {
                        "Parameter": ["Glucose", "BloodPressure", "BMI", "Insulin", "Age"],
                        "Patient Value": [
                            input_glucose,
                            input_bp,
                            input_bmi,
                            input_insulin,
                            input_age,
                        ],
                        "Dataset Median": [
                            round(results["imputed_medians"]["Glucose"], 1),
                            round(results["imputed_medians"]["BloodPressure"], 1),
                            round(results["imputed_medians"]["BMI"], 1),
                            round(results["imputed_medians"]["Insulin"], 1),
                            round(results["imputed_medians"]["Age"], 1),
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # =========================================================
    # TAB 2: Model Performance & Tuning
    # =========================================================
    with tab2:
        st.subheader("📈 Classification Performance & KNN Hyperparameter Selection")

        # Top summary metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Selected Best k</div>
                    <div class="metric-val">{results['best_k']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Test Accuracy</div>
                    <div class="metric-val">{results['test_accuracy'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Precision (Diabetic)</div>
                    <div class="metric-val">{results['precision'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Recall (Diabetic)</div>
                    <div class="metric-val">{results['recall'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Visualizations: Tuning Comparison and Confusion Matrix
        chart_col1, chart_col2 = st.columns([1.1, 1.1])

        with chart_col1:
            st.markdown(f"#### 🔍 Hyperparameter Tuning: $k \\in [3, 5, 7, 9]$")
            k_df_display = results["k_df"].copy()
            k_df_display["Is Best"] = k_df_display["k"].apply(
                lambda x: "★ Best" if x == results["best_k"] else ""
            )

            fig_k, ax_k = plt.subplots(figsize=(6, 4))
            bars = ax_k.bar(
                [f"k={k}" for k in results["k_df"]["k"]],
                results["k_df"]["CV Accuracy"],
                color=["#1f77b4" if k != results["best_k"] else "#2ca02c" for k in results["k_df"]["k"]],
                width=0.5,
                edgecolor="black",
                linewidth=0.8,
            )
            ax_k.set_ylabel("5-Fold CV Accuracy", fontsize=11)
            ax_k.set_title("Cross-Validation Accuracy by k-Value", fontsize=12, fontweight="bold")
            ax_k.set_ylim([0.7, 1.0])
            for bar in bars:
                height = bar.get_height()
                ax_k.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.008,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig_k)
            plt.close(fig_k)

            st.dataframe(
                k_df_display.style.format(
                    {"CV Accuracy": "{:.4f}", "CV Std": "{:.4f}", "Test Accuracy": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with chart_col2:
            st.markdown("#### 🎯 Holdout Test Confusion Matrix")
            cm = results["cm"]
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))

            # Cell annotations with counts and descriptive labels
            labels = [
                [f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"],
                [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"],
            ]

            sns.heatmap(
                cm,
                annot=labels,
                fmt="",
                cmap="Blues",
                cbar=False,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
                yticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
            ax_cm.set_ylabel("Actual Class", fontsize=11, fontweight="bold")
            ax_cm.set_title(
                f"Confusion Matrix (Test N={results['test_size']})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.caption(
                f"Total Test Samples: **{results['test_size']}** | "
                f"Correct Predictions: **{cm[0, 0] + cm[1, 1]}** | "
                f"Misclassifications: **{cm[0, 1] + cm[1, 0]}**"
            )

        # Classification Report Table
        st.markdown("#### 📑 Comprehensive Classification Report")
        report_df = pd.DataFrame(results["cr_dict"]).transpose()
        st.dataframe(
            report_df.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

    # =========================================================
    # TAB 3: Dataset & Preprocessing Details
    # =========================================================
    with tab3:
        st.subheader("📋 Dataset Inspection & Preprocessing Diagnostics")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### 🩺 Data Overview")
            st.write(f"- **Total Rows:** {len(df)}")
            st.write(f"- **Features Used:** `Glucose`, `BloodPressure`, `BMI`, `Insulin`, `Age`")
            st.write(f"- **Excluded Identifier:** `PatientID`")
            st.write(f"- **Target Variable:** `DiabetesStatus` (0 = Non-Diabetic, 1 = Diabetic)")
            st.write(f"- **Stratified Split:** 80% Train ({results['train_size']} rows) / 20% Test ({results['test_size']} rows)")

        with col_d2:
            st.markdown("##### ⚠️ Replaced Invalid 0s & Imputed Medians")
            zero_summary = pd.DataFrame(
                [
                    {
                        "Feature": col,
                        "Invalid 0s Found": results["zero_counts"][col],
                        "Missing Rate": f"{(results['zero_counts'][col] / len(df)) * 100:.1f}%",
                        "Imputed Median Value": round(results["imputed_medians"][col], 2),
                    }
                    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]
                ]
            )
            st.dataframe(zero_summary, use_container_width=True, hide_index=True)

        st.markdown("##### 🔬 First 10 Rows of Patient Dataset")
        st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
