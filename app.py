"""
Diabetes Risk — EDA & Prediction
Streamlit app: loads diabetes_risk.csv, explores it, and trains a model
to predict diabetes_risk (Low / Moderate / High) from patient attributes.
"""

import pandas as pd  # type: ignore[reportMissingModuleSource]
import plotly.express as px # type: ignore[reportMissingModuleSource]
import plotly.graph_objects as go # type: ignore[reportMissingModuleSource]
import streamlit as st # type: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer # type: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier # type: ignore[reportMissingModuleSource]
from sklearn.impute import SimpleImputer # type: ignore[reportMissingModuleSource]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # type: ignore[reportMissingModuleSource]
from sklearn.pipeline import Pipeline # type: ignore[reportMissingModuleSource]
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore[reportMissingModuleSource]

# ----------------------------------------------------------------------------
# Palette (validated categorical / sequential / diverging / status colors)
# ----------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#f0efec", "#e34948"]  # blue -> neutral -> red
STATUS = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#d03b3b"}
RISK_ORDER = ["Low", "Moderate", "High"]
INK = "#0b0b0b"
GRID = "#e1e0d9"

st.set_page_config(page_title="Diabetes Risk Explorer", layout="wide")

CHART_LAYOUT = dict(
    paper_bgcolor="#fcfcfb",
    plot_bgcolor="#fcfcfb",
    font_color=INK,
    margin=dict(t=50), # Only force the top margin, let Plotly auto-size the rest!
)


def style(fig, **kwargs):
    fig.update_layout(**CHART_LAYOUT, **kwargs)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(rgb):
    def chan(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def _color_at(stops, t):
    """stops: list of (position 0-1, hex color), sorted ascending. Linear-interpolates."""
    t = min(max(t, 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= t <= p1:
            local_t = 0 if p1 == p0 else (t - p0) / (p1 - p0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            return (r0 + (r1 - r0) * local_t, g0 + (g1 - g0) * local_t, b0 + (b1 - b0) * local_t)
    return _hex_to_rgb(stops[-1][1])


def heatmap_with_text(z_color, x_labels, y_labels, stops, zmin, zmax, text_values=None, fmt="{:.2f}"):
    """A go.Heatmap with per-cell annotations whose color is chosen for contrast
    against that cell's own background, so text stays legible on both light and
    dark cells."""
    fig = go.Figure(
        data=go.Heatmap(
            z=z_color, x=x_labels, y=y_labels,
            colorscale=[[p, c] for p, c in stops],
            zmin=zmin, zmax=zmax, showscale=True,
        )
    )
    span = (zmax - zmin) or 1
    for i, yl in enumerate(y_labels):
        for j, xl in enumerate(x_labels):
            val = z_color[i][j]
            t = (val - zmin) / span
            rgb = _color_at(stops, t)
            text_color = "#ffffff" if _relative_luminance(rgb) < 0.45 else INK
            display = text_values[i][j] if text_values is not None else fmt.format(val)
            fig.add_annotation(
                x=xl, y=yl, text=str(display), showarrow=False,
                font=dict(color=text_color, size=12),
            )
    return fig


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_risk.csv")
    return df


df = load_data()

st.title("Diabetes Risk Explorer")
st.caption(
    "Exploratory analysis and risk prediction over a synthetic diabetes-risk "
    "dataset of 15,000 patients."
)

tab_overview, tab_eda, tab_predict = st.tabs(
    ["📋 Data Overview", "📊 Exploratory Analysis", "🩺 Predict Risk"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Overview
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{df.isna().sum().sum():,}")
    c4.metric("High-risk patients", f"{(df['diabetes_risk'] == 'High').sum():,}")

    st.subheader("Sample rows")
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Column types")
        dtypes = pd.DataFrame({"column": df.dtypes.index, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Missing values")
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        miss_df = pd.DataFrame(
            {"column": miss.index, "missing": miss.values,
             "missing_%": (miss.values / len(df) * 100).round(1)}
        )
        if len(miss_df):
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                miss_df, x="missing_%", y="column", orientation="h",
                color_discrete_sequence=[CATEGORICAL[0]],
            )
            fig.update_traces(marker_line_width=0)
            style(fig, title="Missing values by column (%)", height=280,
                  yaxis_title="", xaxis_title="% missing")
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            st.write("No missing values.")

    st.subheader("Numeric summary")
    numeric_cols = df.select_dtypes(include="number").columns.drop("patient_id")
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — EDA visualizations
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Diabetes risk distribution")
    risk_counts = df["diabetes_risk"].value_counts().reindex(RISK_ORDER)
    fig = px.bar(
        x=risk_counts.index, y=risk_counts.values,
        color=risk_counts.index, color_discrete_map=STATUS,
        text=risk_counts.values,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    style(fig, title="Patient count by risk level", showlegend=False,
          xaxis_title="Diabetes risk", yaxis_title="Patients", height=380)
    st.plotly_chart(fig, use_container_width=False, theme=None) # <-- Added theme

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age distribution")
        fig = px.histogram(df, x="age", nbins=40, color_discrete_sequence=[SEQ_BLUE[3]])
        fig.update_traces(marker_line_width=0)
        style(fig, xaxis_title="Age", yaxis_title="Count", height=340)
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    with col2:
        st.subheader("BMI distribution by risk level")
        fig = px.histogram(
            df, x="bmi", color="diabetes_risk", category_orders={"diabetes_risk": RISK_ORDER},
            color_discrete_map=STATUS, barmode="overlay", nbins=50, opacity=0.65,
        )
        fig.update_traces(marker_line_width=0)
        style(fig, xaxis_title="BMI", yaxis_title="Count", height=340,
              legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fasting blood sugar vs. HbA1c")
        sample = df.sample(min(3000, len(df)), random_state=1)
        fig = px.scatter(
            sample, x="fasting_blood_sugar", y="hba1c_level", color="diabetes_risk",
            category_orders={"diabetes_risk": RISK_ORDER}, color_discrete_map=STATUS,
            opacity=0.55,
        )
        fig.update_traces(marker=dict(size=5, line_width=0))
        style(fig, xaxis_title="Fasting blood sugar (mg/dL)", yaxis_title="HbA1c (%)",
              height=380, legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    with col4:
        st.subheader("Risk rate by physical activity level")
        rate = (
            df.groupby("physical_activity_level")["diabetes_risk"]
            .value_counts(normalize=True).mul(100).rename("pct").reset_index()
        )
        fig = px.bar(
            rate, x="physical_activity_level", y="pct", color="diabetes_risk",
            category_orders={"diabetes_risk": RISK_ORDER,
                              "physical_activity_level": ["Sedentary", "Moderate", "Active"]},
            color_discrete_map=STATUS, barmode="stack",
        )
        fig.update_traces(marker_line_width=0)
        style(fig, xaxis_title="Physical activity", yaxis_title="% of group",
              height=380, legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    st.divider()
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Risk rate by family history of diabetes")
        rate2 = (
            df.groupby("family_history_diabetes")["diabetes_risk"]
            .value_counts(normalize=True).mul(100).rename("pct").reset_index()
        )
        fig = px.bar(
            rate2, x="family_history_diabetes", y="pct", color="diabetes_risk",
            category_orders={"diabetes_risk": RISK_ORDER},
            color_discrete_map=STATUS, barmode="stack",
        )
        fig.update_traces(marker_line_width=0)
        style(fig, xaxis_title="Family history of diabetes", yaxis_title="% of group",
              height=360, legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    with col6:
        st.subheader("Correlation among numeric features")
        corr_cols = ["age", "bmi", "hours_sleep_per_night", "stress_level",
                     "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
                     "blood_pressure_diastolic", "waist_circumference_cm"]
        corr = df[corr_cols].corr().round(2)
        fig = heatmap_with_text(
            corr.values, list(corr.columns), list(corr.columns),
            stops=[(0.0, DIVERGING[0]), (0.5, DIVERGING[1]), (1.0, DIVERGING[2])],
            zmin=-1, zmax=1,
        )
        style(fig, height=420)
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    st.subheader("Waist circumference vs. BMI")
    sample2 = df.sample(min(3000, len(df)), random_state=2)
    fig = px.scatter(
        sample2, x="bmi", y="waist_circumference_cm", color="diabetes_risk",
        category_orders={"diabetes_risk": RISK_ORDER}, color_discrete_map=STATUS,
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=5, line_width=0))
    style(fig, xaxis_title="BMI", yaxis_title="Waist circumference (cm)",
          height=420, legend_title="Risk")
    st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme
# ----------------------------------------------------------------------------
# TAB 3 — Prediction model
# ----------------------------------------------------------------------------
FEATURES_NUM = [
    "age", "bmi", "hours_sleep_per_night", "stress_level", "fasting_blood_sugar",
    "hba1c_level", "blood_pressure_systolic", "blood_pressure_diastolic",
    "waist_circumference_cm",
]
FEATURES_CAT = [
    "gender", "family_history_diabetes", "physical_activity_level", "diet_type",
    "smoking_status", "alcohol_consumption", "income_bracket",
]
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT


@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[ALL_FEATURES]
    y = data["diabetes_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), FEATURES_CAT),
        ]
    )

    model = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=RISK_ORDER)

    feat_names = (
        list(FEATURES_NUM)
        + list(model.named_steps["prep"].named_transformers_["cat"]
               .named_steps["onehot"].get_feature_names_out(FEATURES_CAT))
    )
    importances = model.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(15)

    return model, acc, report, cm, fi


model, acc, report, cm, fi = train_model(df)

with tab_predict:
    st.subheader("Model performance")
    st.caption("Random Forest classifier predicting Low / Moderate / High diabetes risk, "
               "trained on an 80/20 split of the dataset.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Test accuracy", f"{acc * 100:.1f}%")
        rep_df = pd.DataFrame(report).T.loc[RISK_ORDER, ["precision", "recall", "f1-score", "support"]].round(2)
        st.dataframe(rep_df, use_container_width=True)

    with c2:
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        fig = heatmap_with_text(
            cm_norm, RISK_ORDER, RISK_ORDER,
            stops=[(0.0, SEQ_BLUE[0]), (1.0, SEQ_BLUE[5])],
            zmin=0, zmax=1, text_values=cm,
        )
        style(fig, title="Confusion matrix (test set, counts)",
              xaxis_title="Predicted", yaxis_title="Actual", height=320)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    st.subheader("What drives the model")
    fig = px.bar(
        x=fi.values[::-1], y=fi.index[::-1], orientation="h",
        color_discrete_sequence=[CATEGORICAL[0]],
    )
    fig.update_traces(marker_line_width=0)
    style(fig, title="Top 15 feature importances", xaxis_title="Importance",
          yaxis_title="", height=420)
    st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme

    st.divider()
    st.subheader("Estimate a patient's diabetes risk")
    st.caption("Enter patient attributes and the trained model will predict their risk category.")

    with st.form("predict_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            age = st.slider("Age", 18, 95, 45)
            bmi = st.slider("BMI", 15.0, 60.0, 25.0, 0.1)
            waist = st.slider("Waist circumference (cm)", 50.0, 180.0, 90.0, 0.5)
            sleep = st.slider("Hours of sleep per night", 3.0, 12.0, 7.0, 0.5)
            stress = st.slider("Stress level (1-10)", 1, 10, 5)
        with f2:
            fbs = st.slider("Fasting blood sugar (mg/dL)", 60, 400, 100)
            hba1c = st.slider("HbA1c level (%)", 3.5, 15.0, 5.5, 0.1)
            bp_sys = st.slider("Systolic blood pressure", 80, 250, 120)
            bp_dia = st.slider("Diastolic blood pressure", 40, 140, 80)
        with f3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            family_hist = st.selectbox("Family history of diabetes", ["Yes", "No"])
            activity = st.selectbox("Physical activity level", ["Sedentary", "Moderate", "Active"])
            diet = st.selectbox("Diet type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Pescatarian"])
            smoking = st.selectbox("Smoking status", ["Never", "Current", "Former"])
            alcohol = st.selectbox("Alcohol consumption", ["Never", "Occasional", "Regular"])
            income = st.selectbox("Income bracket", ["Low", "Middle", "High"])

        submitted = st.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "age": age, "bmi": bmi, "hours_sleep_per_night": sleep, "stress_level": stress,
            "fasting_blood_sugar": fbs, "hba1c_level": hba1c,
            "blood_pressure_systolic": bp_sys, "blood_pressure_diastolic": bp_dia,
            "waist_circumference_cm": waist, "gender": gender,
            "family_history_diabetes": family_hist, "physical_activity_level": activity,
            "diet_type": diet, "smoking_status": smoking, "alcohol_consumption": alcohol,
            "income_bracket": income,
        }])

        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        classes = model.named_steps["clf"].classes_
        proba_map = dict(zip(classes, proba))

        st.markdown(
            f"### Predicted risk: "
            f"<span style='color:{STATUS[pred]}'>**{pred}**</span>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({
            "risk": RISK_ORDER,
            "probability": [proba_map.get(r, 0) * 100 for r in RISK_ORDER],
        })
        fig = px.bar(
            proba_df, x="risk", y="probability", color="risk",
            color_discrete_map=STATUS, text=proba_df["probability"].round(1),
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        style(fig, showlegend=False, xaxis_title="Risk level",
              yaxis_title="Predicted probability (%)", height=340)
        st.plotly_chart(fig, use_container_width=True, theme=None) # <-- Added theme
