from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


st.set_page_config(
  page_title="Demo de Detección de Fraude con XGBoost",
    page_icon="\U0001F4B3",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
  --bg: #07111f;
  --panel: rgba(10, 18, 33, 0.86);
  --panel-border: rgba(148, 163, 184, 0.18);
  --text: #e5eefc;
  --muted: #9fb0ca;
  --accent: #7dd3fc;
  --accent-2: #34d399;
  --warn: #fbbf24;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 32%),
    radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 28%),
    linear-gradient(180deg, #050b15 0%, #07111f 48%, #030711 100%);
  color: var(--text);
}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

.hero {
  padding: 1.4rem 1.5rem;
  border: 1px solid var(--panel-border);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(12, 20, 36, 0.92), rgba(7, 17, 31, 0.86));
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
}

.hero h1 {
  margin: 0;
  font-size: 2.35rem;
  line-height: 1.05;
}

.hero p {
  margin: 0.7rem 0 0;
  color: var(--muted);
  font-size: 1rem;
  max-width: 70ch;
}

.badge {
  display: inline-block;
  margin-bottom: 0.8rem;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  background: rgba(125, 211, 252, 0.12);
  color: #bfe8ff;
  border: 1px solid rgba(125, 211, 252, 0.22);
  font-size: 0.8rem;
  letter-spacing: 0.02em;
}

.metric-card {
  padding: 1rem 1rem 0.85rem;
  border-radius: 18px;
  border: 1px solid var(--panel-border);
  background: var(--panel);
}

.metric-label {
  color: var(--muted);
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.metric-value {
  color: var(--text);
  font-size: 1.85rem;
  font-weight: 700;
  margin-top: 0.25rem;
}

.metric-sub {
  color: var(--muted);
  font-size: 0.88rem;
  margin-top: 0.15rem;
}

.panel {
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  background: var(--panel);
  padding: 1rem 1rem 0.9rem;
}

.small-note {
  color: var(--muted);
  font-size: 0.88rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(6, 11, 21, 0.98), rgba(8, 15, 28, 0.95));
  border-right: 1px solid rgba(148, 163, 184, 0.14);
}

[data-testid="stSidebar"] * {
  color: var(--text);
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model_export"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


@dataclass(frozen=True)
class ModelBundle:
  model: XGBClassifier
  imputer: object
  feature_names: list[str]
  baseline_row: pd.Series
  metadata: dict[str, object]


EDITABLE_FEATURES = {
  "TransactionAmt": (120.0, 0.0, 2500.0, 1.0),
  "Transaction_hour": (13, 0, 23, 1),
  "Transaction_day": (15, 1, 31, 1),
  "Transaction_weekday": (2, 0, 6, 1),
  "dist1": (18.0, 0.0, 1000.0, 1.0),
  "dist2": (9.0, 0.0, 1000.0, 1.0),
  "C2": (0.0, 0.0, 100.0, 1.0),
  "C3": (0.0, 0.0, 100.0, 1.0),
  "C9": (0.0, 0.0, 100.0, 1.0),
  "D2": (1.0, 0.0, 1000.0, 1.0),
  "D7": (1.0, 0.0, 1000.0, 1.0),
  "D8": (1.0, 0.0, 1000.0, 1.0),
  "D9": (1.0, 0.0, 1000.0, 1.0),
  "D14": (1.0, 0.0, 1000.0, 1.0),
  "D15": (1.0, 0.0, 1000.0, 1.0),
}

PRODUCTCD_OPTIONS = ["W", "C", "R", "S", "H"]
PRODUCTCD_CODE_MAP = {label: index for index, label in enumerate(sorted(PRODUCTCD_OPTIONS))}
CARD4_OPTIONS = ["NaN", "american express", "discover", "mastercard", "visa"]
CARD4_CODE_MAP = {label: index for index, label in enumerate(sorted(CARD4_OPTIONS))}
CARD6_OPTIONS = ["NaN", "charge card", "credit", "debit", "debit or credit"]
CARD6_CODE_MAP = {label: index for index, label in enumerate(sorted(CARD6_OPTIONS))}
DEVICETYPE_OPTIONS = ["NaN", "desktop", "mobile"]
DEVICETYPE_CODE_MAP = {label: index for index, label in enumerate(sorted(DEVICETYPE_OPTIONS))}


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> ModelBundle:
  if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing XGBoost artifact: {MODEL_PATH}")
  if not IMPUTER_PATH.exists():
    raise FileNotFoundError(f"Missing imputer artifact: {IMPUTER_PATH}")
  if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

  model = joblib.load(MODEL_PATH)
  imputer = joblib.load(IMPUTER_PATH)
  with METADATA_PATH.open("r", encoding="utf-8") as handle:
    metadata = json.load(handle)

  feature_names = list(metadata["feature_names"])
  baseline_row = pd.Series(imputer.statistics_, index=feature_names)

  return ModelBundle(
    model=model,
    imputer=imputer,
    feature_names=feature_names,
    baseline_row=baseline_row,
    metadata=metadata,
  )


def build_scoring_frame(bundle: ModelBundle, overrides: dict[str, object]) -> pd.DataFrame:
  row = bundle.baseline_row.copy()
  for feature_name, feature_value in overrides.items():
    if feature_name in row.index:
      row[feature_name] = feature_value
  return pd.DataFrame([row], columns=bundle.feature_names)


def format_importance_frame(bundle: ModelBundle, top_n: int = 15) -> pd.DataFrame:
  importance_series = pd.Series(bundle.model.feature_importances_, index=bundle.feature_names)
  importance_series = importance_series.sort_values(ascending=False).head(top_n)
  importance_frame = importance_series.reset_index()
  importance_frame.columns = ["feature", "importance"]
  return importance_frame.sort_values("importance", ascending=True)


P_EMAILDOMAIN_OPTIONS = [
  "NaN",
  "aim.com",
  "anonymous.com",
  "aol.com",
  "att.net",
  "bellsouth.net",
  "cableone.net",
  "centurylink.net",
  "cfl.rr.com",
  "charter.net",
  "comcast.net",
  "cox.net",
  "earthlink.net",
  "embarqmail.com",
  "frontier.com",
  "frontiernet.net",
  "gmail",
  "gmail.com",
  "gmx.de",
  "hotmail.co.uk",
  "hotmail.com",
  "hotmail.de",
  "hotmail.es",
  "hotmail.fr",
  "icloud.com",
  "juno.com",
  "live.com",
  "live.com.mx",
  "live.fr",
  "mac.com",
  "mail.com",
  "me.com",
  "msn.com",
  "netzero.com",
  "netzero.net",
  "optonline.net",
  "outlook.com",
  "outlook.es",
  "prodigy.net.mx",
  "protonmail.com",
]
P_EMAILDOMAIN_FE_MAP = {
  option: index / (len(P_EMAILDOMAIN_OPTIONS) - 1)
  for index, option in enumerate(P_EMAILDOMAIN_OPTIONS)
}


def scale_value(value: float, minimum: float, maximum: float) -> float:
  if maximum == minimum:
    return 0.0
  return float((value - minimum) / (maximum - minimum))


bundle = load_model_bundle()
feature_importance_frame = format_importance_frame(bundle)
top_feature_row = feature_importance_frame.sort_values("importance", ascending=False).iloc[0]
top_feature_name = str(top_feature_row["feature"])
top_feature_score = float(top_feature_row["importance"])

st.markdown(
    """
    <div class="hero">
      <div class="badge">Cargado desde model_export</div>
      <h1>Demo de puntuación de fraude con modelo Ensembler</h1>
      <p>
        Explora un evaluador de fraude en vivo construido a partir del artefacto XGBoost exportado en
        <strong>model_export</strong>. La app carga el modelo y el imputador guardados al iniciar y luego
        puntúa una sola transacción sobre el esquema de 263 variables generadas usado durante el entrenamiento.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

col1, col2, col3, col4 = st.columns(4)
metric_boxes = [
  ("Artefacto cargado", "xgb_model.pkl", "XGBClassifier entrenado desde model_export"),
  ("Columnas", f"{len(bundle.feature_names)}", "Entradas generadas que espera el modelo"),
  ("Árboles", f"{bundle.model.n_estimators}", "Rondas de boosting guardadas en el modelo exportado"),
  ("Variable principal", f"{top_feature_score:.3f}", top_feature_name),
]
for column, (label, value, subtext) in zip([col1, col2, col3, col4], metric_boxes):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Simulador de transacción")
    st.caption("Ajusta unos pocos campos interpretables; el resto de las variables generadas conserva los valores por defecto del exportado.")

    with st.sidebar:
        st.header("Transacción de entrada")
        amount = st.slider("TransactionAmt", 0.0, 2500.0, 120.0, 1.0)
        productcd = st.selectbox("ProductCD", PRODUCTCD_OPTIONS, index=0)
        card4 = st.selectbox("card4", CARD4_OPTIONS, index=4)
        card6 = st.selectbox("card6", CARD6_OPTIONS, index=3)
        devicetype = st.selectbox("DeviceType", DEVICETYPE_OPTIONS, index=1)
        transaction_timestamp = st.datetime_input(
            "Transaction timestamp",
            value=datetime(2026, 4, 15, 13, 0),
        )
        p_emaildomain_default = P_EMAILDOMAIN_OPTIONS.index("gmail.com") if "gmail.com" in P_EMAILDOMAIN_OPTIONS else 0
        p_emaildomain = st.selectbox("P_emaildomain", P_EMAILDOMAIN_OPTIONS, index=p_emaildomain_default)
        addr1 = st.slider("addr1", 100.0, 540.0, 320.0, 1.0)
        card2 = st.slider("card2", 100.0, 600.0, 514.0, 1.0)
        d1 = st.slider("D1", 0.0, 636.0, 120.0, 1.0)
        d2 = st.slider("D2", 0.0, 636.0, 1.0, 1.0)
        d4 = st.slider("D4", -122.0, 700.0, 1.0, 1.0)
        d9 = st.slider("D9", 0.0, 0.958333015441894, 0.0, 0.001)
        d10 = st.slider("D10", 0.0, 705.0, 1.0, 1.0)
        d15 = st.slider("D15", -83.0, 709.0, 1.0, 1.0)
        c2 = st.slider("C2", 0.0, 100.0, 1.0, 1.0)
        threshold = st.slider("Umbral de fraude", 0.05, 0.95, 0.5, 0.01)

    transaction_day = int(transaction_timestamp.day)
    transaction_weekday = int(transaction_timestamp.weekday())
    transaction_hour = int(transaction_timestamp.hour)
    productcd_code = int(PRODUCTCD_CODE_MAP[productcd])
    card4_code = int(CARD4_CODE_MAP[card4])
    card6_code = int(CARD6_CODE_MAP[card6])
    devicetype_code = int(DEVICETYPE_CODE_MAP[devicetype])

    sample = build_scoring_frame(
        bundle,
        {
            "TransactionAmt": amount,
            "ProductCD": productcd_code,
            "card4": card4_code,
            "card6": card6_code,
            "DeviceType": devicetype_code,
            "Transaction_hour": transaction_hour,
            "Transaction_day": transaction_day,
            "Transaction_weekday": transaction_weekday,
            "addr1_FE": addr1,
            "card2_FE": card2,
            "P_emaildomain_FE": P_EMAILDOMAIN_FE_MAP.get(p_emaildomain, 0.0),
            "D1_norm": d1,
            "D2_norm": d2,
            "D4_norm": d4,
            "D9_norm": d9,
            "D10_norm": d10,
            "D15_norm": d15,
        },
    )

    risk_score = float(bundle.model.predict_proba(sample)[:, 1][0])
    decision = "Fraude" if risk_score >= threshold else "Legítima"
    decision_color = "#f87171" if decision == "Fraude" else "#34d399"

    st.markdown(
        f"""
        <div class="metric-card" style="margin-top: 1rem;">
          <div class="metric-label">Puntaje de riesgo en vivo</div>
          <div class="metric-value" style="color: {decision_color};">{risk_score:.1%}</div>
          <div class="metric-sub">Decisión con umbral {threshold:.0%}: {decision}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation_cols = st.columns(2)
    with explanation_cols[0]:
        st.metric("Decisión de fraude", decision)
    with explanation_cols[1]:
        st.metric("Umbral", f"{threshold:.0%}")

    st.markdown("#### Entradas del modelo")
    st.dataframe(sample.style.format({
        "TransactionAmt": "${:,.2f}",
        "Transaction_hour": "{:,.0f}",
        "Transaction_day": "{:,.0f}",
        "Transaction_weekday": "{:,.0f}",
      "addr1_FE": "{:,.3f}",
      "card2_FE": "{:,.3f}",
      "P_emaildomain_FE": "{:,.3f}",
      "D1_norm": "{:,.3f}",
      "D2_norm": "{:,.3f}",
      "D4_norm": "{:,.3f}",
      "D9_norm": "{:,.3f}",
      "D10_norm": "{:,.3f}",
      "D15_norm": "{:,.3f}",
    }), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Cómo se comporta el modelo")
    st.caption("El estimador exportado se puntúa directamente; las columnas no expuestas se mantienen con los valores por defecto del imputador.")

    perf_left, perf_right = st.columns(2)
    with perf_left:
        st.metric("Variables generadas", f"{len(bundle.feature_names)}")
        st.metric("Relleno del imputador", str(bundle.imputer.statistics_[0]))
    with perf_right:
        st.metric("Rondas de boosting", f"{bundle.model.n_estimators}")
        st.metric("Umbral", f"{threshold:.0%}")

    st.markdown("#### Importancia de variables")
    st.image(APP_DIR / "assets" / "topfeatures.png", use_container_width=True)

    st.markdown(
        """
        <p class="small-note">
        Esta demo carga el modelo XGBoost guardado desde <code>model_export/xgb_model.pkl</code> y el
        <code>imputer.pkl</code> correspondiente. Solo se expone un pequeño conjunto de controles interpretables;
        el resto de las 263 variables generadas se completa con el paquete exportado.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")