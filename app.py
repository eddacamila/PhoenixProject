from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Phoenix Fraud Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model_export"

PRODUCTCD_OPTIONS = ["W", "C", "R", "S", "H"]
PRODUCTCD_CODE_MAP = {label: i for i, label in enumerate(sorted(PRODUCTCD_OPTIONS))}
PRODUCTCD_LABELS = {
    "W": "W — Web purchase",
    "C": "C — Content / digital",
    "R": "R — Registration",
    "S": "S — Service",
    "H": "H — Home goods",
}

CARD4_OPTIONS = ["NaN", "american express", "discover", "mastercard", "visa"]
CARD4_CODE_MAP = {label: i for i, label in enumerate(sorted(CARD4_OPTIONS))}

CARD6_OPTIONS = ["NaN", "charge card", "credit", "debit", "debit or credit"]
CARD6_CODE_MAP = {label: i for i, label in enumerate(sorted(CARD6_OPTIONS))}

DEVICETYPE_OPTIONS = ["NaN", "desktop", "mobile"]
DEVICETYPE_CODE_MAP = {label: i for i, label in enumerate(sorted(DEVICETYPE_OPTIONS))}

P_EMAILDOMAIN_OPTIONS = [
    "NaN", "aim.com", "anonymous.com", "aol.com", "att.net", "bellsouth.net",
    "cableone.net", "centurylink.net", "cfl.rr.com", "charter.net", "comcast.net",
    "cox.net", "earthlink.net", "embarqmail.com", "frontier.com", "frontiernet.net",
    "gmail", "gmail.com", "gmx.de", "hotmail.co.uk", "hotmail.com", "hotmail.de",
    "hotmail.es", "hotmail.fr", "icloud.com", "juno.com", "live.com", "live.com.mx",
    "live.fr", "mac.com", "mail.com", "me.com", "msn.com", "netzero.com",
    "netzero.net", "optonline.net", "outlook.com", "outlook.es", "prodigy.net.mx",
    "protonmail.com",
]
P_EMAILDOMAIN_FE_MAP = {
    opt: i / (len(P_EMAILDOMAIN_OPTIONS) - 1)
    for i, opt in enumerate(P_EMAILDOMAIN_OPTIONS)
}

MODEL_KEY_MAP = {"XGBoost": "xgb", "LightGBM": "lgb", "Random Forest": "rf"}
MODEL_DISPLAY_NAMES = ["Ensemble", "XGBoost", "LightGBM", "Random Forest"]
MODEL_AUC = {
    "Ensemble": 0.9197,
    "XGBoost": 0.9128,
    "LightGBM": 0.9121,
    "Random Forest": 0.9068,
}

# scenarios for the app

PRESETS: dict[str, dict] = {
    "Custom": {
        "amount": 120.0, "productcd": "W", "card4": "visa", "card6": "debit",
        "devicetype": "desktop", "p_emaildomain": "gmail.com",
        "addr1": 320.0, "card2": 514.0, "c2": 1.0,
        "d1": 120.0, "d2": 1.0, "d4": 1.0, "d9": 0.5, "d10": 50.0, "d15": 200.0,
        "threshold": 0.40,
    },
    "Normal purchase": {
        # Typical low-risk transaction: known card, common email, stable address
        "amount": 75.0, "productcd": "W", "card4": "visa", "card6": "debit",
        "devicetype": "desktop", "p_emaildomain": "gmail.com",
        "addr1": 430.0, "card2": 480.0, "c2": 1.0,
        "d1": 300.0, "d2": 280.0, "d4": 300.0, "d9": 0.45, "d10": 60.0, "d15": 400.0,
        "threshold": 0.40,
    },
    "Card testing attack": {
        # Fraudster tests a stolen card with a tiny amount; many cards tied to same address
        "amount": 1.0, "productcd": "C", "card4": "mastercard", "card6": "credit",
        "devicetype": "mobile", "p_emaildomain": "anonymous.com",
        "addr1": 100.0, "card2": 100.0, "c2": 55.0,
        "d1": 0.0, "d2": 0.0, "d4": 0.0, "d9": 0.92, "d10": 600.0, "d15": 0.0,
        "threshold": 0.40,
    },
    "Account takeover": {
        # Large purchase on a device never used; address just changed; anonymous email
        "amount": 1950.0, "productcd": "W", "card4": "visa", "card6": "credit",
        "devicetype": "mobile", "p_emaildomain": "anonymous.com",
        "addr1": 100.0, "card2": 100.0, "c2": 12.0,
        "d1": 2.0, "d2": 2.0, "d4": 2.0, "d9": 0.88, "d10": 400.0, "d15": 1.0,
        "threshold": 0.40,
    },
    "Friendly fraud / chargeback risk": {
        # Legitimate-looking card; mid-size purchase; suspiciously short D1 after long dormancy
        "amount": 480.0, "productcd": "H", "card4": "visa", "card6": "credit",
        "devicetype": "desktop", "p_emaildomain": "hotmail.com",
        "addr1": 200.0, "card2": 200.0, "c2": 3.0,
        "d1": 1.0, "d2": 1.0, "d4": 1.0, "d9": 0.75, "d10": 200.0, "d15": 5.0,
        "threshold": 0.40,
    },
}


@st.cache_resource(show_spinner=False)
def load_imputer_and_meta() -> tuple:
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    with (MODEL_DIR / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return imputer, meta


@st.cache_resource(show_spinner=False)
def load_model(key: str):
    return joblib.load(MODEL_DIR / f"{key}_model.pkl")


# helpers
def build_sample(baseline: pd.Series, feature_names: list[str], overrides: dict) -> pd.DataFrame:
    row = baseline.copy()
    for k, v in overrides.items():
        if k in row.index:
            row[k] = v
    return pd.DataFrame([row], columns=feature_names)


def predict_proba_one(model, sample: pd.DataFrame) -> float:
    return float(model.predict_proba(sample)[:, 1][0])


imputer, meta = load_imputer_and_meta()
feature_names: list[str] = list(meta["feature_names"])
baseline = pd.Series(imputer.statistics_, index=feature_names)
ensemble_weights: dict = meta["ensemble_weights"]
optimal_threshold = float(meta.get("optimal_threshold", 0.69))


with st.sidebar:
    st.markdown("### 🛡️ Phoenix Fraud Engine")
    st.caption("Real-time transaction risk scoring")
    st.divider()

    st.subheader("Model")
    model_choice = st.radio(
        "Active model",
        MODEL_DISPLAY_NAMES,
        index=0,
        help="Ensemble combines all three models with optimised weights (XGB 0.4 · LGB 0.3 · RF 0.3).",
    )
    if model_choice == "Ensemble":
        w = ensemble_weights
        st.caption(f"Weights — XGB {w['xgb']} · LGB {w['lgb']} · RF {w['rf']}")

    st.divider()

    st.subheader("Scenario")
    preset_name = st.selectbox(
        "Load scenario",
        list(PRESETS.keys()),
        help="Presets load representative transactions. Switch to 'Custom' to adjust freely.",
    )
    p = PRESETS[preset_name]

    if preset_name != "Custom":
        scenario_notes = {
            "Normal purchase":
                "Low-risk baseline: known card, common email, stable address, normal hour.",
            "Card testing attack":
                "Stolen card tested with $1. 55 distinct cards tied to the same billing address (C2=55). Late night.",
            "Account takeover":
                "High-value purchase from a mobile device never seen before. Address changed yesterday. Anonymous email.",
            "Friendly fraud / chargeback risk":
                "Looks legitimate but D1=1 — card last used yesterday after a long gap, plus very recent address change.",
        }
        st.info(scenario_notes.get(preset_name, ""))

    k = preset_name

    st.subheader("Transaction")

    amount = st.slider("Amount (USD)", 0.0, 2500.0, p["amount"], 1.0, key=f"amount_{k}",
        help="Transaction amount. Fraud clusters at very small amounts (card testing) and very large amounts (cash-out).")

    product_label = st.selectbox("Product category", list(PRODUCTCD_LABELS.values()),
        index=list(PRODUCTCD_LABELS.keys()).index(p["productcd"]), key=f"productcd_{k}",
        help="W=Web, C=Digital content, R=Registration, S=Service, H=Home goods.")
    productcd = product_label[0]

    ts = st.datetime_input("Transaction timestamp", value=datetime(2026, 4, 15, 13, 0), key=f"ts_{k}",
        help="Hour and day-of-week are extracted as separate features. Late-night hours carry higher risk.")

    st.subheader("Payment instrument")

    card4 = st.selectbox("Card network", CARD4_OPTIONS,
        index=CARD4_OPTIONS.index(p["card4"]), key=f"card4_{k}",
        help="Visa and Mastercard dominate the dataset; Amex and Discover carry different fraud profiles.")
    card6 = st.selectbox("Card type", CARD6_OPTIONS,
        index=CARD6_OPTIONS.index(p["card6"]), key=f"card6_{k}",
        help="Credit cards show higher fraud rates than debit in card-not-present e-commerce.")
    devicetype = st.selectbox("Device type", DEVICETYPE_OPTIONS,
        index=DEVICETYPE_OPTIONS.index(p["devicetype"]), key=f"devicetype_{k}",
        help="Mobile sessions that don't match a card's historical device profile are a risk signal.")
    p_emaildomain = st.selectbox("Purchaser email domain", P_EMAILDOMAIN_OPTIONS,
        index=P_EMAILDOMAIN_OPTIONS.index(p["p_emaildomain"]), key=f"email_{k}",
        help="Encoded as relative frequency in the full dataset. anonymous.com → very low frequency → high risk.")

    st.subheader("Identity signals")

    addr1 = st.slider("Billing address (addr1)", 100.0, 540.0, p["addr1"], 1.0, key=f"addr1_{k}",
        help="Frequency rank of the billing ZIP code. Low rank = rare address = elevated risk.")
    card2 = st.slider("Card sub-identifier (card2)", 100.0, 600.0, p["card2"], 1.0, key=f"card2_{k}",
        help="Masked card attribute (possibly BIN). Low rank = rarely seen card = elevated risk.")

    st.subheader("Network counters")

    c2 = st.slider("C2 — Cards linked to this address", 0.0, 100.0, p["c2"], 1.0, key=f"c2_{k}",
        help="Vesta counter: distinct cards ever used at this billing address. "
             "C2 > 10 is unusual; C2 > 30 strongly suggests a mule address used in carding operations.")

    st.subheader("Activity history (D-signals)")
    st.caption("Days elapsed since past events, normalised by daily average to remove temporal drift.")

    d1 = st.slider("D1 — Days since last transaction", 0.0, 636.0, p["d1"], 1.0, key=f"d1_{k}",
        help="Very short gap → card testing or automated attack. Very long gap → dormant card suddenly reactivated.")
    d2 = st.slider("D2 — Days since last use of this card", 0.0, 636.0, p["d2"], 1.0, key=f"d2_{k}",
        help="One of the top-8 most predictive features. A card appearing after a long dormancy is a strong signal.")
    d4 = st.slider("D4 — Days since last cardholder transaction", -122.0, 700.0, p["d4"], 1.0, key=f"d4_{k}",
        help="Broader activity measure. Negative after normalisation = transacted more recently than the daily average.")
    d9 = st.slider("D9 — Time-of-day signal", 0.0, 0.958, p["d9"], 0.001, key=f"d9_{k}",
        help="Fractional day: 0.0=midnight · 0.5=noon · 0.92≈10 PM. Late-night activity is a known fraud pattern.")
    d10 = st.slider("D10 — Days since last address verification", 0.0, 705.0, p["d10"], 1.0, key=f"d10_{k}",
        help="A very long gap means the address wasn't re-verified on recent transactions.")
    d15 = st.slider("D15 — Days since last address change", -83.0, 709.0, p["d15"], 1.0, key=f"d15_{k}",
        help="Top-10 feature. D15 near 0 = address changed very recently — a hallmark of account takeover.")

    st.divider()
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, p["threshold"], 0.01, key=f"threshold_{k}",
        help=f"Score ≥ threshold → DECLINE. "
             f"Production-calibrated value (F1-optimal on validation set): {optimal_threshold:.2f}. "
             f"Lowering catches more fraud at the cost of more false positives.",
    )


#feature vector

sample = build_sample(baseline, feature_names, {
    "TransactionAmt": amount,
    "ProductCD": PRODUCTCD_CODE_MAP[productcd],
    "card4": CARD4_CODE_MAP[card4],
    "card6": CARD6_CODE_MAP[card6],
    "DeviceType": DEVICETYPE_CODE_MAP[devicetype],
    "Transaction_hour": int(ts.hour),
    "Transaction_day": int(ts.day),
    "Transaction_weekday": int(ts.weekday()),
    "addr1_FE": addr1,
    "card2_FE": card2,
    "P_emaildomain_FE": P_EMAILDOMAIN_FE_MAP.get(p_emaildomain, 0.0),
    "C2": c2,
    "D2": d2,
    "D9": d9,
    "D15": d15,
    "D1_norm": d1,
    "D2_norm": d2,
    "D4_norm": d4,
    "D9_norm": d9,
    "D10_norm": d10,
    "D15_norm": d15,
})


ensemble_warnings: list[str] = []
score_error: str | None = None
risk_score = 0.0

if model_choice == "Ensemble":
    total_weight = 0.0
    for k_model, w in ensemble_weights.items():
        try:
            risk_score += w * predict_proba_one(load_model(k_model), sample)
            total_weight += w
        except Exception as exc:
            ensemble_warnings.append(f"**{k_model}** skipped — {exc}")
    if total_weight == 0.0:
        score_error = "No models could be loaded."
    elif total_weight < 1.0:
        risk_score /= total_weight
else:
    try:
        risk_score = predict_proba_one(load_model(MODEL_KEY_MAP[model_choice]), sample)
    except Exception as exc:
        score_error = str(exc)

is_fraud = risk_score >= threshold


st.markdown("## 🛡️ Phoenix Fraud Engine")
st.caption(
    f"**{ts.strftime('%Y-%m-%d %H:%M')}** · "
    f"${amount:,.2f} · "
    f"{card4.title()} {card6.title()} · "
    f"{PRODUCTCD_LABELS[productcd]} · "
    f"Model: **{model_choice}** (AUC {MODEL_AUC[model_choice]:.2f}) · "
    f"Scenario: *{preset_name}*"
)

if score_error:
    st.error(f"Scoring error: {score_error}")
    st.stop()

for warn in ensemble_warnings:
    st.warning(f"Ensemble: {warn}")

st.divider()

if is_fraud:
    st.error(
        f" DECLINE / REFER FOR REVIEW — Risk score **{risk_score:.1%}** exceeds threshold {threshold:.0%}. "
        f"Transaction flagged for analyst review before processing."
    )
else:
    st.success(
        f" APPROVED — Risk score **{risk_score:.1%}** is below threshold {threshold:.0%}. "
        f"Transaction passes automated screening."
    )

st.write("")
left, right = st.columns([1, 1], gap="large")


with left:
    st.subheader("Risk assessment")

    c1, c2_col, c3 = st.columns(3)
    c1.metric("Risk score", f"{risk_score:.1%}")
    c2_col.metric("Decision", "Decline / Review" if is_fraud else "Approved")
    c3.metric("Threshold", f"{threshold:.0%}")

    if model_choice == "Ensemble":
        st.subheader("Model committee scores")
        st.caption("Individual model votes before weighting.")
        s1, s2, s3 = st.columns(3)
        for col, (name, key) in zip([s1, s2, s3], MODEL_KEY_MAP.items()):
            try:
                s = predict_proba_one(load_model(key), sample)
                col.metric(name, f"{s:.1%}")
            except Exception:
                col.metric(name, "N/A")

    st.subheader("Input signal summary")
    st.caption(
        "263 features sent to the model. Fields not in the sidebar keep the imputer's "
        "training fill value (−999 = missing / no identity data available)."
    )
    exposed = {
        "Amount (USD)": f"${amount:,.2f}",
        "Product": PRODUCTCD_LABELS[productcd],
        "Card network": card4.title(),
        "Card type": card6.title(),
        "Device": devicetype.title(),
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "Email domain": p_emaildomain,
        "addr1 (freq. rank)": f"{addr1:.0f}",
        "card2 (freq. rank)": f"{card2:.0f}",
        "C2 — cards/address": f"{c2:.0f}",
        "D1 — last txn (days)": f"{d1:.0f}",
        "D2 — last card txn (days)": f"{d2:.0f}",
        "D4 — cardholder activity (days)": f"{d4:.0f}",
        "D9 — time-of-day (fraction)": f"{d9:.3f}",
        "D10 — last addr. verification (days)": f"{d10:.0f}",
        "D15 — last addr. change (days)": f"{d15:.0f}",
    }
    st.dataframe(
        pd.DataFrame(list(exposed.items()), columns=["Signal", "Value"]).set_index("Signal"),
        use_container_width=True,
    )


with right:
    st.subheader("Model card")

    st.metric("Validation AUC", f"{MODEL_AUC[model_choice]:.4f}")

    if model_choice == "Ensemble":
        st.caption(
            "Weighted average of three independent classifiers trained on a temporal 80/20 split. "
            "Outperforms every individual model by capturing complementary error patterns "
            "across different tree-building strategies."
        )
        wa, wb, wc = st.columns(3)
        wa.metric("XGBoost weight", ensemble_weights["xgb"])
        wb.metric("LightGBM weight", ensemble_weights["lgb"])
        wc.metric("RF weight", ensemble_weights["rf"])
        aa, ab, ac = st.columns(3)
        aa.metric("XGBoost AUC", f"{MODEL_AUC['XGBoost']:.4f}")
        ab.metric("LightGBM AUC", f"{MODEL_AUC['LightGBM']:.4f}")
        ac.metric("Random Forest AUC", f"{MODEL_AUC['Random Forest']:.4f}")
    else:
        try:
            model_obj = load_model(MODEL_KEY_MAP[model_choice])
            params = model_obj.get_params() if hasattr(model_obj, "get_params") else {}

            if model_choice == "XGBoost":
                st.caption(
                    "Sequential gradient boosted trees. Hyperparameters tuned via randomised search "
                    "with early stopping on the temporal validation set."
                )
                p1, p2, p3 = st.columns(3)
                p1.metric("Boosting rounds", model_obj.n_estimators)
                p2.metric("Learning rate", f"{params.get('learning_rate', 0):.4f}")
                p3.metric("Max depth", params.get("max_depth", "—"))
                p4, p5 = st.columns(2)
                p4.metric("Subsample", f"{params.get('subsample', 0):.2f}")
                p5.metric("Col sample", f"{params.get('colsample_bytree', 0):.2f}")
            elif model_choice == "LightGBM":
                st.caption(
                    "Leaf-wise gradient boosted trees. is_unbalance=True handles the class imbalance. "
                    "Early stopping on the temporal validation set."
                )
                p1, p2, p3 = st.columns(3)
                p1.metric("Boosting rounds", model_obj.n_estimators)
                p2.metric("Learning rate", f"{params.get('learning_rate', 0):.4f}")
                p3.metric("Num leaves", params.get("num_leaves", "—"))
                p4, p5 = st.columns(2)
                p4.metric("Subsample", f"{params.get('subsample', 0):.2f}")
                p5.metric("Min child samples", params.get("min_child_samples", "—"))
            elif model_choice == "Random Forest":
                st.caption(
                    "Bagged parallel trees — structurally different from the boosting models, "
                    "providing complementary diversity in the ensemble. class_weight='balanced' "
                    "compensates for the class imbalance."
                )
                p1, p2 = st.columns(2)
                p1.metric("Trees", model_obj.n_estimators)
                p2.metric("Class weight", str(params.get("class_weight", "None")))
                p3, p4 = st.columns(2)
                p3.metric("Max depth", params.get("max_depth", "—"))
                mf = params.get("max_features", "—")
                p4.metric("Max features", f"{mf:.2f}" if isinstance(mf, float) else str(mf))
        except Exception:
            pass

    st.subheader("Top 20 feature importance (ensemble-weighted)")
    st.caption(
        "Weighted average of XGB · LGB · RF importances. "
        "TransactionAmt and C2 dominate; normalised D-columns confirm the temporal-drift "
        "correction added predictive signal."
    )
    img_path = APP_DIR / "assets" / "topfeatures.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Feature importance chart not found at assets/topfeatures.png")


st.divider()
with st.expander("Dataset & methodology", expanded=False):
    st.markdown(
        """
        **Source** — IEEE-CIS Fraud Detection Challenge, provided by Vesta Corporation.

        **Volume** — 590,540 e-commerce transactions · **3.5 % fraud rate**.

        **Schema** — Two tables merged on `TransactionID`:
        - *Transaction table*: payment amounts, product codes, masked card attributes (card1–card6),
          Vesta counter variables (C1–C14), time-delta variables (D1–D15), and anonymised V-columns.
        - *Identity table*: device type, device fingerprint, network attributes (id_01–id_38).
          Not all transactions have identity data; missing fields are filled with −999.

        >  **Note on variable names** — The labels assigned to D and C variables in this interface
        > (e.g. *"days since last transaction"*, *"cards per address"*) are **interpretive names**
        > drawn from domain knowledge, not confirmed definitions.
        > Vesta Corporation masks the exact meaning of these features; they represent complex
        > interaction signals that may encode combinations of behaviours rather than a single
        > raw measurement. Treat the labels as a useful approximation, not ground truth.

        **Variable groups in the model:**

        | Group | Examples | What they capture |
        |---|---|---|
        | Amount | `TransactionAmt`, `TransactionAmt_cents` | Size and rounding patterns |
        | Product & card | `ProductCD`, `card4`, `card6` | Category and instrument type |
        | Temporal | `Transaction_hour`, `Transaction_day`, `Transaction_weekday` | Time-of-day and weekly cycles |
        | D-signals (raw) | `D2`, `D9`, `D15` | Time-delta variables from Vesta (interpreted as days since past events) |
        | D-signals (norm) | `D1_norm`–`D15_norm` | Same, minus daily mean — removes temporal drift across the dataset |
        | C-counters | `C2`, `C3`, `C9` | Vesta counter variables (interpreted as network-level aggregates) |
        | Frequency-encoded | `addr1_FE`, `card2_FE`, `P_emaildomain_FE` | Rarity of address, card sub-ID, email domain in the training population |
        | UID aggregates | `TransactionAmt_card1_mean`, `D9_card1_std` | Behavioural baseline per card / card+address hierarchy |
        | Uniqueness counters | `card1_addr1_nunique`, `card1_P_emaildomain_nunique` | Distinct addresses/emails observed per card |
        | Identity | `id_01`–`id_38`, `DeviceType`, `DeviceInfo_FE` | Device and network fingerprint |
        | V-columns | subset of `V1`–`V338` | Masked Vesta proprietary signals |

        **Feature selection** — Highly collinear variable pairs (|r| > 0.7) reduced via greedy selection:
        for each pair, the variable with lower correlation to fraud is dropped.

        **Training split** — Temporal 80/20: trains on the oldest 80 % of transactions,
        evaluates on the most recent 20 % it has never seen — simulating production deployment.

        **Production threshold** — The F1-optimal threshold on the validation ensemble is **0.69**.
        This is intentionally conservative: at a low base fraud rate, false positives (blocking good customers)
        are expensive, so the model requires high confidence before declining.
        The default threshold in this tool is set to **0.40** to make the scoring range visible across scenarios.
        """
    )
