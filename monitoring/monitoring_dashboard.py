# ============================================================
# MONITORING DASHBOARD — Resume Screener AI
# ============================================================
# 5 Sections:
#   1. Real-Time Monitoring Alerts
#   2. Champion-Challenger Log
#   3. KPIs + Charts (F1, accuracy, coverage)
#   4. PSI Drift Monitoring
#   5. Recent Predictions Log
#
# Sidebar: live resume screening via API
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

st.set_page_config(
    page_title = "Resume Screener AI — Dashboard",
    page_icon  = "📄",
    layout     = "wide",
)
st.title("📄 Resume Screener AI — Monitoring Dashboard")
st.caption("HR Tech · Naukri.com / Darwinbox / Workday equivalent stack")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "RESUME_SCREENER_API_URL",
    "http://localhost:8000"
) + "/predict"

# ── PSI thresholds ────────────────────────────────────────────
PSI_MODERATE     = 0.10
PSI_HIGH         = 0.20
SCORE_MEAN_ALERT = 0.50

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "resume_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs",          "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "resume_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "resume_models", "challenger_log.json")
REGISTRY_PATH   = os.path.join(BASE_DIR, "resume_models", "latest_model.json")
EXP_RESULTS     = os.path.join(BASE_DIR, "resume_models", "model_experiment_results.csv")


# ============================================================
# SIDEBAR — LIVE SCREENING
# ============================================================

st.sidebar.header("🔮 Screen a Resume")
st.sidebar.caption("Paste resume text + optional JD to get a match score")

resume_text = st.sidebar.text_area(
    "Resume Text", height=200,
    value="Python developer with 5 years of experience in machine learning, "
          "TensorFlow, scikit-learn, pandas, SQL and data visualization. "
          "B.Tech Computer Science from IIT Bombay. Led NLP pipeline projects "
          "for text classification. Strong analytical and communication skills."
)
jd_text = st.sidebar.text_area(
    "Job Description (optional)", height=100,
    value="Looking for a Python Developer with machine learning and SQL skills. "
          "Experience with TensorFlow and scikit-learn preferred. 3+ years required."
)
years_req   = st.sidebar.number_input("Min. Years Required", min_value=0, max_value=20, value=0)

if st.sidebar.button("🚀 Screen Resume"):
    if not resume_text.strip():
        st.sidebar.error("Please paste resume text.")
    else:
        payload = {
            "resume_text":     resume_text,
            "job_description": jd_text,
            "years_required":  years_req,
        }
        with st.sidebar:
            with st.spinner("Calling API... First request may take 30–60s (cold start)"):
                try:
                    response = requests.post(API_URL, json=payload, timeout=90)
                    if response.status_code == 200:
                        res    = response.json()
                        band   = res.get("match_band", "WEAK")
                        rec    = res.get("recommendation", "REJECT")
                        colors = {"SHORTLIST": "green", "REVIEW": "orange", "REJECT": "red"}
                        color  = colors.get(rec, "gray")

                        st.success("Prediction received!")
                        st.markdown(f"**Predicted Category:** `{res.get('predicted_category')}`")
                        st.markdown(f"**Final Score:** `{res.get('final_score')}`")
                        st.markdown(f"**Match Band:** `{band}`")
                        st.markdown(
                            f"<h3 style='color:{color}'>Decision: {rec}</h3>",
                            unsafe_allow_html=True
                        )
                        if res.get("keyword_overlap") is not None:
                            st.markdown(f"**Keyword Overlap:** `{res['keyword_overlap']:.1%}`")
                        if res.get("rule_triggered"):
                            st.warning(f"Rule triggered: {res['rule_triggered']}")

                        st.markdown("**Top-3 Categories:**")
                        for t in res.get("top3_categories", []):
                            st.markdown(f"- `{t['category']}` — {t['confidence']:.1%}")

                    else:
                        st.error(f"API error: HTTP {response.status_code}")
                        st.code(response.text[:300])

                except requests.exceptions.Timeout:
                    st.warning("Request timed out. Render is waking up. Wait 30s and retry.")
                except Exception as e:
                    st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Section 1 — Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)

    if "score" in df_monitor.columns:
        avg_score = df_monitor["score"].mean()

        if avg_score < SCORE_MEAN_ALERT:
            st.error(
                f"⚠️ ALERT: Average match confidence score ({avg_score:.3f}) "
                f"is below threshold ({SCORE_MEAN_ALERT}). "
                "Model may be uncertain — consider retraining."
            )
            alerts_found = True

    if "correct" in df_monitor.columns:
        acc = df_monitor["correct"].mean()
        if acc < 0.75:
            st.error(
                f"⚠️ ALERT: Recent prediction accuracy ({acc:.1%}) is below 75%. "
                "Production drift detected."
            )
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    high_psi = df_psi[df_psi["psi"] >= PSI_HIGH] if "psi" in df_psi.columns else pd.DataFrame()
    if not high_psi.empty:
        for _, row in high_psi.iterrows():
            st.error(
                f"⚠️ PSI CRITICAL: Feature '{row.get('feature', 'N/A')}' "
                f"PSI={row['psi']:.3f} >= {PSI_HIGH} — RETRAIN RECOMMENDED"
            )
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems nominal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION-CHALLENGER LOG
# ============================================================

st.markdown("---")
st.subheader("🏆 Section 2 — Champion-Challenger System")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        history = json.load(f)

    if history:
        latest = history[-1]
        decision = latest.get("decision", "N/A")
        dec_color = "🟢" if decision == "PROMOTED" else "🔴"

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Decision", f"{dec_color} {decision}")
        col2.metric("Challenger F1", f"{latest.get('challenger_f1', 'N/A')}")
        col3.metric("Champion F1",   f"{latest.get('champion_f1',   'N/A')}")

        gates = latest.get("gates", {})
        if gates:
            g_col1, g_col2, g_col3 = st.columns(3)
            g_col1.metric("Gate 1: F1 Improvement", "✅" if gates.get("f1_improvement_passed") else "❌")
            g_col2.metric("Gate 2: Accuracy",        "✅" if gates.get("accuracy_passed")       else "❌")
            g_col3.metric("Gate 3: Train-Test Gap",   "✅" if gates.get("gap_passed")            else "❌")

        st.markdown(f"**Reason:** {latest.get('reason', 'N/A')}")

        with st.expander("Full Challenger History"):
            st.dataframe(pd.DataFrame(history), use_container_width=True)
else:
    st.info("No challenger log found — train a model first.")


# ============================================================
# SECTION 3 — KPIs + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Section 3 — Model KPIs + Charts")

if os.path.exists(EXP_RESULTS):
    df_exp = pd.read_csv(EXP_RESULTS, index_col=0)

    kpi_cols = [c for c in ["f1_macro", "test_acc", "top3_acc", "roc_auc_ovr"] if c in df_exp.columns]

    if kpi_cols:
        # KPI summary for best model
        best_row = df_exp["f1_macro"].idxmax() if "f1_macro" in df_exp.columns else df_exp.index[0]
        cols     = st.columns(len(kpi_cols))
        for i, kpi in enumerate(kpi_cols):
            val = df_exp.loc[best_row, kpi]
            cols[i].metric(kpi.replace("_", " ").title(), f"{val:.4f}")

        # Bar chart: F1 macro per model
        if "f1_macro" in df_exp.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_exp["f1_macro"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("F1 Macro")
            ax.set_title("Model Comparison — F1 Macro")
            ax.axvline(df_exp["f1_macro"].max(), color="red", linestyle="--", alpha=0.5, label="Best")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

elif os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)
    if "correct" in df_monitor.columns:
        col1, col2 = st.columns(2)
        col1.metric("Recent Accuracy",    f"{df_monitor['correct'].mean():.1%}")
        col2.metric("Avg Match Score",    f"{df_monitor['score'].mean():.3f}" if "score" in df_monitor.columns else "N/A")

        if "pred" in df_monitor.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_monitor["pred"].value_counts().head(15).plot(kind="barh", ax=ax, color="teal")
            ax.set_xlabel("Count")
            ax.set_title("Prediction Distribution — Top Categories")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
else:
    st.info("No experiment results found — run training pipeline first.")


# ============================================================
# SECTION 4 — PSI DRIFT MONITORING
# ============================================================

st.markdown("---")
st.subheader("📉 Section 4 — PSI Drift Monitoring")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)

    if not df_psi.empty:
        def _psi_status_color(psi_val):
            if psi_val >= PSI_HIGH:
                return "🔴 HIGH — Retrain"
            elif psi_val >= PSI_MODERATE:
                return "🟡 MODERATE — Monitor"
            return "🟢 STABLE"

        df_psi["Status"] = df_psi["psi"].apply(_psi_status_color)

        st.dataframe(df_psi.style.format({"psi": "{:.4f}"}), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(df_psi["feature"], df_psi["psi"], color=[
            "red" if p >= PSI_HIGH else "orange" if p >= PSI_MODERATE else "green"
            for p in df_psi["psi"]
        ])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"High ({PSI_HIGH})")
        ax.set_xlabel("PSI")
        ax.set_title("Feature Population Stability Index")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
else:
    st.info("No PSI drift report found — run training pipeline first.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("🔍 Section 5 — Recent Predictions Log")

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)

    if not df_log.empty:
        st.markdown(f"**Total predictions logged:** {len(df_log)}")

        # Summary stats
        if "recommendation" in df_log.columns:
            dist = df_log["recommendation"].value_counts()
            cols = st.columns(len(dist))
            for i, (rec, cnt) in enumerate(dist.items()):
                emoji = {"SHORTLIST": "✅", "REVIEW": "⚠️", "REJECT": "❌"}.get(rec, "📌")
                cols[i].metric(f"{emoji} {rec}", cnt)

        # Recent table
        display_cols = [c for c in [
            "timestamp", "predicted_category", "final_score",
            "match_band", "recommendation", "rule_triggered"
        ] if c in df_log.columns]

        st.dataframe(
            df_log[display_cols].tail(50).sort_values(
                "timestamp", ascending=False
            ) if "timestamp" in df_log.columns else df_log[display_cols].tail(50),
            use_container_width=True,
        )

        # Score distribution chart
        if "final_score" in df_log.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            df_log["final_score"].dropna().hist(bins=20, ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel("Final Match Score")
            ax.set_ylabel("Count")
            ax.set_title("Match Score Distribution — Recent Predictions")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
else:
    st.info("No prediction logs found. Send some requests to the API first.")

st.markdown("---")
st.caption("Resume Screener AI · Built by Narendra Kalam · MSc CS (Gold Medalist)")
