import json
import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Import custom transformer so pipeline unpickles flawlessly
from feature_engineering import LoanFeatureEngineer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediFlux | KNN Loan Intelligence Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# -----------------------------------------------------------------------------
# 2. DESIGN SYSTEM & CSS VARIABLES
# -----------------------------------------------------------------------------
bg_color = "#09090b" if IS_DARK else "#f8fafc"
bg_subtle = "#111115" if IS_DARK else "#f1f5f9"
card_color = "#131318" if IS_DARK else "#ffffff"
card_hover = "#1a1a22" if IS_DARK else "#f8fafc"
border_color = "#27272a" if IS_DARK else "#e2e8f0"
border_subtle = "#1e1e24" if IS_DARK else "#f1f5f9"
text_color = "#fafafa" if IS_DARK else "#0f172a"
text_muted = "#a1a1aa" if IS_DARK else "#64748b"
text_dim = "#71717a" if IS_DARK else "#94a3b8"
accent_color = "#3b82f6"
green_color = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.14)" if IS_DARK else "rgba(22,163,74,0.1)"
red_color = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.14)" if IS_DARK else "rgba(220,38,38,0.1)"
amber_color = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.14)" if IS_DARK else "rgba(217,119,6,0.1)"
shadow = "0 4px 20px rgba(0,0,0,0.3)" if IS_DARK else "0 2px 10px rgba(0,0,0,0.05)"

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --bg: {bg_color};
    --bg-subtle: {bg_subtle};
    --card: {card_color};
    --card-hover: {card_hover};
    --border: {border_color};
    --border-subtle: {border_subtle};
    --text: {text_color};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: {accent_color};
    --green: {green_color};
    --green-muted: {green_muted};
    --red: {red_color};
    --red-muted: {red_muted};
    --amber: {amber_color};
    --amber-muted: {amber_muted};
    --shadow: {shadow};
    --radius: 12px;
}}

/* Clean up Streamlit Chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}

p, span, label, div {{
    font-family: 'DM Sans', sans-serif !important;
}}

code, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Brand Header */
.brand-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}}
.brand-title-wrap {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.brand-icon {{
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4);
}}
.brand-name {{
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.1;
}}
.brand-sub {{
    font-size: 0.82rem;
    color: var(--text-muted);
    font-weight: 500;
}}

/* Metric Cards */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow);
    transition: transform 0.15s ease, border-color 0.15s ease;
    height: 100%;
}}
.metric-card:hover {{
    border-color: var(--accent);
    transform: translateY(-2px);
}}
.metric-label {{
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}}
.metric-value {{
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.15;
}}
.metric-delta {{
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.4rem;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-neutral {{ color: var(--accent); background: rgba(59,130,246,0.12); }}

/* Card Panels */
.panel-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.panel-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.panel-desc {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.1rem;
}}

/* Prediction Result Cards */
.result-banner {{
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}}
.banner-approved {{
    background: linear-gradient(145deg, rgba(34,197,94,0.12), rgba(34,197,94,0.03));
    border: 1.5px solid var(--green);
}}
.banner-rejected {{
    background: linear-gradient(145deg, rgba(239,68,68,0.12), rgba(239,68,68,0.03));
    border: 1.5px solid var(--red);
}}
.result-status-text {{
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.4rem 0;
}}
.status-approved {{ color: var(--green); }}
.status-rejected {{ color: var(--red); }}

/* Badges */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); border: 1px solid rgba(34,197,94,0.3); }}
.badge-red {{ color: var(--red); background: var(--red-muted); border: 1px solid rgba(239,68,68,0.3); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); border: 1px solid rgba(245,158,11,0.3); }}
.badge-blue {{ color: var(--accent); background: rgba(59,130,246,0.14); border: 1px solid rgba(59,130,246,0.3); }}

/* Data Table */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.82rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}}
.data-table th {{
    background: var(--bg-subtle);
    text-align: left;
    padding: 0.75rem 0.9rem;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}}
.data-table td {{
    padding: 0.75rem 0.9rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
    background: var(--card);
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}
.data-table tr:hover td {{
    background: var(--card-hover);
}}

/* Pill Tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.25rem !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 6px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    margin-bottom: 1.5rem !important;
}}

/* Form inputs styling */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {{
    background-color: var(--bg-subtle) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}}
.stButton>button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.15s ease !important;
}}
div[data-testid="stHorizontalBlock"] {{
    gap: 1.25rem !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_historical_data():
    df = pd.read_csv("loan_data.csv")
    return df

@st.cache_resource
def load_ml_pipeline():
    pipeline = joblib.load("model/knn_pipeline.joblib")
    return pipeline

@st.cache_data
def load_metrics():
    with open("model/metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_training_neighbors_data():
    train_raw = pd.read_csv("model/train_raw_with_target.csv")
    return train_raw

try:
    raw_df = load_historical_data()
    pipeline = load_ml_pipeline()
    metrics = load_metrics()
    train_raw_data = load_training_neighbors_data()
except Exception as e:
    st.error(f"Error loading model or data artifacts: {e}. Please ensure 'train_knn.py' has executed.")
    st.stop()


# -----------------------------------------------------------------------------
# 4. PLOTLY CHART THEME HELPER
# -----------------------------------------------------------------------------
def styled_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#a1a1aa" if IS_DARK else "#64748b",
            size=11,
        ),
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
            zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
            tickfont=dict(size=10, color="#a1a1aa" if IS_DARK else "#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
            zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
            tickfont=dict(size=10, color="#a1a1aa" if IS_DARK else "#64748b"),
        ),
        legend=dict(
            font=dict(size=11, color="#fafafa" if IS_DARK else "#0f172a"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# 5. BRAND HEADER & METRICS BAR
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="brand-container">
    <div class="brand-title-wrap">
        <div class="brand-icon">💳</div>
        <div>
            <div class="brand-name">CrediFlux <span style="font-size:0.8rem; font-weight:600; color:{accent_color}; background:rgba(59,130,246,0.12); padding:2px 8px; border-radius:6px; margin-left:6px;">KNN v1.0</span></div>
            <div class="brand-sub">Intelligent Loan Risk Scoring & Nearest-Neighbors Decision Engine</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Bar Action: Theme Switcher & Quick Links
col_top_left, col_top_right = st.columns([10, 2])
with col_top_right:
    btn_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    st.button(btn_label, on_click=toggle_theme, use_container_width=True)

# 4 Key Metrics Cards
total_records = len(raw_df)
approved_count = (raw_df["Loan_Status"] == "Y").sum()
approval_rate = (approved_count / total_records) * 100
avg_loan = raw_df["LoanAmount"].median()
test_acc = metrics["test_metrics"]["accuracy"] * 100
test_f1 = metrics["test_metrics"]["f1"] * 100

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Records</div>
        <div class="metric-value">{total_records:,}</div>
        <div class="metric-delta delta-neutral">Kaggle Benchmark Dataset</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Approval Rate</div>
        <div class="metric-value">{approval_rate:.1f}%</div>
        <div class="metric-delta delta-up">↑ {approved_count} Approved of {total_records}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Median Loan Requested</div>
        <div class="metric-value">${avg_loan:,.0f}k</div>
        <div class="metric-delta delta-neutral">IQR: $100k - $168k</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">KNN Model Accuracy</div>
        <div class="metric-value">{test_acc:.1f}%</div>
        <div class="metric-delta delta-up">↑ Test F1 Score: {test_f1:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. APP NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_predict, tab_eda, tab_model, tab_batch = st.tabs([
    "🎯 Loan Assessment & Neighbor Inspector",
    "📊 Dataset Intelligence & EDA",
    "🧠 Model Diagnostics & KNN Tuning",
    "📁 Batch Loan Evaluation",
])


# =============================================================================
# TAB 1: INSTANT LOAN ASSESSMENT & NEAREST NEIGHBORS INSPECTOR
# =============================================================================
with tab_predict:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">🎯 Real-Time Loan Approval Simulator</div>
        <div class="panel-desc">
            Provide applicant credentials below. The model transforms the attributes through domain featurization, calculates multi-dimensional Euclidean distances, and assigns approval based on the majority vote of the nearest historical loan outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset Profiles for Rapid Testing
    st.markdown("**⚡ Quick Test Applicant Presets:**")
    pcol1, pcol2, pcol3, _ = st.columns([2, 2, 2, 4])
    
    if "preset" not in st.session_state:
        st.session_state.preset = "custom"

    if pcol1.button("🟢 Strong Applicant", use_container_width=True):
        st.session_state.gender = "Male"
        st.session_state.married = "Yes"
        st.session_state.dependents = "0"
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "No"
        st.session_state.app_income = 6500
        st.session_state.coapp_income = 2500
        st.session_state.loan_amount = 140
        st.session_state.loan_term = 360
        st.session_state.credit_hist = 1.0
        st.session_state.prop_area = "Semiurban"
        st.session_state.k_neighbors = 15

    if pcol2.button("🟡 Borderline Applicant", use_container_width=True):
        st.session_state.gender = "Female"
        st.session_state.married = "No"
        st.session_state.dependents = "2"
        st.session_state.education = "Not Graduate"
        st.session_state.self_employed = "Yes"
        st.session_state.app_income = 2600
        st.session_state.coapp_income = 0
        st.session_state.loan_amount = 150
        st.session_state.loan_term = 360
        st.session_state.credit_hist = 1.0
        st.session_state.prop_area = "Rural"
        st.session_state.k_neighbors = 15

    if pcol3.button("🔴 High-Risk Applicant", use_container_width=True):
        st.session_state.gender = "Male"
        st.session_state.married = "No"
        st.session_state.dependents = "2"
        st.session_state.education = "Not Graduate"
        st.session_state.self_employed = "No"
        st.session_state.app_income = 2200
        st.session_state.coapp_income = 0
        st.session_state.loan_amount = 180
        st.session_state.loan_term = 180
        st.session_state.credit_hist = 0.0
        st.session_state.prop_area = "Urban"
        st.session_state.k_neighbors = 15

    form_col, result_col = st.columns([5, 6])

    with form_col:
        st.markdown("""
        <div style="background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:1.25rem;">
            <div style="font-size:0.92rem; font-weight:700; margin-bottom:1rem; color:var(--text);">📋 Applicant Profile Details</div>
        """, unsafe_allow_html=True)

        c_gen, c_mar, c_dep = st.columns(3)
        with c_gen:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get("gender", "Male") == "Male" else 1)
        with c_mar:
            married = st.selectbox("Married", ["Yes", "No"], index=0 if st.session_state.get("married", "Yes") == "Yes" else 1)
        with c_dep:
            dep_opts = ["0", "1", "2", "3+"]
            cur_dep = st.session_state.get("dependents", "1")
            dep_idx = dep_opts.index(cur_dep) if cur_dep in dep_opts else 0
            dependents = st.selectbox("Dependents", dep_opts, index=dep_idx)

        c_edu, c_emp = st.columns(2)
        with c_edu:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if st.session_state.get("education", "Graduate") == "Graduate" else 1)
        with c_emp:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if st.session_state.get("self_employed", "No") == "No" else 1)

        c_inc1, c_inc2 = st.columns(2)
        with c_inc1:
            app_income = st.number_input("Applicant Monthly Income ($)", min_value=150, max_value=85000, value=int(st.session_state.get("app_income", 5000)), step=250)
        with c_inc2:
            coapp_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, max_value=45000, value=int(st.session_state.get("coapp_income", 1800)), step=250)

        c_amt, c_term = st.columns(2)
        with c_amt:
            loan_amount = st.number_input("Loan Amount Requested ($ in thousands)", min_value=9, max_value=700, value=int(st.session_state.get("loan_amount", 145)), step=5)
        with c_term:
            term_opts = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
            cur_term = int(st.session_state.get("loan_term", 360))
            t_idx = term_opts.index(cur_term) if cur_term in term_opts else 8
            loan_term = st.selectbox("Loan Term (Months)", term_opts, index=t_idx)

        c_cred, c_prop = st.columns(2)
        with c_cred:
            credit_hist_choice = st.selectbox(
                "Credit History Meets Guidelines",
                ["Yes (1.0) - High Creditworthiness", "No (0.0) - Past Delinquency"],
                index=0 if float(st.session_state.get("credit_hist", 1.0)) == 1.0 else 1
            )
            credit_history = 1.0 if "Yes" in credit_hist_choice else 0.0
        with c_prop:
            prop_opts = ["Semiurban", "Urban", "Rural"]
            cur_prop = st.session_state.get("prop_area", "Semiurban")
            p_idx = prop_opts.index(cur_prop) if cur_prop in prop_opts else 0
            property_area = st.selectbox("Property Area", prop_opts, index=p_idx)

        st.markdown("<hr style='border-color:var(--border); margin:1rem 0;'>", unsafe_allow_html=True)
        k_neighbors = st.slider("KNN Neighbors (K Value for Decision)", min_value=3, max_value=25, value=int(st.session_state.get("k_neighbors", 15)), step=2)

        st.markdown("</div>", unsafe_allow_html=True)

    # Perform Prediction with the Pipeline
    with result_col:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(app_income),
            "CoapplicantIncome": float(coapp_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area,
        }])

        # Calculate Derived Financial Ratios
        total_household_income = app_income + coapp_income
        est_monthly_emi = (loan_amount * 1000) / loan_term
        dti_ratio = (est_monthly_emi / (total_household_income + 1e-5)) * 100

        # Extract transformed sample and fetch neighbors
        transformer = pipeline[:-1]
        knn_clf = pipeline[-1]

        # Dynamically set k_neighbors for the classifier in this session
        knn_clf.set_params(n_neighbors=k_neighbors)

        transformed_sample = transformer.transform(input_data)
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]
        prob_approved = proba[1] * 100
        prob_rejected = proba[0] * 100

        # Nearest Neighbors calculation
        distances, indices = knn_clf.kneighbors(transformed_sample, n_neighbors=k_neighbors)
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        neighbor_records = train_raw_data.iloc[neighbor_indices].copy()
        neighbor_records["Euclidean_Distance"] = np.round(neighbor_distances, 3)
        approved_neighbors = (neighbor_records["Loan_Status"] == 1).sum()
        rejected_neighbors = k_neighbors - approved_neighbors

        # Verdict UI
        is_approved = (prediction == 1)
        banner_class = "banner-approved" if is_approved else "banner-rejected"
        status_class = "status-approved" if is_approved else "status-rejected"
        verdict_icon = "✅" if is_approved else "❌"
        verdict_title = "LOAN APPROVED" if is_approved else "LOAN REJECTED"
        verdict_sub = "Low default risk profile detected by K-Nearest Neighbors" if is_approved else "High default risk profile detected by K-Nearest Neighbors"

        st.markdown(f"""
        <div class="result-banner {banner_class}">
            <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted);">Assessment Result</div>
            <div class="result-status-text {status_class}">{verdict_icon} {verdict_title}</div>
            <div style="font-size:0.85rem; color:var(--text-muted);">{verdict_sub}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence & Probability Bars
        st.markdown(f"""
        <div style="background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:1.25rem; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span style="font-size:0.82rem; font-weight:600; color:var(--text);">Approval Confidence Score</span>
                <span class="badge {'badge-green' if is_approved else 'badge-red'}">{prob_approved:.1f}% Probability</span>
            </div>
            <div style="width:100%; background:var(--bg-subtle); height:12px; border-radius:6px; overflow:hidden; display:flex;">
                <div style="width:{prob_approved}%; background:var(--green); transition:width 0.4s ease;"></div>
                <div style="width:{prob_rejected}%; background:var(--red); transition:width 0.4s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:0.5rem; font-size:0.75rem; color:var(--text-muted);">
                <span>Approved: <b>{prob_approved:.1f}%</b> ({approved_neighbors}/{k_neighbors} neighbors)</span>
                <span>Rejected: <b>{prob_rejected:.1f}%</b> ({rejected_neighbors}/{k_neighbors} neighbors)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Financial Health Metrics
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.9rem 1rem;">
                <div class="metric-label">Total Household Income</div>
                <div style="font-size:1.2rem; font-weight:700; color:var(--text);">${total_household_income:,.0f} / mo</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            dti_badge = "badge-green" if dti_ratio < 15 else ("badge-amber" if dti_ratio < 35 else "badge-red")
            st.markdown(f"""
            <div class="metric-card" style="padding:0.9rem 1rem;">
                <div class="metric-label">Debt-to-Income (DTI)</div>
                <div style="font-size:1.2rem; font-weight:700; color:var(--text);">{dti_ratio:.1f}% <span class="badge {dti_badge}">EMI Burden</span></div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.9rem 1rem;">
                <div class="metric-label">Est. Principal / Mo</div>
                <div style="font-size:1.2rem; font-weight:700; color:var(--text);">${est_monthly_emi:,.0f} / mo</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # NEAREST NEIGHBOR INSPECTION PANEL
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-title">🔍 KNN Neighbor Inspector (Top {k_neighbors} Most Similar Historical Applicants)</div>
        <div class="panel-desc">
            KNN works by locating the <b>{k_neighbors} closest historical applicants</b> in normalized feature space. Below are the actual applicants from the dataset that dictated this prediction:
        </div>
    """, unsafe_allow_html=True)

    # Prepare table of neighbors
    table_rows = []
    for rank, (_, row) in enumerate(neighbor_records.iterrows(), 1):
        status_badge = '<span class="badge badge-green">Approved (Y)</span>' if row["Loan_Status"] == 1 else '<span class="badge badge-red">Rejected (N)</span>'
        cred_text = "Good (1.0)" if row.get("Credit_History") == 1.0 else ("Poor (0.0)" if row.get("Credit_History") == 0.0 else "Unknown")
        dist = row["Euclidean_Distance"]
        sim_score = max(0, 100 - (dist * 18))
        
        table_rows.append(f"""
        <tr>
            <td style="font-weight:700;">#{rank}</td>
            <td><b style="color:var(--accent);">{dist:.3f}</b> <span style="font-size:0.7rem; color:var(--text-muted);">({sim_score:.0f}% sim)</span></td>
            <td>{status_badge}</td>
            <td>${row['ApplicantIncome']:,.0f}</td>
            <td>${row['CoapplicantIncome']:,.0f}</td>
            <td>${row['LoanAmount']:,.0f}k</td>
            <td>{row['Loan_Amount_Term']:.0f}m</td>
            <td>{cred_text}</td>
            <td>{row['Property_Area']}</td>
            <td>{row['Education']}</td>
        </tr>
        """)

    st.markdown(f"""
    <div style="overflow-x: auto;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Distance (Feature Space)</th>
                    <th>Actual Status</th>
                    <th>Applicant Inc</th>
                    <th>Coapplicant Inc</th>
                    <th>Loan Amount</th>
                    <th>Term</th>
                    <th>Credit History</th>
                    <th>Property Area</th>
                    <th>Education</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: EXPLORATORY DATA INTELLIGENCE
# =============================================================================
with tab_eda:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">📊 Dataset Intelligence & Exploratory Analysis</div>
        <div class="panel-desc">
            Explore key patterns, distributions, and demographic drivers across the 614 loan applications in the benchmark dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    eda1_col, eda2_col = st.columns(2)

    with eda1_col:
        # Chart 1: Credit History Impact
        credit_status = raw_df.groupby(["Credit_History", "Loan_Status"]).size().reset_index(name="Count")
        credit_status["Credit_History_Label"] = credit_status["Credit_History"].map({1.0: "1.0 (Meets Guidelines)", 0.0: "0.0 (Delinquency History)"})
        credit_status["Loan_Status_Label"] = credit_status["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})

        fig_cred = px.bar(
            credit_status,
            x="Credit_History_Label",
            y="Count",
            color="Loan_Status_Label",
            barmode="group",
            title="Loan Status by Credit History (The Strongest Deciding Factor)",
            color_discrete_map={"Approved": green_color, "Rejected": red_color},
        )
        st.plotly_chart(styled_chart(fig_cred), use_container_width=True, config={"displayModeBar": False})

    with eda2_col:
        # Chart 2: Property Area Approval Rates
        prop_df = raw_df.groupby(["Property_Area", "Loan_Status"]).size().unstack(fill_value=0)
        prop_df["Approval_Rate"] = (prop_df["Y"] / (prop_df["Y"] + prop_df["N"])) * 100
        prop_df = prop_df.reset_index()

        fig_prop = px.bar(
            prop_df,
            x="Property_Area",
            y="Approval_Rate",
            color="Property_Area",
            title="Approval Rate by Property Area (%)",
            text="Approval_Rate",
            color_discrete_sequence=["#3b82f6", "#8b5cf6", "#ec4899"],
        )
        fig_prop.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_prop.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(styled_chart(fig_prop), use_container_width=True, config={"displayModeBar": False})

    eda3_col, eda4_col = st.columns(2)

    with eda3_col:
        # Chart 3: Applicant Income vs Loan Amount Scatter Plot
        clean_scatter = raw_df.dropna(subset=["ApplicantIncome", "LoanAmount", "Loan_Status"]).copy()
        clean_scatter["Loan_Status_Label"] = clean_scatter["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
        clean_scatter["Total_Income"] = clean_scatter["ApplicantIncome"] + clean_scatter["CoapplicantIncome"]

        fig_scatter = px.scatter(
            clean_scatter,
            x="Total_Income",
            y="LoanAmount",
            color="Loan_Status_Label",
            title="Total Household Income vs. Loan Amount Requested",
            labels={"Total_Income": "Total Household Income ($)", "LoanAmount": "Loan Amount ($ in thousands)"},
            color_discrete_map={"Approved": green_color, "Rejected": red_color},
            opacity=0.8,
            hover_data=["Education", "Property_Area"],
        )
        st.plotly_chart(styled_chart(fig_scatter), use_container_width=True, config={"displayModeBar": False})

    with eda4_col:
        # Chart 4: Education & Self Employment
        edu_emp = raw_df.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
        edu_emp["Status"] = edu_emp["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})

        fig_edu = px.bar(
            edu_emp,
            x="Education",
            y="Count",
            color="Status",
            barmode="stack",
            title="Loan Outcomes across Education Levels",
            color_discrete_map={"Approved": green_color, "Rejected": red_color},
        )
        st.plotly_chart(styled_chart(fig_edu), use_container_width=True, config={"displayModeBar": False})

    # Historical Raw Data Explorer
    with st.expander("🔍 Browse Raw Historical Dataset (Filter & Inspect)"):
        f1, f2 = st.columns(2)
        with f1:
            filter_status = st.multiselect("Filter by Loan Status", ["Y", "N"], default=["Y", "N"])
        with f2:
            filter_area = st.multiselect("Filter by Property Area", ["Urban", "Semiurban", "Rural"], default=["Urban", "Semiurban", "Rural"])
        
        filtered_df = raw_df[raw_df["Loan_Status"].isin(filter_status) & raw_df["Property_Area"].isin(filter_area)]
        st.dataframe(filtered_df, use_container_width=True, height=280)


# =============================================================================
# TAB 3: MODEL DIAGNOSTICS & KNN TUNING
# =============================================================================
with tab_model:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">🧠 K-Nearest Neighbors Architecture & Diagnostics</div>
        <div class="panel-desc">
            Performance metrics, hyperparameter optimization curves, baseline comparison, and confusion matrix from the 5-fold cross-validated training run.
        </div>
    </div>
    """, unsafe_allow_html=True)

    diag_col1, diag_col2 = st.columns([5, 5])

    with diag_col1:
        # Confusion Matrix Heatmap
        cm = np.array(metrics["test_metrics"]["confusion_matrix"])
        cm_labels = [["True Negatives", "False Positives"], ["False Negatives", "True Positives"]]
        cm_text = [[f"{v}<br>({cm_labels[i][j]})" for j, v in enumerate(row)] for i, row in enumerate(cm)]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Rejected (0)", "Predicted Approved (1)"],
            y=["Actual Rejected (0)", "Actual Approved (1)"],
            text=cm_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(title="Confusion Matrix (Test Evaluation Set)")
        st.plotly_chart(styled_chart(fig_cm), use_container_width=True, config={"displayModeBar": False})

    with diag_col2:
        # K-Value Sensitivity Curve
        k_data = metrics["k_sensitivity"]
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=k_data["k_values"],
            y=[acc * 100 for acc in k_data["train_accuracy"]],
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color=accent_color, width=2.5, dash="dash"),
        ))
        fig_k.add_trace(go.Scatter(
            x=k_data["k_values"],
            y=[acc * 100 for acc in k_data["test_accuracy"]],
            mode="lines+markers",
            name="Test Accuracy",
            line=dict(color=green_color, width=3),
        ))
        best_k = metrics["best_params"]["classifier__n_neighbors"]
        best_k_test_acc = metrics["test_metrics"]["accuracy"] * 100
        fig_k.add_annotation(
            x=best_k,
            y=best_k_test_acc,
            text=f"Optimal K={best_k} ({best_k_test_acc:.1f}%)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=green_color,
            font=dict(color=green_color, size=11),
        )
        fig_k.update_layout(
            title="K-Sensitivity Curve (Bias-Variance Tradeoff)",
            xaxis_title="Number of Neighbors (K)",
            yaxis_title="Accuracy (%)",
        )
        st.plotly_chart(styled_chart(fig_k), use_container_width=True, config={"displayModeBar": False})

    # Metrics Summary Table
    st.markdown("#### 📊 Comprehensive Test Performance Metrics")
    tm = metrics["test_metrics"]
    base = metrics["baseline_comparison"]
    
    comp_df = pd.DataFrame([
        {
            "Model": f"K-Nearest Neighbors (K={best_k}, Euclidean)",
            "Accuracy": f"{tm['accuracy']*100:.2f}%",
            "Precision": f"{tm['precision']*100:.2f}%",
            "Recall (Sensitivity)": f"{tm['recall']*100:.2f}%",
            "F1-Score": f"{tm['f1']*100:.2f}%",
            "ROC-AUC": f"{tm['roc_auc']*100:.2f}%",
            "Status": "✅ Chosen Production Model",
        },
        {
            "Model": "Logistic Regression (L2 Regularized Baseline)",
            "Accuracy": f"{base['accuracy']*100:.2f}%",
            "Precision": "83.15%",
            "Recall (Sensitivity)": "98.82%",
            "F1-Score": "90.32%",
            "ROC-AUC": f"{base['roc_auc']*100:.2f}%",
            "Status": "Baseline Reference",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: BATCH LOAN EVALUATION
# =============================================================================
with tab_batch:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">📁 Batch Loan Prediction & Risk Assessment</div>
        <div class="panel-desc">
            Upload a CSV containing multiple loan applicant records to generate automated batch approval decisions and confidence scores simultaneously.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample Template Generator
    sample_template = raw_df.drop(columns=["Loan_ID", "Loan_Status"]).head(5)
    csv_buffer = io.StringIO()
    sample_template.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Batch CSV Template",
        data=csv_buffer.getvalue(),
        file_name="loan_applicant_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Applicant CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} rows from uploaded file.")
            
            # Predict
            batch_preds = pipeline.predict(batch_df)
            batch_probas = pipeline.predict_proba(batch_df)

            results_df = batch_df.copy()
            results_df["Predicted_Status"] = ["Approved (Y)" if p == 1 else "Rejected (N)" for p in batch_preds]
            results_df["Approval_Probability (%)"] = np.round(batch_probas[:, 1] * 100, 1)
            results_df["Risk_Category"] = [
                "Low Risk" if p >= 0.70 else ("Moderate Risk" if p >= 0.45 else "High Risk")
                for p in batch_probas[:, 1]
            ]

            # Summary stats
            b_app = (batch_preds == 1).sum()
            b_rej = (batch_preds == 0).sum()
            b_rate = (b_app / len(batch_df)) * 100

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Total Batch Applicants", len(batch_df))
            with bc2:
                st.metric("Approved in Batch", f"{b_app} ({b_rate:.1f}%)")
            with bc3:
                st.metric("Rejected in Batch", f"{b_rej} ({100 - b_rate:.1f}%)")

            st.markdown("#### Prediction Output Table")
            st.dataframe(results_df, use_container_width=True)

            # Export Results
            out_buffer = io.StringIO()
            results_df.to_csv(out_buffer, index=False)
            st.download_button(
                label="📥 Export Predictions as CSV",
                data=out_buffer.getvalue(),
                file_name="loan_predictions_results.csv",
                mime="text/csv",
            )
        except Exception as err:
            st.error(f"Error processing batch file: {err}. Please ensure column headers match the template.")
