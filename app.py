"""
============================================================
  Tesla Stock Forecasting — Streamlit Deployment
  Models: GRU · Stacked LSTM · Bidirectional LSTM
  Run:    streamlit run app.py
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import io
import os
import pickle

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla LSTM Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark header bar */
[data-testid="stHeader"] { background: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c9d8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #7aa2d4 !important; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #58a6ff; font-family: 'Space Mono', monospace; }
[data-testid="stMetricLabel"] { color: #7aa2d4 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace; }

/* Main area */
.main .block-container { background: #070b14; padding-top: 1.5rem; }

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d6b9e;
    margin: 2rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2d4a;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    font-family: 'Space Mono', monospace;
}
.badge-green { background: #0d2818; color: #56d364; border: 1px solid #1a4d2e; }
.badge-blue  { background: #0d1f35; color: #58a6ff; border: 1px solid #1a3a5c; }
.badge-amber { background: #2d1f0a; color: #e3b341; border: 1px solid #4d3510; }

/* Hero title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6f0ff;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.85rem;
    color: #4d7ab5;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Stacked info rows */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #0f1e30;
    font-size: 0.82rem;
}
.info-row .key { color: #4d7ab5; }
.info-row .val { color: #c9d8f0; font-family: 'Space Mono', monospace; }

/* Plotly chart container */
.js-plotly-plot { border-radius: 10px; }

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6c, #0d2040);
    color: #58a6ff;
    border: 1px solid #1e3a6e;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e4480, #112550);
    border-color: #2a4e8c;
    color: #79bbff;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0a1020;
    border: 1px dashed #1e3a6e;
    border-radius: 10px;
    padding: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0a0e1a; border-bottom: 1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"] { color: #4d7ab5; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* Progress bar */
.stProgress > div > div { background: #58a6ff; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e2d4a; border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader { background: #0a1020; border: 1px solid #1e2d4a; border-radius: 8px; color: #7aa2d4; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

PLOTLY_THEME = dict(
    paper_bgcolor="#070b14",
    plot_bgcolor="#0a0e1a",
    font=dict(family="Space Mono, monospace", color="#7aa2d4", size=11),
    xaxis=dict(gridcolor="#0f1e30", linecolor="#1e2d4a", tickcolor="#1e2d4a"),
    yaxis=dict(gridcolor="#0f1e30", linecolor="#1e2d4a", tickcolor="#1e2d4a"),
    margin=dict(l=40, r=20, t=40, b=40),
)

MODEL_COLORS = {
    "GRU":                "#56d364",
    "Stacked LSTM":       "#58a6ff",
    "Bidirectional LSTM": "#f78166",
}

# ── CORE PIPELINE FUNCTIONS ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.columns = [c.strip().title() for c in df.columns]
    expected = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected if c in df.columns]]
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@st.cache_data(show_spinner=False)
def add_features(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    df["MA10"]     = df["Close"].rolling(10).mean()
    df["MA50"]     = df["Close"].rolling(50).mean()
    df["EMA20"]    = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI14"]    = compute_rsi(df["Close"])
    for lag in [1, 2, 3, 5, 10]:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)
    df["Return_1d"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df


def preprocess(df: pd.DataFrame):
    feature_cols = list(df.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[feature_cols])
    target_idx = feature_cols.index("Close")
    return scaled, scaler, feature_cols, target_idx


def create_sequences(data: np.ndarray, target_idx: int, time_steps: int = 60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps: i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def split_data(X, y, split=0.80):
    cut = int(len(X) * split)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    val_cut = int(len(X_train) * 0.90)
    return (X_train[:val_cut], X_train[val_cut:], X_test,
            y_train[:val_cut], y_train[val_cut:], y_test)


def build_model(model_type: str, input_shape: tuple):
    model = Sequential(name=model_type.replace(" ", "_"))
    model.add(Input(shape=input_shape))
    if model_type == "GRU":
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(64))
    elif model_type == "Stacked LSTM":
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def inverse_close(preds, scaler, n_features, target_idx):
    dummy = np.zeros((len(preds), n_features))
    dummy[:, target_idx] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, target_idx]


# ── PLOTLY CHART BUILDERS ────────────────────────────────────────────────────

def chart_price_history(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"
    ), row=1, col=1)
    if "MA10" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA10"], name="MA10",
            line=dict(color="#e3b341", width=1, dash="dot")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA50"], name="MA50",
            line=dict(color="#f78166", width=1, dash="dot")
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color="rgba(88,166,255,0.25)"
    ), row=2, col=1)
    fig.update_layout(**PLOTLY_THEME, height=480,
                      legend=dict(orientation="h", y=1.02, x=0),
                      title_text="Price History with Moving Averages",
                      title_font=dict(color="#c9d8f0", size=13))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def chart_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI14"], name="RSI 14",
        line=dict(color="#56d364", width=1.5)
    ))
    fig.add_hline(y=70, line=dict(color="#f78166", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#58a6ff", dash="dash", width=1))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(247,129,102,0.05)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(88,166,255,0.05)",  line_width=0)
    fig.update_layout(**PLOTLY_THEME, height=280,
                      title_text="RSI (14-day)",
                      title_font=dict(color="#c9d8f0", size=13))
    return fig


def chart_correlation(df: pd.DataFrame) -> go.Figure:
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#7aa2d4")),
    ))
    fig.update_layout(**PLOTLY_THEME, height=320,
                      title_text="Correlation Heatmap",
                      title_font=dict(color="#c9d8f0", size=13))
    return fig


def chart_predictions(y_true, results: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_true, name="Actual",
        line=dict(color="#ffffff", width=2), opacity=0.9
    ))
    for name, y_pred in results.items():
        fig.add_trace(go.Scatter(
            y=y_pred, name=name,
            line=dict(color=MODEL_COLORS.get(name, "#aaa"), width=1.5, dash="dot"),
            opacity=0.85
        ))
    fig.update_layout(**PLOTLY_THEME, height=420,
                      title_text="Actual vs Predicted — Close Price (Test Set)",
                      title_font=dict(color="#c9d8f0", size=13),
                      legend=dict(orientation="h", y=1.02),
                      xaxis_title="Test samples (chronological)",
                      yaxis_title="Price (USD)")
    return fig


def chart_loss(histories: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(histories),
                        subplot_titles=list(histories.keys()))
    for i, (name, h) in enumerate(histories.items(), start=1):
        c = MODEL_COLORS.get(name, "#aaa")
        fig.add_trace(go.Scatter(
            y=h["loss"], name="Train",
            line=dict(color=c, width=1.5),
            showlegend=(i == 1)
        ), row=1, col=i)
        fig.add_trace(go.Scatter(
            y=h["val_loss"], name="Validation",
            line=dict(color=c, width=1.5, dash="dash"),
            showlegend=(i == 1)
        ), row=1, col=i)
    fig.update_layout(**PLOTLY_THEME, height=320,
                      title_text="Training vs Validation Loss",
                      title_font=dict(color="#c9d8f0", size=13),
                      legend=dict(orientation="h", y=1.1))
    return fig


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem 0 0.5rem;'>
          <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #58a6ff; font-weight: 700;'>TSLA LSTM</div>
          <div style='font-size: 0.7rem; color: #3d6b9e; letter-spacing: 0.1em;'>FORECASTING SUITE v1.0</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Tesla stock CSV with Date, Open, High, Low, Close, Volume columns"
        )

        st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)
        selected_models = st.multiselect(
            "Models to train",
            ["GRU", "Stacked LSTM", "Bidirectional LSTM"],
            default=["GRU", "Stacked LSTM"],
        )

        st.markdown('<div class="section-title">Hyperparameters</div>', unsafe_allow_html=True)
        time_steps = st.slider("Time steps (window)", 20, 120, 60, 5)
        epochs     = st.slider("Max epochs", 10, 100, 50, 5)
        batch_size = st.slider("Batch size", 16, 128, 32, 16)
        split      = st.slider("Train split %", 60, 90, 80, 5) / 100

        st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 0.75rem; color: #3d6b9e; line-height: 1.7;'>
        End-to-end LSTM pipeline.<br>
        Features: MA10, MA50, EMA20, RSI14, Lag(1-10), Return_1d.<br>
        Scaler: MinMaxScaler [0,1].<br>
        Optimizer: Adam · Loss: MSE
        </div>
        """, unsafe_allow_html=True)

    return uploaded, selected_models, time_steps, epochs, batch_size, split


# ── MAIN APP ─────────────────────────────────────────────────────────────────

def main():
    uploaded, selected_models, time_steps, epochs, batch_size, split = render_sidebar()

    # ── Hero header ──────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown("""
        <div class="hero-title">Tesla Stock Forecasting</div>
        <div class="hero-sub">LSTM · GRU · Bidirectional LSTM &nbsp;|&nbsp; Time-Series Deep Learning</div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown("""
        <div style='text-align:right; padding-top:0.5rem;'>
          <span class="badge badge-blue">TensorFlow 2.x</span><br><br>
          <span class="badge badge-green">Streamlit</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── No file state ────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; color: #1e3a6e;'>
          <div style='font-size: 3rem; margin-bottom: 1rem;'>📂</div>
          <div style='font-family: Space Mono, monospace; font-size: 0.9rem; color: #2a4e8c;'>
            Upload your Tesla stock CSV in the sidebar to begin
          </div>
          <div style='font-size: 0.75rem; color: #1a3050; margin-top: 0.5rem;'>
            Expected columns: Date · Open · High · Low · Close · Volume
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        df_raw  = load_and_clean(uploaded.read())
        df_feat = add_features(df_raw)

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total rows",    f"{len(df_raw):,}")
    m2.metric("Date range",    f"{df_raw.index.min().year}–{df_raw.index.max().year}")
    m3.metric("Latest close",  f"${df_raw['Close'].iloc[-1]:.2f}")
    m4.metric("All-time high", f"${df_raw['Close'].max():.2f}")
    pct = (df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[0]) / df_raw['Close'].iloc[0] * 100
    m5.metric("Total return",  f"{pct:.0f}%", delta=f"{pct:.0f}%")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_eda, tab_features, tab_train, tab_results = st.tabs([
        "📊  EDA", "🔧  Features", "🚀  Train Models", "📈  Results"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — EDA
    # ════════════════════════════════════════════════════════════════════════
    with tab_eda:
        st.markdown('<div class="section-title">Price History</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_price_history(df_feat), use_container_width=True)

        col_rsi, col_corr = st.columns([1, 1])
        with col_rsi:
            st.markdown('<div class="section-title">Momentum — RSI</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rsi(df_feat), use_container_width=True)
        with col_corr:
            st.markdown('<div class="section-title">Correlation</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_correlation(df_raw), use_container_width=True)

        with st.expander("Raw data sample"):
            st.dataframe(df_raw.tail(20).style.format("{:.2f}"),
                         use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — FEATURES
    # ════════════════════════════════════════════════════════════════════════
    with tab_features:
        st.markdown('<div class="section-title">Engineered Features</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_ma = go.Figure()
            for col, c in [("Close","#ffffff"), ("MA10","#e3b341"), ("MA50","#f78166"), ("EMA20","#56d364")]:
                fig_ma.add_trace(go.Scatter(
                    x=df_feat.index[-500:], y=df_feat[col].iloc[-500:],
                    name=col, line=dict(color=c, width=1.5 if col=="Close" else 1.2,
                                       dash="solid" if col=="Close" else "dot")
                ))
            fig_ma.update_layout(**PLOTLY_THEME, height=320,
                                 title_text="Moving Averages (last 500 days)",
                                 title_font=dict(color="#c9d8f0", size=13))
            st.plotly_chart(fig_ma, use_container_width=True)

        with col_b:
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=df_feat["Return_1d"] * 100,
                nbinsx=80,
                marker_color="#58a6ff",
                marker_line_color="#0a0e1a",
                marker_line_width=0.5,
                name="Daily return %"
            ))
            fig_ret.update_layout(**PLOTLY_THEME, height=320,
                                  title_text="Daily Return Distribution",
                                  title_font=dict(color="#c9d8f0", size=13),
                                  xaxis_title="Return (%)", yaxis_title="Count")
            st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown('<div class="section-title">Feature Table (last 10 rows)</div>', unsafe_allow_html=True)
        display_cols = ["Close", "MA10", "MA50", "EMA20", "RSI14", "Return_1d", "Lag_1", "Lag_5"]
        st.dataframe(
            df_feat[display_cols].tail(10).style.format("{:.4f}").background_gradient(
                cmap="Blues", subset=["Close"]
            ),
            use_container_width=True,
        )

        st.markdown('<div class="section-title">Preprocessing Summary</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Features", len(df_feat.columns))
        c2.metric("Rows after dropna", len(df_feat))
        c3.metric("Time steps (window)", time_steps)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRAIN
    # ════════════════════════════════════════════════════════════════════════
    with tab_train:
        st.markdown('<div class="section-title">Training Configuration</div>', unsafe_allow_html=True)

        cfg_cols = st.columns(4)
        cfg_cols[0].metric("Selected models",  len(selected_models))
        cfg_cols[1].metric("Window (steps)",   time_steps)
        cfg_cols[2].metric("Max epochs",       epochs)
        cfg_cols[3].metric("Train / Test",     f"{int(split*100)} / {100 - int(split*100)}")

        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
            return

        st.markdown("**Models queued:**")
        for m in selected_models:
            c = MODEL_COLORS.get(m, "#aaa")
            st.markdown(f'<span class="badge" style="background:#0a1020;color:{c};border:1px solid {c}30;">{m}</span>&nbsp;',
                        unsafe_allow_html=True)

        st.markdown("")
        run_btn = st.button("▶  Train Selected Models", type="primary")

        if run_btn or st.session_state.get("trained"):
            if run_btn:
                # ── Preprocess ──────────────────────────────────────────
                scaled, scaler, feature_cols, target_idx = preprocess(df_feat)
                n_features = len(feature_cols)
                X, y = create_sequences(scaled, target_idx, time_steps)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, split)
                input_shape = (time_steps, n_features)

                results     = {}
                histories   = {}
                predictions = {}

                overall_bar = st.progress(0, text="Starting training...")

                for idx, model_name in enumerate(selected_models):
                    st.markdown(f'<div class="section-title">Training — {model_name}</div>',
                                unsafe_allow_html=True)
                    epoch_placeholder = st.empty()
                    loss_placeholder  = st.empty()

                    model = build_model(model_name, input_shape)

                    # Streamlit-friendly training via EarlyStopping callback
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            p = (idx / len(selected_models)) + \
                                ((epoch + 1) / epochs) / len(selected_models)
                            overall_bar.progress(
                                min(p, 0.99),
                                text=f"Training {model_name} — epoch {epoch+1}"
                            )
                            epoch_placeholder.markdown(
                                f"`Epoch {epoch+1:3d}` &nbsp; "
                                f"loss: `{logs.get('loss', 0):.5f}` &nbsp; "
                                f"val_loss: `{logs.get('val_loss', 0):.5f}`"
                            )

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True),
                            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-6),
                            StreamlitCallback(),
                        ],
                        verbose=0,
                    )

                    y_pred_s  = model.predict(X_test, verbose=0)
                    y_pred    = inverse_close(y_pred_s, scaler, n_features, target_idx)
                    y_true    = inverse_close(y_test,   scaler, n_features, target_idx)
                    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mae       = float(mean_absolute_error(y_true, y_pred))

                    results[model_name]     = {"RMSE": rmse, "MAE": mae}
                    histories[model_name]   = {"loss": history.history["loss"],
                                               "val_loss": history.history["val_loss"]}
                    predictions[model_name] = y_pred
                    epoch_placeholder.empty()

                overall_bar.progress(1.0, text="Training complete ✓")

                # Save to session state
                st.session_state["trained"]     = True
                st.session_state["y_true"]      = y_true
                st.session_state["results"]     = results
                st.session_state["histories"]   = histories
                st.session_state["predictions"] = predictions

            st.success("All models trained. View results in the **Results** tab →")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — RESULTS
    # ════════════════════════════════════════════════════════════════════════
    with tab_results:
        if not st.session_state.get("trained"):
            st.info("Train models first in the **Train Models** tab.")
            return

        y_true      = st.session_state["y_true"]
        results     = st.session_state["results"]
        histories   = st.session_state["histories"]
        predictions = st.session_state["predictions"]

        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

        # Metric cards per model
        res_cols = st.columns(len(results))
        best_rmse_model = min(results, key=lambda k: results[k]["RMSE"])
        for col, (name, metrics) in zip(res_cols, results.items()):
            delta_str = "⭐ Best" if name == best_rmse_model else None
            col.metric(
                label=f"{name} — RMSE",
                value=f"${metrics['RMSE']:.2f}",
                delta=delta_str,
            )
            col.metric(
                label=f"{name} — MAE",
                value=f"${metrics['MAE']:.2f}",
            )

        st.divider()

        # Results table
        df_res = pd.DataFrame(results).T.round(4).sort_values("RMSE")
        df_res.index.name = "Model"
        st.markdown('<div class="section-title">Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            df_res.style.highlight_min(axis=0, color="#0d2818").format("{:.4f}"),
            use_container_width=True,
        )

        # Predictions chart
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_predictions(y_true, predictions), use_container_width=True)

        # Loss curves
        st.markdown('<div class="section-title">Loss Curves</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_loss(histories), use_container_width=True)

        # Download results
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        export_df = pd.DataFrame({"Actual": y_true})
        for name, pred in predictions.items():
            export_df[name] = pred

        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇  Download predictions CSV",
            data=csv_bytes,
            file_name="tesla_predictions.csv",
            mime="text/csv",
        )


# ── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "trained" not in st.session_state:
        st.session_state["trained"] = False
    main()
