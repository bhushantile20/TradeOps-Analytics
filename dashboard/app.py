import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import yaml
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="TradeOps Analytics – BTC Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Premium Dark CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0B1220;
        color: #F1F5F9;
    }
    .stApp {
        background-color: #0B1220;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] * {
        color: #94A3B8;
    }
    .stMetric {
        background-color: rgba(255,255,255,0.04);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 16px !important;
    }
    .stMetric label { color: #64748B !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.1em; }
    .stMetric [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 26px !important; font-weight: 700; }
    h1 { color: #F1F5F9 !important; font-size: 26px !important; font-weight: 800 !important; }
    h2 { color: #CBD5E1 !important; font-size: 18px !important; font-weight: 700 !important; }
    h3 { color: #94A3B8 !important; font-size: 14px !important; font-weight: 600 !important; }
    div[data-testid="stSelectbox"] label { color: #64748B !important; font-size: 12px !important; }
    .dataframe { background: rgba(255,255,255,0.03) !important; }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 13px;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 0 15px rgba(59,130,246,0.3);
    }
    hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# --- Load Project Config ---
def load_config():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        st.error("❌ `config.yaml` not found. Please ensure you run this app from the project root directory.")
        st.stop()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# --- Helper Functions ---
@st.cache_data(ttl=120)  # Cache for 2 minutes
def load_data():
    path = config['data']['processed_path']
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df.index)
    return df


def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


PLOTLY_DARK_LAYOUT = dict(
    plot_bgcolor='rgba(11, 18, 32, 0)',
    paper_bgcolor='rgba(11, 18, 32, 0)',
    font=dict(color='#94A3B8', family='Inter, sans-serif', size=12),
    xaxis=dict(showgrid=False, zeroline=False, color='#475569'),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        color='#475569'
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor='rgba(15,23,42,0.8)',
        bordercolor='rgba(255,255,255,0.1)',
        borderwidth=1,
        font=dict(color='#94A3B8', size=11)
    ),
    hovermode='x unified'
)


# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0;'>
        <div style='display:flex; align-items:center; gap: 10px; margin-bottom:6px;'>
            <svg width="22" height="22" fill="#3B82F6" viewBox="0 0 20 20"><path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/></svg>
            <span style='font-size:18px; font-weight:800; color:#F1F5F9;'>TradeOps Analytics</span>
        </div>
        <div style='font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.12em; padding-left:4px;'>● LIVE SYSTEM</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.06); margin:0 0 12px 0;'>
    """, unsafe_allow_html=True)

    selection = st.radio(
        "Go to",
        ["📊  Market Overview", "📈  Technical Analysis", "🤖  Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin: 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; color:#334155; padding: 8px 0;'>Model Version: v2.1.0-prod</div>", unsafe_allow_html=True)


# --- Load Data ---
df = load_data()
if df.empty:
    st.error("⚠️ Processed data not found. Check `config.yaml` path for `data.processed_path`.")
    st.stop()


# ==========================================
# PAGE 1: MARKET OVERVIEW
# ==========================================
if "Market Overview" in selection:
    st.markdown("## BTC/USD Live Market Overview")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    try:
        price_24h_ago = df['close'].shift(24).iloc[-1]
        pct_change = ((df['close'].iloc[-1] / price_24h_ago) - 1) * 100
    except Exception:
        pct_change = 0.0

    col1.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}")
    col2.metric("24H Change", f"{pct_change:+.2f}%", delta_color="normal")
    col3.metric("RSI (1H)", f"{df['RSI'].iloc[-1]:.1f}" if 'RSI' in df.columns else "N/A")
    col4.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}" if 'volume' in df.columns else "N/A")

    st.markdown("---")

    # Main Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        name='BTC/USD',
        line={"color": '#3B82F6', "width": 2},
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.07)'
    ))
    fig.update_layout(
        title="BTC/USD Closing Price Historical",
        height=500,
        **{k: v for k, v in PLOTLY_DARK_LAYOUT.items()}
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 2: TECHNICAL ANALYSIS
# ==========================================
elif "Technical Analysis" in selection:
    st.markdown("## Technical Indicator Analysis")
    st.markdown("---")

    indicator = st.selectbox("Select Indicator", ["RSI & Moving Averages", "MACD Oscillator"])

    if indicator == "RSI & Moving Averages":
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price & Moving Averages", "RSI Oscillator")
        )

        # Price and MAs
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Price', line={"color": '#F1F5F9', "width": 1.5}), row=1, col=1)
        if 'moving_average_10' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['moving_average_10'], name='SMA 10', line={"color": '#3B82F6', "width": 1.5}), row=1, col=1)
        if 'moving_average_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['moving_average_20'], name='SMA 20', line={"color": '#8B5CF6', "width": 1.5}), row=1, col=1)

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI', line={"color": '#F59E0B', "width": 1.2}), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.5)", row=2, col=1)

        fig.update_layout(height=700, **{k: v for k, v in PLOTLY_DARK_LAYOUT.items()})
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    else:  # MACD
        macd, signal = calculate_macd(df['close'])
        histogram = macd - signal

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=histogram,
            name='Histogram',
            marker_color=['rgba(16,185,129,0.6)' if v >= 0 else 'rgba(239,68,68,0.6)' for v in histogram]
        ))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=macd, name='MACD', line={"color": '#3B82F6', "width": 1.5}))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=signal, name='Signal', line={"color": '#F59E0B', "width": 1.5}))
        fig.update_layout(title="MACD Oscillator", height=500, **{k: v for k, v in PLOTLY_DARK_LAYOUT.items()})
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 3: MODEL PERFORMANCE & COMPARISON
# ==========================================
elif "Model Performance" in selection:
    st.markdown("## AI Model Forecast Visualization")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox(
            "Active Forecast Engine",
            ["ARIMA", "Linear Regression", "LSTM", "RNN"]
        )

    # Train/test split
    split_idx = int(len(df) * config['models']['train_test_split'])
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    fig_pred = go.Figure()

    # Historical line
    fig_pred.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        name='Historical Price',
        mode='lines',
        line={"color": '#3B82F6', "width": 2.5, "shape": 'spline'},
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.08)'
    ))

    # Load model and generate forecast
    forecast_dates = test_df['timestamp']
    forecast_prices = None

    if model_choice == "ARIMA":
        arima_path = "models/arima/arima_model.pkl"
        if os.path.exists(arima_path):
            with st.spinner("Loading ARIMA model..."):
                try:
                    model_fit = joblib.load(arima_path)
                    forecast_prices = model_fit.forecast(steps=len(test_df))
                    st.success(f"✅ ARIMA model loaded from `{arima_path}`")
                except Exception as e:
                    st.error(f"ARIMA load error: {e}")
        else:
            st.warning(f"⚠️ ARIMA model file not found at `{arima_path}`. Run the training pipeline first.")

    elif model_choice == "Linear Regression":
        lr_path = "models/linear_regression/lr_model.pkl"
        if os.path.exists(lr_path):
            with st.spinner("Loading Linear Regression model..."):
                try:
                    lr_model = joblib.load(lr_path)
                    lr_features = config['models']['linear_regression']['features']
                    available_features = [f for f in lr_features if f in test_df.columns]
                    if len(available_features) < len(lr_features):
                        missing = set(lr_features) - set(available_features)
                        st.warning(f"⚠️ Missing features: {missing}. Using available features only.")
                    returns_pred = lr_model.predict(test_df[available_features])
                    last_train_price = df['close'].iloc[split_idx - 1]
                    forecast_prices = last_train_price * np.exp(np.cumsum(returns_pred))
                    st.success(f"✅ Linear Regression model loaded from `{lr_path}`")
                except Exception as e:
                    st.error(f"LR model error: {e}")
        else:
            st.warning(f"⚠️ LR model not found at `{lr_path}`. Run training pipeline first.")

    elif model_choice in ["LSTM", "RNN"]:
        st.info(f"""
        **{model_choice}** deep learning model is disabled in the Streamlit environment due to TensorFlow incompatibility with Python 3.14+.
        
        ➡️ View {model_choice} results in the Django Dashboard: `http://127.0.0.1:8005/dashboard/`
        """)

    # Plot forecast if available
    if forecast_prices is not None:
        connect_x = list(train_df['timestamp'].iloc[-1:]) + list(forecast_dates)
        connect_y = list(train_df['close'].iloc[-1:]) + list(forecast_prices)

        fig_pred.add_trace(go.Scatter(
            x=connect_x,
            y=connect_y,
            name=f'{model_choice} Forecast',
            mode='lines',
            line={"color": '#8B5CF6', "width": 2.5, "dash": 'dash', "shape": 'spline'},
            fill='tozeroy',
            fillcolor='rgba(139,92,246,0.07)'
        ))

        split_date = train_df['timestamp'].iloc[-1]
        fig_pred.add_vline(
            x=split_date,
            line_width=1,
            line_dash="dash",
            line_color="rgba(156,163,175,0.5)",
            annotation_text="Prediction Gateway",
            annotation_position="top right",
            annotation_font=dict(color="#9CA3AF", size=11)
        )

        # Show error metrics
        actual_test = test_df['close'].values[:len(forecast_prices)]
        pred_arr = np.array(forecast_prices)[:len(actual_test)]
        if len(actual_test) > 0 and len(pred_arr) > 0:
            rmse = np.sqrt(np.mean((actual_test - pred_arr) ** 2))
            mae = np.mean(np.abs(actual_test - pred_arr))
            mape = np.mean(np.abs((actual_test - pred_arr) / actual_test)) * 100
            st.markdown("---")
            st.markdown(f"**{model_choice} Forecast Results on Test Set**")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"${rmse:,.2f}")
            m2.metric("MAE", f"${mae:,.2f}")
            m3.metric("MAPE", f"{mape:.2f}%")

    # Layout
    fig_pred.update_layout(
        title=f"BTC/USD — {model_choice} Forecast vs Historical",
        height=580,
        **{k: v for k, v in PLOTLY_DARK_LAYOUT.items()}
    )
    st.plotly_chart(fig_pred, use_container_width=True)
