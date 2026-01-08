import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path to ensure imports from src work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.agent_graph import TradingAgentOrchestrator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Trading Agents Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    h1 { color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- ASYNC STREAM HANDLER ---
async def run_analysis_process(ticker, enable_debate):
    """
    Consumes the async generator from the orchestrator and 
    updates the UI logs and progress bar in real-time.
    """
    orchestrator = TradingAgentOrchestrator()
    
    # Placeholders for streaming feedback
    status_msg = st.empty()
    progress_bar = st.progress(0)
    log_expander = st.expander("Detailed Activity Logs", expanded=True)
    log_content = ""
    
    # Iterate through the asynchronous generator
    async for update in orchestrator.analyze_stock_stream(ticker, enable_debate):
        # 1. Update Progress Bar
        if "step" in update:
            progress_bar.progress(update["step"] / 7)
        
        # 2. Update Status Message and Logs
        if "message" in update:
            status_msg.info(update["message"])
            log_content += f"[{datetime.now().strftime('%H:%M:%S')}] {update['message']}\n"
            log_expander.code(log_content)
            
        # 3. Handle Agent Toasts
        if update.get("status") == "agent_done":
            st.toast(f" {update['agent']} analysis complete!")

        # 4. Handle Final Result
        if update.get("status") == "complete":
            st.session_state.analysis_results = update["final_decision"]
            status_msg.success("Analysis Complete! Loading dashboard...")
            st.rerun() 

# --- CHART HELPERS ---
def create_price_chart(price_data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_data.index, open=price_data['Open'],
        high=price_data['High'], low=price_data['Low'],
        close=price_data['Close'], name='Price'
    ))
    fig.update_layout(title='Market Price Action', template='plotly_white', xaxis_rangeslider_visible=False, height=500)
    return fig

def create_consensus_chart(agent_responses):
    agents = [r.agent_name for r in agent_responses[:-1]]
    confidences = [r.confidence * 100 for r in agent_responses[:-1]]
    recommendations = [r.recommendation for r in agent_responses[:-1]]
    
    color_map = {'STRONG BUY': '#00CC00', 'BUY': '#66FF66', 'HOLD': '#FFAA00', 'SELL': '#FF6666', 'STRONG SELL': '#CC0000'}
    colors = [color_map.get(rec, '#CCCCCC') for rec in recommendations]
    
    fig = go.Figure(go.Bar(
        x=agents, y=confidences,
        text=[f"{rec}" for rec in recommendations], textposition='auto',
        marker_color=colors
    ))
    fig.update_layout(title="Agent Confidence & Recommendations", yaxis_title="Confidence %", template='plotly_white', height=400)
    return fig

# --- MAIN UI ---
def main():
    st.title("🤖 AI Multi-Agent Trading Platform")
    st.markdown("##### *Powered by Multi-Agent LLM Orchestration*")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        enable_debate = st.checkbox("Enable Agent Debate", value=True)
        
        st.divider()
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not ticker:
                st.error("Please enter a ticker symbol.")
            else:
                st.session_state.analysis_results = None # Clear old results
                asyncio.run(run_analysis_process(ticker, enable_debate))

    # --- DASHBOARD DISPLAY ---
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        
        # Metrics Row
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ticker", res.ticker)
        m2.metric("Final Decision", res.final_recommendation)
        m3.metric("Confidence", f"{res.confidence:.0%}")
        m4.metric("Current Price", f"${res.market_data['statistics']['current_price']:.2f}")

        # Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Agent Reasoning", "📝 Raw Data"])
        
        with tab1:
            st.plotly_chart(create_price_chart(res.market_data['price_data']), use_container_width=True)
            st.plotly_chart(create_consensus_chart(res.agent_responses), use_container_width=True)
            
        with tab2:
            st.subheader("Head Trader Synthesis")
            st.info(res.trader_analysis)
            st.divider()
            for resp in res.agent_responses:
                badge = "🟢" if "BUY" in resp.recommendation else "🔴" if "SELL" in resp.recommendation else "🟡"
                with st.expander(f"{badge} **{resp.agent_name}** ({resp.confidence:.0%})"):
                    st.write(resp.analysis)
        
        with tab3:
            st.json(res.to_dict())

if __name__ == "__main__":
    main()