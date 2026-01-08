import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.agent_graph import TradingAgentOrchestrator
from src.data.market_data import MarketDataFetcher

# Page configuration
st.set_page_config(
    page_title="AI Trading Agents Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = TradingAgentOrchestrator()

def create_price_chart(price_data):
    """Create interactive price chart with indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'SMA_20' in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1.5)
        ))
    
    if 'SMA_50' in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1.5)
        ))
    
    fig.update_layout(
        title='Price Chart with Technical Indicators',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_agent_consensus_chart(agent_responses):
    """Create visualization of agent recommendations"""
    recommendations = []
    confidences = []
    agents = []
    
    for response in agent_responses[:-1]:  # Exclude final trader decision
        agents.append(response.agent_name)
        recommendations.append(response.recommendation)
        confidences.append(response.confidence * 100)
    
    # Create color mapping
    color_map = {
        'STRONG BUY': '#00CC00',
        'BUY': '#66FF66',
        'HOLD': '#FFAA00',
        'SELL': '#FF6666',
        'STRONG SELL': '#CC0000'
    }
    
    colors = [color_map.get(rec, '#CCCCCC') for rec in recommendations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=agents,
            y=confidences,
            marker_color=colors,
            text=[f"{rec}<br>{conf:.0f}%" for rec, conf in zip(recommendations, confidences)],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Agent Recommendations & Confidence Levels',
        yaxis_title='Confidence (%)',
        xaxis_title='Agent',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_technical_indicators_chart(price_data):
    """Create chart for RSI and other indicators"""
    fig = go.Figure()
    
    if 'RSI' in price_data.columns:
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        yaxis_title='RSI Value',
        xaxis_title='Date',
        height=300,
        template='plotly_white',
        yaxis_range=[0, 100]
    )
    
    return fig

def display_agent_analysis(response):
    """Display individual agent analysis"""
    # Recommendation badge
    rec_colors = {
        'STRONG BUY': '🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG SELL': '🔴'
    }
    
    badge = rec_colors.get(response.recommendation, '⚪')
    
    with st.expander(f"{badge} **{response.agent_name}** - {response.recommendation} ({response.confidence:.0%} confidence)", expanded=False):
        st.markdown("##### Key Points:")
        for i, reason in enumerate(response.reasoning, 1):
            st.markdown(f"{i}. {reason}")
        
        st.markdown("---")
        st.markdown("##### Full Analysis:")
        st.write(response.analysis)

# Main App
def main():
    st.title("🤖 AI Multi-Agent Trading Platform")
    st.markdown("##### *Powered by GPT-4 | Built with LangGraph-inspired Architecture*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, TSLA, NVDA)"
        ).upper()
        
        enable_debate = st.checkbox(
            "Enable Agent Debate",
            value=True,
            help="Allow agents to debate conflicting views"
        )
        
        st.markdown("---")
        
        analyze_button = st.button(" Run Analysis", type="primary", width='stretch')
        
        st.markdown("---")
        st.markdown("#### About")
        st.info("""
        This platform uses multiple specialized AI agents:
        
        -  **Technical Analyst**: Chart patterns & indicators
        -  **Fundamental Analyst**: Financial metrics
        -  **Sentiment Analyst**: News & market psychology
        -  **Risk Manager**: Position sizing & risk assessment
        -  **Head Trader**: Final decision synthesis
        """)
    
    # Main content
    if analyze_button:
        if not ticker:
            st.error("Please enter a stock ticker")
            return
        
        with st.spinner(f" AI agents analyzing {ticker}..."):
            try:
                # Run analysis
                orchestrator = st.session_state.orchestrator
                decision = orchestrator.analyze_stock(ticker, enable_debate=enable_debate)
                st.session_state.analysis_results = decision
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
    
    # Display results
    if st.session_state.analysis_results:
        decision = st.session_state.analysis_results
        
        # Header metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${decision.market_data['statistics']['current_price']:.2f}",
                f"{decision.market_data['statistics']['price_change_pct']:.2f}%"
            )
        
        
        with col2:
            rec_color_text = {
                'STRONG BUY': '🟢 STRONG BUY',
                'BUY': '🟢 BUY',
                'HOLD': '🟡 HOLD',
                'SELL': '🔴 SELL',
                'STRONG SELL': '🔴 STRONG SELL'
            }
            st.metric(
                "Final Decision",
                rec_color_text.get(decision.final_recommendation, decision.final_recommendation),
                f"{decision.confidence:.0%} confidence"
            )
        
        with col3:
            st.metric(
                "30-Day Return",
                f"{decision.market_data['statistics']['30d_return']:.2f}%"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{decision.market_data['statistics']['volatility']:.1f}%",
                "Annualized"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Agent Analysis", " Charts", " Full Report"])
        
        with tab1:
            st.subheader("Executive Summary")
            
            # Final trader decision
            st.markdown("###  Head Trader Decision")
            final_response = decision.agent_responses[-1]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("##### Key Reasoning:")
                for i, reason in enumerate(final_response.reasoning, 1):
                    st.markdown(f"**{i}.** {reason}")
            
            with col2:
                st.markdown("##### Consensus Metrics")
                
                # Safely get quantitative consensus
                qc = getattr(decision, 'quantitative_consensus', {})
                
                # 1. Handle Agreement Level Safely
                agreement = qc.get('agreement_level')
                if agreement is not None:
                    try:
                        agreement_text = f"{float(agreement):.0%}"
                    except (ValueError, TypeError):
                        agreement_text = "N/A"
                else:
                    agreement_text = "N/A"
                
                # 2. Handle Consensus Score Safely
                consensus_score = qc.get('consensus_score')
                if consensus_score is not None:
                    try:
                        consensus_score_text = f"{float(consensus_score):.2f}"
                    except (ValueError, TypeError):
                        consensus_score_text = str(consensus_score)
                else:
                    consensus_score_text = "N/A"
                
                st.metric("Agent Agreement", agreement_text)
                st.metric("Consensus Score", consensus_score_text)
            
            # Agent consensus visualization
            
            st.markdown("---")
            st.plotly_chart(
                create_agent_consensus_chart(decision.agent_responses),
                width='stretch'
            )
        
        with tab2:
            st.subheader("Individual Agent Analysis")
            
            for response in decision.agent_responses:
                display_agent_analysis(response)
        
        with tab3:
            st.subheader("Technical Charts")
            
            price_data = decision.market_data['price_data']
            
            # Price chart
            st.plotly_chart(
                create_price_chart(price_data),
                width='stretch'
            )
            
            # RSI chart
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_technical_indicators_chart(price_data),
                    width='stretch'
                )
            
            with col2:
                # Volume chart
                fig = px.bar(
                    price_data,
                    x=price_data.index,
                    y='Volume',
                    title='Trading Volume'
                )
                fig.update_layout(height=300, template='plotly_white')
                st.plotly_chart(fig, width='stretch')
        
        with tab4:
            st.subheader("Complete Analysis Report")
            
            st.markdown(f"**Ticker:** {decision.ticker}")
            st.markdown(f"**Analysis Date:** {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Final Recommendation:** {decision.final_recommendation}")
            st.markdown(f"**Confidence Level:** {decision.confidence:.1%}")
            
            st.markdown("---")
            st.markdown("### Head Trader Analysis")
            st.write(decision.agent_responses[-1].analysis)
            
            st.markdown("---")
            st.markdown("### Individual Agent Reports")
            
            for response in decision.agent_responses[:-1]:  # Exclude final decision
                st.markdown(f"#### {response.agent_name}")
                st.markdown(f"**Recommendation:** {response.recommendation} ({response.confidence:.0%})")
                st.write(response.analysis)
                st.markdown("---")

if __name__ == "__main__":
    main()