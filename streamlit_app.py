import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Indicador Markov-Queue BTC")

# Sidebar com parâmetros
with st.sidebar:
    st.header("Configurações")
    start_date = st.date_input("Data inicial", datetime(2023, 1, 1))
    end_date = st.date_input("Data final", datetime.today())
    rsi_period = st.slider("Período do RSI", 2, 50, 14)
    sma_period = st.slider("Período da SMA", 50, 500, 200)

# Função para baixar dados
@st.cache_data
def load_data():
    try:
        data = yf.download("BTC-USD", start=start_date, end=end_date + pd.Timedelta(days=1))
        return data[['Close']].dropna()
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

# Carregar dados
btc_data = load_data()

if not btc_data.empty:
    # Cálculos técnicos
    btc_data = btc_data.copy()
    btc_data['SMA'] = btc_data['Close'].rolling(sma_period).mean()
    
    # Cálculo do RSI
    delta = btc_data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean().replace(0, np.nan)
    btc_data['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Bollinger Bands - Versão simplificada e segura
    bb_window = 20
    btc_data['BB_Mid'] = btc_data['Close'].rolling(bb_window).mean()
    btc_data['BB_Std'] = btc_data['Close'].rolling(bb_window).std()
    btc_data['BB_Upper'] = btc_data['BB_Mid'] + (2 * btc_data['BB_Std'])
    btc_data['BB_Lower'] = btc_data['BB_Mid'] - (2 * btc_data['BB_Std'])
    btc_data['BB_Width'] = ((btc_data['BB_Upper'] - btc_data['BB_Lower']) / btc_data['BB_Mid']) * 100
    
    # Definir estados de mercado
    conditions = [
        (btc_data['Close'] > btc_data['SMA']) & (btc_data['RSI'] > 60),
        (btc_data['Close'] < btc_data['SMA']) & (btc_data['RSI'] < 40),
        (btc_data['BB_Width'] < 0.5)
    ]
    btc_data['Estado'] = np.select(conditions, ['Bull', 'Bear', 'Consolid'], 'Neutro')
    btc_data = btc_data.dropna()

    # Gráfico principal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Close'], name='Preço BTC', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['SMA'], name=f'SMA {sma_period}', line=dict(color='orange', dash='dot')))
    
    # Adicionar áreas coloridas
    color_map = {
        'Bull': 'rgba(46,139,87,0.2)',
        'Bear': 'rgba(178,34,34,0.2)',
        'Consolid': 'rgba(30,144,255,0.2)'
    }
    
    for estado, color in color_map.items():
        mask = btc_data['Estado'] == estado
        changes = mask.astype(int).diff()
        starts = btc_data.index[changes == 1]
        ends = btc_data.index[changes == -1]
        
        if len(starts) > 0:
            if len(starts) > len(ends):
                ends = np.append(ends, btc_data.index[-1])
            
            for start, end in zip(starts, ends):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=color, layer="below",
                    line_width=0
                )

    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar últimos dados
    st.subheader("📊 Últimos Sinais")
    st.dataframe(
        btc_data.tail(10)[['Close', 'SMA', 'RSI', 'BB_Width', 'Estado']].style.format({
            'Close': '{:.2f}', 'SMA': '{:.2f}', 
            'RSI': '{:.1f}', 'BB_Width': '{:.2f}%'
        }),
        use_container_width=True
    )

else:
    st.warning("Não foi possível carregar os dados do BTC. Verifique sua conexão com a internet.")
