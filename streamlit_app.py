import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração básica do app
st.set_page_config(layout="wide")
st.title("📊 Indicador Markov-Queue para BTC")

# Parâmetros ajustáveis
with st.sidebar:
    st.header("Configurações")
    start_date = st.date_input("Data inicial", datetime(2023, 1, 1))
    end_date = st.date_input("Data final", datetime.today())
    rsi_period = st.slider("Período do RSI", 2, 50, 14)
    sma_period = st.slider("Período da SMA", 50, 500, 200)

# Baixar dados
@st.cache_data
def get_data():
    try:
        data = yf.download("BTC-USD", start=start_date, end=end_date + pd.Timedelta(days=1))
        return data[['Close']].dropna()
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

df = get_data()

if not df.empty:
    # Cálculos básicos
    df['SMA'] = df['Close'].rolling(sma_period).mean()
    
    # Cálculo seguro do RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Bollinger Bands simplificado
    rolling_mean = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = rolling_mean + 2 * rolling_std
    df['BB_Lower'] = rolling_mean - 2 * rolling_std
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / rolling_mean) * 100
    
    # Definir estados
    conditions = [
        (df['Close'] > df['SMA']) & (df['RSI'] > 60),
        (df['Close'] < df['SMA']) & (df['RSI'] < 40),
        (df['BB_Width'] < 0.5)
    ]
    df['Estado'] = np.select(conditions, ['Bull', 'Bear', 'Consolid'], 'Neutro')
    df = df.dropna()

    # Gráfico simplificado
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Preço BTC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name=f'SMA {sma_period}'))
    
    # Cores de fundo
    for estado, color in [('Bull', 'rgba(0,255,0,0.1)'), ('Bear', 'rgba(255,0,0,0.1)'), ('Consolid', 'rgba(0,0,255,0.1)')]:
        mask = df['Estado'] == estado
        if mask.any():
            starts = df.index[mask & ~mask.shift(1).fillna(False)]
            ends = df.index[mask & ~mask.shift(-1).fillna(False)]
            for s, e in zip(starts, ends):
                fig.add_vrect(x0=s, x1=e, fillcolor=color, layer="below", line_width=0)

    st.plotly_chart(fig, use_container_width=True)
    
    # Últimos sinais
    st.dataframe(df.tail(10)[['Close', 'SMA', 'RSI', 'Estado']].style.format({
        'Close': '{:.2f}', 'SMA': '{:.2f}', 'RSI': '{:.2f}'
    }), use_container_width=True)

else:
    st.warning("Não foi possível carregar os dados. Tente novamente mais tarde.")
