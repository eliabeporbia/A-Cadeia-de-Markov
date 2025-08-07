import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Indicador Markov-Queue BTC - Versão Estável")

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
        data = yf.download("BTC-USD", 
                         start=start_date, 
                         end=end_date + pd.Timedelta(days=1),
                         progress=False)
        return data['Close'].rename('Close').to_frame()  # Garante que é um DataFrame com uma coluna
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return pd.DataFrame()

# Carregar dados
df = load_data()

if not df.empty:
    # Cálculos técnicos
    df['SMA'] = df['Close'].rolling(sma_period).mean()
    
    # Cálculo do RSI seguro
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
    
    # Remover NaN
    df = df.dropna()
    
    # Definir estados
    conditions = [
        (df['Close'] > df['SMA']) & (df['RSI'] > 60),
        (df['Close'] < df['SMA']) & (df['RSI'] < 40),
        (df['BB_Width'] < 0.5)
    ]
    df['Estado'] = np.select(conditions, ['Bull', 'Bear', 'Consolid'], 'Neutro')

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Preço BTC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name=f'SMA {sma_period}'))
    
    # Cores de fundo
    for estado, color in [('Bull', 'rgba(0,255,0,0.1)'), ('Bear', 'rgba(255,0,0,0.1)'), ('Consolid', 'rgba(0,0,255,0.1)')]:
        mask = df['Estado'] == estado
        starts = df.index[mask & ~mask.shift(1).fillna(False)]
        ends = df.index[mask & ~mask.shift(-1).fillna(False)]
        
        if len(starts) > 0:
            if len(starts) > len(ends):
                ends = ends.append(pd.Index([df.index[-1]]))
            
            for start, end in zip(starts, ends):
                fig.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0)

    st.plotly_chart(fig, use_container_width=True)
    
    # Últimos dados
    st.subheader("📊 Últimos Sinais")
    st.dataframe(
        df.tail(10)[['Close', 'SMA', 'RSI', 'Estado']].style.format({
            'Close': '{:.2f}', 'SMA': '{:.2f}', 'RSI': '{:.1f}'
        }),
        use_container_width=True
    )

else:
    st.warning("Não foi possível carregar os dados. Verifique sua conexão e as datas selecionadas.")
