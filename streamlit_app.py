
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Markov-Queue BTC Indicator - Dashboard")

# Sidebar (parâmetros ajustáveis)
st.sidebar.header("Configurações")
start_date = st.sidebar.date_input("Data inicial", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Data final", pd.to_datetime("today"))
rsi_period = st.sidebar.slider("Período do RSI", 2, 50, 14)
sma_period = st.sidebar.slider("Período da SMA", 50, 500, 200)

# Baixar dados do BTC
@st.cache_data
def load_data():
    return yf.download("BTC-USD", start=start_date, end=end_date)

df = load_data()

# Cálculos técnicos
df["SMA"] = df["Close"].rolling(sma_period).mean()
df["RSI"] = 100 - (100 / (1 + (df["Close"].diff(1).clip(lower=0).rolling(rsi_period).mean() / 
                      -df["Close"].diff(1).clip(upper=0).rolling(rsi_period).mean()))
df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = df["Close"].rolling(20).mean() + 2*df["Close"].rolling(20).std(), \
                                                 df["Close"].rolling(20).mean(), \
                                                 df["Close"].rolling(20).mean() - 2*df["Close"].rolling(20).std()
df["BB_Width"] = ((df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]) * 100

# Definir estados
conditions = [
    (df["Close"] > df["SMA"]) & (df["RSI"] > 60),  # Bull
    (df["Close"] < df["SMA"]) & (df["RSI"] < 40),   # Bear
    (df["BB_Width"] < 0.5)                          # Consolid
]
choices = ["Bull", "Bear", "Consolid"]
df["Estado"] = np.select(conditions, choices, default="Neutro")

# Plot com Plotly
fig = go.Figure()

# Adicionar preço e SMA
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], name=f"SMA {sma_period}", line=dict(color="orange")))

# Adicionar cores de fundo
for estado, color in zip(["Bull", "Bear", "Consolid"], ["rgba(46,139,87,0.2)", "rgba(178,34,34,0.2)", "rgba(30,144,255,0.2)"]):
    subset = df[df["Estado"] == estado]
    if not subset.empty:
        fig.add_vrect(
            x0=subset.index[0], x1=subset.index[-1],
            fillcolor=color, layer="below", line_width=0
        )

# Layout do gráfico
fig.update_layout(
    title="BTC com Markov-Queue Indicator",
    xaxis_title="Data",
    yaxis_title="Preço (USD)",
    hovermode="x unified",
    showlegend=True,
    height=600
)

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Mostrar últimos sinais
st.subheader("📈 Últimos Sinais")
last_signals = df[["Close", "SMA", "RSI", "BB_Width", "Estado"]].tail(10).style.apply(
    lambda x: ["background: rgba(46,139,87,0.3)" if v == "Bull" else 
               "background: rgba(178,34,34,0.3)" if v == "Bear" else
               "background: rgba(30,144,255,0.3)" if v == "Consolid" else "" for v in x], 
    subset=["Estado"])
st.dataframe(last_signals, use_container_width=True)

# Explicação dos estados
st.markdown("""
### 🎨 Legenda dos Estados
- **🟢 Bull Market**: Preço acima da SMA + RSI > 60 (compra)
- **🔴 Bear Market**: Preço abaixo da SMA + RSI < 40 (venda)
- **🔵 Consolidação**: Volatilidade baixa (Bandas de Bollinger estreitas)
- **⚪ Neutro**: Sem sinal claro
""")
