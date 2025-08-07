import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Markov-Queue BTC Indicator - Dashboard")

# Sidebar (parâmetros ajustáveis)
with st.sidebar:
    st.header("Configurações")
    start_date = st.date_input("Data inicial", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Data final", pd.to_datetime("today"))
    rsi_period = st.slider("Período do RSI", 2, 50, 14)
    sma_period = st.slider("Período da SMA", 50, 500, 200)
    st.markdown("---")
    st.info("Configure os parâmetros técnicos acima")

# Baixar dados do BTC
@st.cache_data
def load_data():
    try:
        data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        return data.dropna()  # Remove linhas com NaN
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Cálculos técnicos com tratamento de NaN
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Evita divisão por zero
        return 100 - (100 / (1 + rs))

    # Aplicar cálculos e remover NaN novamente
    df["SMA"] = df["Close"].rolling(sma_period).mean()
    df["RSI"] = calculate_rsi(df["Close"], rsi_period)
    df["BB_Upper"] = df["Close"].rolling(20).mean() + 2*df["Close"].rolling(20).std()
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Lower"] = df["Close"].rolling(20).mean() - 2*df["Close"].rolling(20).std()
    df["BB_Width"] = ((df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]) * 100
    
    # Remover linhas com valores NaN após cálculos
    df = df.dropna()
    
    # Definir estados com verificação de NaN
    conditions = [
        (df["Close"] > df["SMA"]) & (df["RSI"] > 60) & (~df["Close"].isna()),
        (df["Close"] < df["SMA"]) & (df["RSI"] < 40) & (~df["Close"].isna()),
        (df["BB_Width"] < 0.5) & (~df["Close"].isna())
    ]
    choices = ["Bull", "Bear", "Consolid"]
    df["Estado"] = np.select(conditions, choices, default="Neutro")

    # Plot com Plotly
    fig = go.Figure()

    # Adicionar preço e SMA
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Close"], 
        name="BTC Price", 
        line=dict(color="gold"),
        hovertemplate="%{y:.2f} USD<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["SMA"], 
        name=f"SMA {sma_period}", 
        line=dict(color="orange", dash="dot"),
        hovertemplate="%{y:.2f} USD<extra></extra>"
    ))

    # Adicionar cores de fundo
    color_map = {
        "Bull": "rgba(46,139,87,0.2)",
        "Bear": "rgba(178,34,34,0.2)",
        "Consolid": "rgba(30,144,255,0.2)"
    }

    for estado in color_map:
        subset = df[df["Estado"] == estado]
        if not subset.empty:
            fig.add_vrect(
                x0=subset.index[0], 
                x1=subset.index[-1],
                fillcolor=color_map[estado], 
                layer="below", 
                line_width=0,
                annotation_text=estado,
                annotation_position="top left"
            )

    # Layout do gráfico
    fig.update_layout(
        title=f"BTC/USD - Markov-Queue Indicator (Último: {df['Close'].iloc[-1]:.2f} USD)",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        hovermode="x unified",
        showlegend=True,
        height=600,
        template="plotly_dark"
    )

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar últimos sinais
    st.subheader("📈 Últimos Sinais")
    last_signals = df[["Close", "SMA", "RSI", "BB_Width", "Estado"]].tail(10)
    
    # Formatação condicional
    def color_negative_red(val):
        color = ""
        if val.name == "Estado":
            if val == "Bull":
                color = "background-color: rgba(46,139,87,0.3)"
            elif val == "Bear":
                color = "background-color: rgba(178,34,34,0.3)"
            elif val == "Consolid":
                color = "background-color: rgba(30,144,255,0.3)"
        return color

    st.dataframe(
        last_signals.style.applymap(color_negative_red).format({
            "Close": "{:.2f}",
            "SMA": "{:.2f}",
            "RSI": "{:.2f}",
            "BB_Width": "{:.2f}%"
        }), 
        use_container_width=True
    )

    # Explicação dos estados
    with st.expander("ℹ️ Como interpretar os sinais"):
        st.markdown("""
        ### 🎨 Legenda dos Estados
        - **🟢 Bull Market**: 
          - *Condições*: Preço acima da SMA + RSI > 60
          - *Estratégia*: Considerar compras (long)
          
        - **🔴 Bear Market**: 
          - *Condições*: Preço abaixo da SMA + RSI < 40
          - *Estratégia*: Considerar vendas (short) ou ficar fora
          
        - **🔵 Consolidação**: 
          - *Condições*: Volatilidade baixa (Bandas de Bollinger estreitas < 0.5%)
          - *Estratégia*: Esperar rompimento (não operar contra o range)
          
        - **⚪ Neutro**: 
          - *Condições*: Sem sinal claro
          - *Estratégia*: Analisar outros fatores ou aguardar confirmação
        """)
else:
    st.warning("Nenhum dado foi carregado. Verifique sua conexão com a internet e as datas selecionadas.")
