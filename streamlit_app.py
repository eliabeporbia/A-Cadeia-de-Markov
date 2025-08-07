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

# Função para baixar e preparar dados
@st.cache_data
def load_and_prepare_data():
    try:
        # Baixar dados
        data = yf.download("BTC-USD", start=start_date, end=end_date + pd.Timedelta(days=1))
        
        if data.empty:
            return pd.DataFrame()
            
        # Criar DataFrame com índice explícito
        df = pd.DataFrame({
            'Close': data['Close'].values
        }, index=data.index)
        
        # Calcular SMA
        df['SMA'] = df['Close'].rolling(sma_period).mean()
        
        # Cálculo do RSI seguro
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean().replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # Bollinger Bands com alinhamento garantido
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = bb_middle + (2 * bb_std)
        df['BB_Lower'] = bb_middle - (2 * bb_std)
        df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / bb_middle) * 100
        
        # Remover valores NaN
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {e}")
        return pd.DataFrame()

# Carregar e preparar dados
btc_data = load_and_prepare_data()

if not btc_data.empty:
    # Definir estados de mercado com verificação de NaN
    conditions = [
        btc_data['Close'].gt(btc_data['SMA']) & btc_data['RSI'].gt(60),
        btc_data['Close'].lt(btc_data['SMA']) & btc_data['RSI'].lt(40),
        btc_data['BB_Width'].lt(0.5)
    ]
    btc_data['Estado'] = np.select(conditions, ['Bull', 'Bear', 'Consolid'], 'Neutro')

    # Gráfico principal
    fig = go.Figure()
    
    # Adicionar preço e SMA
    fig.add_trace(go.Scatter(
        x=btc_data.index,
        y=btc_data['Close'],
        name='Preço BTC',
        line=dict(color='gold')
    ))
    fig.add_trace(go.Scatter(
        x=btc_data.index,
        y=btc_data['SMA'],
        name=f'SMA {sma_period}',
        line=dict(color='orange', dash='dot')
    ))
    
    # Adicionar áreas coloridas
    color_map = {
        'Bull': 'rgba(46,139,87,0.2)',
        'Bear': 'rgba(178,34,34,0.2)',
        'Consolid': 'rgba(30,144,255,0.2)'
    }
    
    for estado, color in color_map.items():
        mask = btc_data['Estado'] == estado
        starts = btc_data.index[mask & ~mask.shift(1).fillna(False)]
        ends = btc_data.index[mask & ~mask.shift(-1).fillna(False)]
        
        if len(starts) > 0:
            if len(starts) > len(ends):
                ends = ends.append(pd.Index([btc_data.index[-1]]))
            
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
    st.warning("Não foi possível carregar os dados. Verifique sua conexão e as datas selecionadas.")

# Adicionar explicação
with st.expander("ℹ️ Como interpretar os sinais"):
    st.markdown("""
    **🟢 Bull Market**: Preço > SMA + RSI > 60 (Considere comprar)  
    **🔴 Bear Market**: Preço < SMA + RSI < 40 (Considere vender)  
    **🔵 Consolidação**: Volatilidade baixa (BB Width < 0.5%)  
    **⚪ Neutro**: Sem sinal claro
    """)
