import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Indicador Markov-Queue BTC - Versão Final")

# Sidebar com parâmetros
with st.sidebar:
    st.header("Configurações")
    data_inicio = st.date_input("Data inicial", datetime(2023, 1, 1))
    data_fim = st.date_input("Data final", datetime.today())
    periodo_rsi = st.slider("Período do RSI", 2, 50, 14)
    periodo_sma = st.slider("Período da SMA", 50, 500, 200)

# Função para baixar dados - CORRIGIDA
@st.cache_data
def carregar_dados():
    try:
        # Renomeando para evitar conflito com a função str()
        dados = yf.download("BTC-USD", 
                          start=data_inicio, 
                          end=data_fim + pd.Timedelta(days=1),
                          progress=False)
        return dados['Close'].to_frame(name='Close')  # Garantindo estrutura correta
    except Exception as erro:
        st.error(f"Erro ao baixar dados: {erro}")
        return pd.DataFrame()

# Carregar dados
dados_btc = carregar_dados()

if not dados_btc.empty:
    # Cálculos técnicos
    dados_btc['SMA'] = dados_btc['Close'].rolling(periodo_sma).mean()
    
    # Cálculo do RSI
    delta = dados_btc['Close'].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo_rsi).mean()
    media_perda = perda.rolling(periodo_rsi).mean().replace(0, np.nan)
    dados_btc['RSI'] = 100 - (100 / (1 + (media_ganho / media_perda)))
    
    # Bollinger Bands
    media_bb = dados_btc['Close'].rolling(20).mean()
    desvio_bb = dados_btc['Close'].rolling(20).std()
    dados_btc['BB_Upper'] = media_bb + 2 * desvio_bb
    dados_btc['BB_Lower'] = media_bb - 2 * desvio_bb
    dados_btc['BB_Width'] = ((dados_btc['BB_Upper'] - dados_btc['BB_Lower']) / media_bb) * 100
    
    # Remover NaN
    dados_btc = dados_btc.dropna()
    
    # Definir estados
    condicoes = [
        (dados_btc['Close'] > dados_btc['SMA']) & (dados_btc['RSI'] > 60),
        (dados_btc['Close'] < dados_btc['SMA']) & (dados_btc['RSI'] < 40),
        (dados_btc['BB_Width'] < 0.5)
    ]
    dados_btc['Estado'] = np.select(condicoes, ['Bull', 'Bear', 'Consolid'], 'Neutro')

    # Gráfico
    figura = go.Figure()
    figura.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['Close'],
        name='Preço BTC',
        line=dict(color='gold')
    ))
    figura.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['SMA'],
        name=f'SMA {periodo_sma}',
        line=dict(color='orange', dash='dot')
    ))
    
    # Cores de fundo
    cores_estado = {
        'Bull': 'rgba(46,139,87,0.2)',
        'Bear': 'rgba(178,34,34,0.2)',
        'Consolid': 'rgba(30,144,255,0.2)'
    }
    
    for estado, cor in cores_estado.items():
        mascara = dados_btc['Estado'] == estado
        mudancas = mascara.astype(int).diff()
        inicios = dados_btc.index[mudancas == 1]
        fins = dados_btc.index[mudancas == -1]
        
        if len(inicios) > 0:
            if len(inicios) > len(fins):
                fins = fins.append(pd.Index([dados_btc.index[-1]]))
            
            for inicio, fim in zip(inicios, fins):
                figura.add_vrect(
                    x0=inicio, x1=fim,
                    fillcolor=cor, layer="below",
                    line_width=0
                )

    st.plotly_chart(figura, use_container_width=True)
    
    # Últimos dados
    st.subheader("📊 Últimos Sinais")
    st.dataframe(
        dados_btc.tail(10)[['Close', 'SMA', 'RSI', 'BB_Width', 'Estado']].style.format({
            'Close': '{:.2f}', 
            'SMA': '{:.2f}', 
            'RSI': '{:.1f}', 
            'BB_Width': '{:.2f}%'
        }),
        use_container_width=True
    )

else:
    st.warning("Não foi possível carregar os dados. Verifique sua conexão e as datas selecionadas.")

# Explicação
with st.expander("ℹ️ Como interpretar os sinais"):
    st.markdown("""
    **🟢 Bull Market**: Preço > SMA + RSI > 60 (Tendência de alta)  
    **🔴 Bear Market**: Preço < SMA + RSI < 40 (Tendência de baixa)  
    **🔵 Consolidação**: BB Width < 0.5% (Mercado lateral)  
    **⚪ Neutro**: Sem sinal claro
    """)
