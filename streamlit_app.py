import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Markov-Queue BTC Indicator - Dashboard")

# Sidebar (parâmetros ajustáveis)
with st.sidebar:
    st.header("Configurações")
    start_date = st.date_input("Data inicial", datetime(2023, 1, 1))
    end_date = st.date_input("Data final", datetime.today())
    rsi_period = st.slider("Período do RSI", 2, 50, 14)
    sma_period = st.slider("Período da SMA", 50, 500, 200)
    st.markdown("---")
    st.info("Configure os parâmetros técnicos acima")

# Baixar dados do BTC com tratamento robusto
@st.cache_data
def load_data():
    try:
        data = yf.download("BTC-USD", 
                          start=start_date, 
                          end=end_date + pd.Timedelta(days=1),  # Garante que inclui a end_date
                          progress=False)
        return data[['Close']].copy()  # Retorna apenas a coluna Close como DataFrame
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Cálculos técnicos com garantia de alinhamento
    close_prices = df['Close'].dropna()
    
    # Calcula SMA
    sma = close_prices.rolling(sma_period).mean()
    
    # Calcula RSI
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calcula Bollinger Bands
    rolling_mean = close_prices.rolling(20).mean()
    rolling_std = close_prices.rolling(20).std()
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    bb_width = ((bb_upper - bb_lower) / rolling_mean) * 100
    
    # Cria DataFrame alinhado
    analysis_df = pd.DataFrame({
        'Close': close_prices,
        'SMA': sma,
        'RSI': rsi,
        'BB_Width': bb_width
    }).dropna()
    
    # Definir estados
    conditions = [
        (analysis_df['Close'] > analysis_df['SMA']) & (analysis_df['RSI'] > 60),
        (analysis_df['Close'] < analysis_df['SMA']) & (analysis_df['RSI'] < 40),
        (analysis_df['BB_Width'] < 0.5)
    ]
    choices = ['Bull', 'Bear', 'Consolid']
    analysis_df['Estado'] = np.select(conditions, choices, default='Neutro')

    # Plot com Plotly
    fig = go.Figure()
    
    # Adicionar preço
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Close'],
        name='BTC Price',
        line=dict(color='gold'),
        hovertemplate='%{y:.2f} USD<extra></extra>'
    ))
    
    # Adicionar SMA
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['SMA'],
        name=f'SMA {sma_period}',
        line=dict(color='orange', dash='dot'),
        hovertemplate='%{y:.2f} USD<extra></extra>'
    ))

    # Adicionar áreas coloridas
    for estado, color in zip(['Bull', 'Bear', 'Consolid'], 
                           ['rgba(46,139,87,0.2)', 'rgba(178,34,34,0.2)', 'rgba(30,144,255,0.2)']):
        mask = analysis_df['Estado'] == estado
        if mask.any():
            starts = analysis_df.index[mask & ~mask.shift(1).fillna(False)]
            ends = analysis_df.index[mask & ~mask.shift(-1).fillna(False)]
            
            for start, end in zip(starts, ends):
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=color,
                    layer='below',
                    line_width=0
                )

    # Layout do gráfico
    fig.update_layout(
        title=f'BTC/USD - Markov-Queue Indicator (Último: {analysis_df["Close"].iloc[-1]:.2f} USD)',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        showlegend=True,
        height=600,
        template='plotly_dark'
    )

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar últimos sinais
    st.subheader('📈 Últimos Sinais')
    
    # Formatação condicional
    def highlight_state(val):
        color_map = {
            'Bull': 'background-color: rgba(46,139,87,0.3)',
            'Bear': 'background-color: rgba(178,34,34,0.3)',
            'Consolid': 'background-color: rgba(30,144,255,0.3)'
        }
        return color_map.get(val, '')

    last_signals = analysis_df[['Close', 'SMA', 'RSI', 'BB_Width', 'Estado']].tail(10)
    styled_df = last_signals.style.format({
        'Close': '{:.2f}',
        'SMA': '{:.2f}',
        'RSI': '{:.2f}',
        'BB_Width': '{:.2f}%'
    }).applymap(highlight_state, subset=['Estado'])
    
    st.dataframe(styled_df, use_container_width=True)

    # Explicação dos estados
    with st.expander('ℹ️ Como interpretar os sinais'):
        st.markdown('''
        ### 🎨 Legenda dos Estados
        - **🟢 Bull Market**: 
          - *Condições*: Preço acima da SMA + RSI > 60
          - *Estratégia*: Considerar compras (long)
          
        - **🔴 Bear Market**: 
          - *Condições*: Preço abaixo da SMA + RSI < 40
          - *Estratégia*: Considerar vendas (short) ou ficar fora
          
        - **🔵 Consolidação**: 
          - *Condições*: Volatilidade baixa (Bandas de Bollinger estreitas < 0.5%)
          - *Estratégia*: Esperar rompimento
          
        - **⚪ Neutro**: 
          - *Condições*: Sem sinal claro
          - *Estratégia*: Analisar outros fatores
        ''')

else:
    st.warning('Nenhum dado foi carregado. Verifique sua conexão com a internet.')
