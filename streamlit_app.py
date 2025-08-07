import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("🚀 BTC/USD - Gráfico Profissional")

@st.cache_data
def carregar_dados():
    """Baixa e processa dados com tratamento de erros robusto"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 anos de dados
        
        # Baixa dados com timeout de 10 segundos
        dados = yf.download(
            "BTC-USD",
            start=start_date,
            end=end_date,
            progress=False,
            timeout=10
        )
        
        # Verificação completa de dados inválidos
        if dados.empty or len(dados) < 10:
            raise ValueError("Dados insuficientes")
            
        # Seleciona e limpa colunas essenciais
        dados = dados[['Close', 'Volume']].copy()
        dados.replace([np.inf, -np.inf, 0], np.nan, inplace=True)
        dados.ffill(inplace=True)
        dados.dropna(inplace=True)
        
        # Calcula indicadores técnicos com tratamento de erros
        dados['SMA_50'] = dados['Close'].rolling(50, min_periods=1).mean()
        dados['SMA_200'] = dados['Close'].rolling(200, min_periods=1).mean()
        
        # Cálculo seguro do RSI
        delta = dados['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        dados['RSI'] = 100 - (100 / (1 + rs))
        
        return dados.dropna()
        
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
        return pd.DataFrame()

# Interface principal
dados = carregar_dados()

if not dados.empty:
    # Gráfico profissional
    fig = go.Figure()
    
    # 1. Linha do Preço (estilo profissional)
    fig.add_trace(go.Scatter(
        x=dados.index,
        y=dados['Close'],
        name='Preço BTC',
        line=dict(color='#F7931A', width=2.5),
        hovertemplate="<b>%{y:.2f} USD</b>",
        opacity=0.8
    ))
    
    # 2. Médias Móveis
    fig.add_trace(go.Scatter(
        x=dados.index,
        y=dados['SMA_50'],
        name='Média 50 Dias',
        line=dict(color='#3498DB', width=1.5, dash='dot'),
        visible='legendonly'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados.index,
        y=dados['SMA_200'],
        name='Média 200 Dias',
        line=dict(color='#E74C3C', width=1.5, dash='dash'),
        visible='legendonly'
    ))
    
    # 3. Área de Volume (opcional)
    fig.add_trace(go.Bar(
        x=dados.index,
        y=dados['Volume'],
        name='Volume',
        marker_color='rgba(100, 100, 100, 0.3)',
        yaxis='y2'
    ))
    
    # Layout avançado
    fig.update_layout(
        title='<b>BITCOIN (BTC/USD) - Análise Técnica</b>',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        height=700,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seção de análise
    with st.expander("📈 Análise Detalhada"):
        st.write(f"**Último preço:** {dados['Close'].iloc[-1]:.2f} USD")
        st.write(f"**Variação 24h:** {(dados['Close'].iloc[-1]/dados['Close'].iloc[-2]-1)*100:.2f}%")
        
        col1, col2 = st.columns(2)
        col1.metric("RSI Atual", f"{dados['RSI'].iloc[-1]:.1f}")
        col2.metric("Volume 24h", f"{dados['Volume'].iloc[-1]:,.0f}")
        
        st.dataframe(
            dados.tail(10)[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Volume']]
            .style.format({
                'Close': '{:.2f}',
                'SMA_50': '{:.2f}',
                'SMA_200': '{:.2f}',
                'RSI': '{:.1f}',
                'Volume': '{:,.0f}'
            }),
            height=300
        )
else:
    st.warning("""
    ⚠️ Não foi possível carregar os dados. 
    Verifique sua conexão com a internet ou tente novamente mais tarde.
    """)

# Rodapé profissional
st.markdown("---")
st.caption(f"""
Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} | 
Dados: Yahoo Finance | Desenvolvido com Python e Streamlit
""")
