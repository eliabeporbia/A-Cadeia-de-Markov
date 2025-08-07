import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 BTC/USD - Monitor em Tempo Real")

@st.cache_data
def carregar_dados():
    """Baixa dados com tratamento robusto de erros"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 ano de dados
        
        dados = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        
        # Verificação de dados inválidos
        if dados.empty or 'Close' not in dados.columns:
            st.error("Erro: Dados inválidos do Yahoo Finance")
            return None
            
        # Limpeza dos dados
        dados = dados[['Close', 'Volume']].copy()
        dados = dados.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        
        # Cálculo de médias móveis
        dados['SMA_50'] = dados['Close'].rolling(50, min_periods=1).mean()
        dados['SMA_200'] = dados['Close'].rolling(200, min_periods=1).mean()
        
        return dados
    
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None

# Carrega dados
dados = carregar_dados()

if dados is not None and not dados.empty:
    try:
        # Gráfico interativo
        fig = go.Figure()
        
        # Linha do preço (BTC)
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['Close'],
            name='Preço BTC',
            line=dict(color='#F7931A', width=2),
            hovertemplate="<b>%{y:.2f} USD</b>"
        ))
        
        # Médias móveis
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['SMA_50'],
            name='Média 50 Dias',
            line=dict(color='#3498DB', width=1, dash='dot'),
            visible='legendonly'
        ))
        
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['SMA_200'],
            name='Média 200 Dias',
            line=dict(color='#E74C3C', width=1, dash='dash'),
            visible='legendonly'
        ))
        
        # Layout
        fig.update_layout(
            title='<b>Preço do Bitcoin (BTC/USD)</b>',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            hovermode='x unified',
            height=600,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas (formatação segura)
        if len(dados) >= 2:
            ultimo_preco = dados['Close'].iloc[-1]
            variacao = ((dados['Close'].iloc[-1] / dados['Close'].iloc[-2] - 1) * 100)
            
            col1, col2 = st.columns(2)
            col1.metric("Último Preço", f"${ultimo_preco:,.2f}")
            col2.metric("Variação 24h", f"{variacao:.2f}%")
        
        # Tabela de histórico (sem .style.format)
        st.subheader("Últimos 10 Dias")
        dados_tabela = dados.tail(10).copy()
        
        # Formatação manual para evitar erros
        dados_tabela['Close'] = dados_tabela['Close'].apply(lambda x: f"${x:,.2f}")
        dados_tabela['SMA_50'] = dados_tabela['SMA_50'].apply(lambda x: f"${x:,.2f}")
        dados_tabela['SMA_200'] = dados_tabela['SMA_200'].apply(lambda x: f"${x:,.2f}")
        dados_tabela['Volume'] = dados_tabela['Volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(dados_tabela[['Close', 'SMA_50', 'SMA_200', 'Volume']])
        
    except Exception as e:
        st.error(f"Erro ao exibir dados: {str(e)}")
else:
    st.warning("⚠️ Não foi possível carregar os dados. Tente novamente mais tarde.")

# Rodapé
st.markdown("---")
st.caption(f"ℹ️ Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dados: Yahoo Finance")
