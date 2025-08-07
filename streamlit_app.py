import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 BTC/USD - Monitor Profissional")

@st.cache_data
def carregar_dados():
    """Baixa e processa dados com tratamento robusto de erros"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 anos de dados
        
        dados = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        
        # Verificação completa dos dados
        if dados.empty or 'Close' not in dados.columns:
            st.error("Dados inválidos recebidos da API")
            return None
            
        # Limpeza e preparação dos dados
        dados = dados[['Close', 'Volume']].copy()
        dados = dados.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        
        # Cálculo seguro de indicadores
        dados['SMA_50'] = dados['Close'].rolling(50, min_periods=1).mean()
        dados['SMA_200'] = dados['Close'].rolling(200, min_periods=1).mean()
        
        return dados
    
    except Exception as e:
        st.error(f"Erro na coleta de dados: {str(e)}")
        return None

# Interface principal
dados = carregar_dados()

if dados is not None and not dados.empty:
    try:
        # Gráfico principal
        fig = go.Figure()
        
        # Linha do preço (BTC)
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['Close'],
            name='Preço BTC',
            line=dict(color='#F7931A', width=2.5),
            hovertemplate="<b>%{y:.2f} USD</b>"
        ))
        
        # Médias móveis
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['SMA_50'],
            name='Média 50 Dias',
            line=dict(color='#3498db', width=1.5, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['SMA_200'],
            name='Média 200 Dias',
            line=dict(color='#e74c3c', width=1.5, dash='dash')
        ))
        
        # Layout profissional
        fig.update_layout(
            title='<b>BITCOIN (BTC/USD) - Análise Técnica</b>',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            hovermode='x unified',
            height=650,
            template='plotly_dark',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seção de métricas - FORMA SEGURA DE FORMATAR
        if len(dados) >= 2:
            ultimo_preco = dados['Close'].iloc[-1]
            preco_anterior = dados['Close'].iloc[-2]
            variacao = ((ultimo_preco / preco_anterior) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Último Preço", f"${ultimo_preco:,.2f}")
            col2.metric("Variação 24h", f"{variacao:.2f}%")
            col3.metric("Volume 24h", f"{dados['Volume'].iloc[-1]:,.0f}")
        
        # Tabela de dados - FORMA SEGURA DE EXIBIR
        st.subheader("Histórico Recente")
        dados_exibir = dados.tail(10).copy()
        
        # Formatação segura sem usar .style.format()
        dados_exibir['Close'] = dados_exibir['Close'].apply(lambda x: f"${x:,.2f}")
        dados_exibir['SMA_50'] = dados_exibir['SMA_50'].apply(lambda x: f"${x:,.2f}")
        dados_exibir['SMA_200'] = dados_exibir['SMA_200'].apply(lambda x: f"${x:,.2f}")
        dados_exibir['Volume'] = dados_exibir['Volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(dados_exibir[['Close', 'SMA_50', 'SMA_200', 'Volume']], height=350)
        
    except Exception as e:
        st.error(f"Erro na visualização: {str(e)}")
else:
    st.warning("Não foi possível carregar os dados do Bitcoin. Por favor, tente novamente mais tarde.")

# Rodapé profissional
st.markdown("---")
st.caption(f"ℹ️ Dados do Yahoo Finance | Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
