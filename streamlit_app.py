import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("💰 BTC/USD - Painel Profissional")

@st.cache_data
def carregar_dados():
    """Baixa e processa dados com tratamento robusto de erros"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 ano de dados
        
        dados = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        
        # Verificação completa dos dados
        if dados.empty or 'Close' not in dados.columns:
            st.error("Erro: Dados inválidos recebidos da API")
            return None
            
        # Limpeza e preparação dos dados
        dados = dados[['Close', 'Volume']].copy()
        dados = dados.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        
        # Cálculo seguro de indicadores
        dados['SMA_50'] = dados['Close'].rolling(50, min_periods=1).mean()
        dados['SMA_200'] = dados['Close'].rolling(200, min_periods=1).mean()
        
        return dados
    
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None

# Função segura para formatar valores
def formatar_valor(valor, tipo='preço'):
    """Formata valores sem usar .style.format()"""
    try:
        if tipo == 'preço':
            return f"${float(valor):,.2f}"
        elif tipo == 'porcentagem':
            return f"{float(valor):.2f}%"
        elif tipo == 'volume':
            return f"{int(valor):,}"
        else:
            return str(valor)
    except:
        return "N/A"

# Interface principal
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
            line=dict(color='#3498DB', width=1, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados['SMA_200'],
            name='Média 200 Dias',
            line=dict(color='#E74C3C', width=1, dash='dash')
        ))
        
        # Layout
        fig.update_layout(
            title='<b>BITCOIN (BTC/USD) - Análise Técnica</b>',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            hovermode='x unified',
            height=600,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seção de métricas
        st.subheader("📊 Métricas Atuais")
        
        if len(dados) >= 2:
            col1, col2, col3 = st.columns(3)
            
            # Último preço formatado manualmente
            ultimo_preco = formatar_valor(dados['Close'].iloc[-1], 'preço')
            col1.metric("Último Preço", ultimo_preco)
            
            # Variação 24h formatada manualmente
            variacao = ((dados['Close'].iloc[-1] / dados['Close'].iloc[-2] - 1) * 100
            variacao_formatada = formatar_valor(variacao, 'porcentagem')
            col2.metric("Variação 24h", variacao_formatada)
            
            # Volume formatado manualmente
            volume = formatar_valor(dados['Volume'].iloc[-1], 'volume')
            col3.metric("Volume 24h", volume)
        
        # Tabela de histórico (formatação segura)
        st.subheader("📈 Histórico Recente")
        
        # Criamos uma cópia para não modificar os dados originais
        dados_exibir = dados.tail(10).copy()
        
        # Aplicamos formatação manual em cada coluna
        dados_exibir['Close'] = dados_exibir['Close'].apply(lambda x: formatar_valor(x, 'preço'))
        dados_exibir['SMA_50'] = dados_exibir['SMA_50'].apply(lambda x: formatar_valor(x, 'preço'))
        dados_exibir['SMA_200'] = dados_exibir['SMA_200'].apply(lambda x: formatar_valor(x, 'preço'))
        dados_exibir['Volume'] = dados_exibir['Volume'].apply(lambda x: formatar_valor(x, 'volume'))
        
        # Exibimos a tabela
        st.dataframe(dados_exibir[['Close', 'SMA_50', 'SMA_200', 'Volume']])
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
else:
    st.warning("⚠️ Não foi possível carregar os dados. Tente novamente mais tarde.")

# Rodapé
st.markdown("---")
st.caption(f"ℹ️ Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dados: Yahoo Finance")
