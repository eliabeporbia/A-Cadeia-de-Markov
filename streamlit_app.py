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
        start_date = end_date - timedelta(days=365)
        
        dados = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        
        # Verificação completa dos dados
        if dados.empty or 'Close' not in dados.columns:
            st.error("Dados inválidos recebidos do Yahoo Finance")
            return None
            
        # Limpeza dos dados
        dados = dados[['Close', 'Volume']].copy()
        dados = dados.replace([np.inf, -np.inf, 0], np.nan).ffill().dropna()
        
        # Cálculo de indicadores
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
        # Verificação final antes de acessar os dados
        if len(dados) > 0 and 'Close' in dados.columns:
            # Gráfico principal
            fig = go.Figure()
            
            # Linha do preço
            fig.add_trace(go.Scatter(
                x=dados.index,
                y=dados['Close'],
                name='Preço BTC',
                line=dict(color='#F7931A', width=2)
            ))
            
            # Layout
            fig.update_layout(
                title='Preço do Bitcoin (BTC/USD)',
                xaxis_title='Data',
                yaxis_title='Preço (USD)',
                height=600,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Exibição segura dos valores
            ultimo_preco = dados['Close'].iloc[-1] if len(dados) > 0 else None
            if ultimo_preco is not None:
                st.write(f"**Último preço:** {ultimo_preco:.2f} USD")
                
                # Cálculo seguro da variação
                if len(dados) >= 2:
                    variacao = ((dados['Close'].iloc[-1] / dados['Close'].iloc[-2]) - 1) * 100
                    st.write(f"**Variação 24h:** {variacao:.2f}%")
                
                # Tabela com os últimos valores
                st.dataframe(
                    dados[['Close', 'SMA_50', 'SMA_200']].tail(10).style.format({
                        'Close': '{:.2f}',
                        'SMA_50': '{:.2f}',
                        'SMA_200': '{:.2f}'
                    }),
                    height=300
                )
            else:
                st.warning("Não foi possível obter o último preço")
        else:
            st.error("Estrutura de dados inválida")
            
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
else:
    st.warning("Não foi possível carregar dados do Bitcoin. Tente novamente mais tarde.")

# Rodapé
st.markdown("---")
st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
