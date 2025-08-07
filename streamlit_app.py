import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import plotly.graph_objects as go

# Configuração do app
st.set_page_config(layout="wide")
st.title("🤖 Indicador BTC Autoajustável")

# Função para carregar/treinar modelo
def carregar_modelo():
    if os.path.exists('modelo_btc.pkl'):
        try:
            modelo = joblib.load('modelo_btc.pkl')
            st.sidebar.success("Modelo carregado com sucesso!")
            return modelo
        except:
            st.sidebar.warning("Erro ao carregar modelo. Criando novo...")
            return RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        st.sidebar.info("Criando novo modelo...")
        return RandomForestClassifier(n_estimators=200, random_state=42)

# Funções para processamento de dados
def criar_features(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['BB_Upper'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
    df['BB_Lower'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
    df['Retorno_1D'] = df['Close'].pct_change()
    df['Retorno_7D'] = df['Close'].pct_change(7)
    df['Volatilidade'] = df['Close'].rolling(7).std()
    
    return df.dropna()

def criar_target(df, dias_futuro=3):
    df['Target'] = (df['Close'].shift(-dias_futuro) > df['Close']).astype(int)
    return df.dropna()

@st.cache_data
def carregar_dados():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*3)
    dados = yf.download("BTC-USD", start=start_date, end=end_date)
    dados = dados[['Close']].copy()
    dados = criar_features(dados)
    dados = criar_target(dados)
    return dados

# Treinamento do modelo
def treinar_modelo(dados):
    X = dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 'Retorno_1D', 'Retorno_7D', 'Volatilidade']]
    y = dados['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    modelo = carregar_modelo()
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, predicoes)
    
    joblib.dump(modelo, 'modelo_btc.pkl')
    st.sidebar.success(f"Modelo atualizado! Acurácia: {acuracia:.2%}")
    return modelo

# Interface principal
dados = carregar_dados()
modelo = carregar_modelo()

# Auto-treinamento (semanal)
if 'ultimo_treinamento' not in st.session_state or (datetime.now() - st.session_state.ultimo_treinamento).days >= 7:
    modelo = treinar_modelo(dados)
    st.session_state.ultimo_treinamento = datetime.now()
    st.session_state.modelo = modelo
    st.toast("Modelo auto-atualizado!", icon="🤖")

# Previsões
dados['Previsao'] = modelo.predict(dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 
                                        'Retorno_1D', 'Retorno_7D', 'Volatilidade']])

# Visualização corrigida
fig = go.Figure()

# 1. Linha do Preço do BTC (PRINCIPAL)
fig.add_trace(go.Scatter(
    x=dados.index,
    y=dados['Close'],
    name='Preço BTC',
    line=dict(color='#F7931A', width=2),  # Laranja Bitcoin
    hovertemplate="<b>Preço: %{y:.2f} USD</b><extra></extra>"
))

# 2. Sinais de Compra (DESTAQUE)
compras = dados[dados['Previsao'] == 1]
fig.add_trace(go.Scatter(
    x=compras.index,
    y=compras['Close'],
    mode='markers',
    marker=dict(
        color='#00FF7F',
        size=10,
        symbol='triangle-up',
        line=dict(width=2, color='DarkGreen')
    ),
    name='Sinal de Compra',
    hovertemplate="<b>Sinal de Compra</b><br>Preço: %{y:.2f} USD<extra></extra>"
))

# 3. Médias Móveis (OPCIONAIS)
fig.add_trace(go.Scatter(
    x=dados.index,
    y=dados['SMA_50'],
    name='Média 50 Dias',
    line=dict(color='#1E90FF', width=1),
    visible='legendonly'
))

fig.add_trace(go.Scatter(
    x=dados.index,
    y=dados['SMA_200'],
    name='Média 200 Dias',
    line=dict(color='#FF6347', width=1),
    visible='legendonly'
))

# Layout profissional
fig.update_layout(
    title='<b>Preço do Bitcoin com Sinais de Compra</b>',
    xaxis_title='Data',
    yaxis_title='Preço (USD)',
    hovermode='x unified',
    height=700,
    template='plotly_dark',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=60, b=20)
)

# Exibição
st.plotly_chart(fig, use_container_width=True)

# Controles
with st.sidebar:
    if st.button("🔁 Retreinar Modelo Agora"):
        modelo = treinar_modelo(dados)
        st.rerun()
    
    st.info(f"Último treinamento: {st.session_state.ultimo_treinamento.strftime('%d/%m/%Y %H:%M')}")
    
    with st.expander("⚙️ Configurações"):
        dias_previsao = st.slider("Horizonte de previsão (dias)", 1, 7, 3)
        st.caption("Recomendado: 3 dias para melhor acurácia")

# Explicações
with st.expander("📚 Como Interpretar o Gráfico"):
    st.markdown("""
    ## 📊 Elementos do Gráfico:
    - **Linha Laranja**: Preço histórico do Bitcoin (BTC-USD)
    - **Marcadores Verdes**: Sinais de compra gerados pelo modelo
    - **Médias Móveis**: Ative/desative na legenda (50 e 200 dias)

    ## 🔍 Dica:
    - Zoom com mouse (selecione área)
    - Passe o mouse sobre os pontos para detalhes
    - Clique na legenda para mostrar/esconder elementos
    """)

# Importância das features
if st.checkbox("📊 Mostrar importância dos indicadores"):
    importancias = pd.DataFrame({
        'Indicador': ['SMA 50', 'SMA 200', 'RSI', 'Bollinger Superior', 'Bollinger Inferior', 'Retorno 1D', 'Retorno 7D', 'Volatilidade'],
        'Importância': modelo.feature_importances_
    }).sort_values('Importância', ascending=False)
    
    st.bar_chart(importancias.set_index('Indicador'), color='#F7931A')
