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

# 1. Função para garantir dados limpos
def limpar_dados(df):
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('D').ffill()
    return df

# 2. Carregar modelo
def carregar_modelo():
    if os.path.exists('modelo_btc.pkl'):
        modelo = joblib.load('modelo_btc.pkl')
        st.sidebar.success("Modelo carregado!")
        return modelo
    return RandomForestClassifier(n_estimators=200, random_state=42)

# 3. Processamento de dados
def criar_features(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Médias móveis
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # Bollinger Bands
    df['BB_Upper'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
    df['BB_Lower'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
    
    # Outras features
    df['Retorno_1D'] = df['Close'].pct_change()
    df['Retorno_7D'] = df['Close'].pct_change(7)
    df['Volatilidade'] = df['Close'].rolling(7).std()
    
    return df.dropna()

def criar_target(df, dias=3):
    df['Target'] = (df['Close'].shift(-dias) > df['Close']).astype(int)
    return df.dropna()

@st.cache_data
def carregar_dados():
    end = datetime.today()
    start = end - timedelta(days=365*3)
    df = yf.download("BTC-USD", start=start, end=end)
    df = limpar_dados(df)
    df = criar_features(df)
    df = criar_target(df)
    return df

# 4. Treinamento
def treinar_modelo(dados):
    X = dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 'Retorno_1D', 'Retorno_7D', 'Volatilidade']]
    y = dados['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    modelo = carregar_modelo()
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, 'modelo_btc.pkl')
    return modelo

# 5. Interface principal
dados = carregar_dados()
modelo = carregar_modelo()

# Auto-treinamento
if 'ultimo_treinamento' not in st.session_state:
    modelo = treinar_modelo(dados)
    st.session_state.ultimo_treinamento = datetime.now()

# Previsões
dados['Previsao'] = modelo.predict(dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 
                                        'Retorno_1D', 'Retorno_7D', 'Volatilidade']])

# 6. GRÁFICO CORRIGIDO (100% FUNCIONAL)
fig = go.Figure()

# Linha do Preço (OBRIGATÓRIA)
fig.add_trace(go.Scatter(
    x=dados.index,
    y=dados['Close'],
    name='Preço BTC',
    line=dict(color='#F7931A', width=2),
    hovertemplate="<b>%{y:.2f} USD</b>"
))

# Sinais de Compra (DESTAQUE)
compras = dados[dados['Previsao'] == 1]
fig.add_trace(go.Scatter(
    x=compras.index,
    y=compras['Close'],
    mode='markers',
    marker=dict(
        color='#00FF7F',
        size=10,
        symbol='triangle-up',
        line=dict(width=1, color='DarkGreen')
    ),
    name='Sinal de Compra'
))

# Layout Garantido
fig.update_layout(
    title='<b>BITCOIN - Preço e Sinais</b>',
    xaxis_title='Data',
    yaxis_title='Preço (USD)',
    template='plotly_dark',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Controles
if st.sidebar.button("🔄 Atualizar Modelo"):
    modelo = treinar_modelo(dados)
    st.rerun()

# Debug (opcional)
if st.checkbox("Mostrar dados brutos"):
    st.dataframe(dados)
