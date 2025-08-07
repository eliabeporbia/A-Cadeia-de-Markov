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

# Carregar modelo existente ou criar novo
def carregar_modelo():
    if os.path.exists('modelo_btc.pkl'):
        try:
            modelo = joblib.load('modelo_btc.pkl')
            st.sidebar.success("Modelo carregado com sucesso!")
            return modelo
        except:
            st.sidebar.warning("Erro ao carregar modelo. Criando novo modelo...")
            return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        st.sidebar.info("Criando novo modelo...")
        return RandomForestClassifier(n_estimators=100, random_state=42)

modelo = carregar_modelo()

# Função para criar features
def criar_features(df):
    # Cálculo do RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))  # Evita divisão por zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Outros indicadores
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

# Função para criar target (rótulos)
def criar_target(df, dias_futuro=3):
    df['Target'] = (df['Close'].shift(-dias_futuro) > df['Close']).astype(int)
    return df.dropna()

# Baixar e preparar dados
@st.cache_data
def carregar_dados():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*3)  # 3 anos de dados
    
    dados = yf.download("BTC-USD", start=start_date, end=end_date)
    dados = criar_features(dados)
    dados = criar_target(dados)
    return dados

dados = carregar_dados()

# Treinar/atualizar modelo - AGORA OBRIGATÓRIO
def treinar_modelo():
    X = dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 'Retorno_1D', 'Retorno_7D', 'Volatilidade']]
    y = dados['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, predicoes)
    
    joblib.dump(modelo, 'modelo_btc.pkl')
    st.sidebar.success(f"Modelo treinado! Acurácia: {acuracia:.2%}")
    return modelo

# Treinar o modelo ao iniciar (ou usar o carregado se já existir)
if not hasattr(modelo, 'feature_importances_'):
    modelo = treinar_modelo()

# Fazer previsões APÓS garantir que o modelo está treinado
dados['Previsao'] = modelo.predict(dados[['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 
                                        'Retorno_1D', 'Retorno_7D', 'Volatilidade']])

# Visualização
fig = go.Figure()

# Preço
fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço BTC', line=dict(color='gold')))

# Sinais de compra
compras = dados[dados['Previsao'] == 1]
fig.add_trace(go.Scatter(
    x=compras.index,
    y=compras['Close'],
    mode='markers',
    marker=dict(color='green', size=8),
    name='Sinal de Compra'
))

# SMA
fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_50'], name='SMA 50', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_200'], name='SMA 200', line=dict(color='red')))

# Layout
fig.update_layout(
    title='BTC/USD com Sinais Autoajustáveis',
    xaxis_title='Data',
    yaxis_title='Preço (USD)',
    hovermode='x unified',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Botão para retreinar manualmente
if st.sidebar.button("Retreinar Modelo"):
    modelo = treinar_modelo()
    st.experimental_rerun()

# Restante do seu código (explicação, importância das features, etc.)
with st.expander("ℹ️ Como funciona este indicador autoajustável"):
    st.markdown("""
    ## Sistema de Autoaprendizado
    
    Este indicador usa machine learning para:
    
    1. **Coletar dados históricos** do BTC
    2. **Extrair features técnicas** (SMA, RSI, Bollinger Bands)
    3. **Definir um alvo** (se o preço subirá nos próximos dias)
    4. **Treinar um modelo** de classificação
    5. **Ajustar-se automaticamente** com novos dados
    
    ## Principais características:
    
    - 🤖 **Autoaprendizado**: Melhora com o tempo ao ser re-treinado
    - 🔄 **Auto-correção**: Ajusta-se a novas condições de mercado
    - 📈 **Adaptabilidade**: Aprende padrões específicos do BTC
    - 💾 **Persistência**: Salva o modelo entre sessões
    
    ## Como usar:
    
    1. Clique em "Retreinar Modelo" periodicamente
    2. Observe os sinais de compra (pontos verdes)
    3. O modelo mostrará sua confiança (acurácia)
    4. O sistema continuará aprendendo com novos dados
    """)

# Mostrar importância das features
if st.checkbox("Mostrar importância das features"):
    importancias = pd.DataFrame({
        'Feature': ['SMA_50', 'SMA_200', 'RSI', 'BB_Upper', 'BB_Lower', 'Retorno_1D', 'Retorno_7D', 'Volatilidade'],
        'Importância': modelo.feature_importances_
    }).sort_values('Importância', ascending=False)
    
    st.bar_chart(importancias.set_index('Feature'))

# Configurações avançadas
with st.sidebar.expander("Configurações Avançadas"):
    dias_previsao = st.slider("Dias para previsão", 1, 7, 3)
    limite_confianca = st.slider("Limite de confiança", 0.5, 0.9, 0.7)
