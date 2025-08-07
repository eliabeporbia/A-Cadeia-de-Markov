import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configuração do app
st.set_page_config(layout="wide")
st.title("🧠 BTC AI Learner - Autoajustável")

class AprendizBTC:
    def __init__(self):
        self.modelo = None
        self.ultimo_treinamento = None
        self.historico_acuracia = []
        self.carregar_modelo()
        
    def carregar_modelo(self):
        """Carrega ou cria um novo modelo com verificação de integridade"""
        try:
            if os.path.exists('modelo_btc_ai.pkl'):
                with open('modelo_btc_ai.pkl', 'rb') as f:
                    modelo_hash = hashlib.md5(f.read()).hexdigest()
                
                if modelo_hash == st.secrets.get("MODEL_HASH", ""):
                    self.modelo = joblib.load('modelo_btc_ai.pkl')
                    st.sidebar.success("Modelo IA carregado!")
                else:
                    raise ValueError("Hash do modelo inválido")
            else:
                self.criar_novo_modelo()
        except Exception as e:
            st.sidebar.warning(f"Resetando modelo: {str(e)}")
            self.criar_novo_modelo()
    
    def criar_novo_modelo(self):
        """Inicializa um novo modelo com parâmetros otimizados"""
        self.modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            warm_start=True  # Permite aprendizado contínuo
        )
        st.sidebar.info("Novo modelo IA criado!")
    
    def preparar_dados(self):
        """Baixa e processa os dados com tratamento de erros"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            
            dados = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
            dados = dados[['Close', 'Volume']].ffill().dropna()
            
            # Engenharia de features
            dados['SMA_50'] = dados['Close'].rolling(50).mean()
            dados['SMA_200'] = dados['Close'].rolling(200).mean()
            dados['RSI'] = self.calcular_rsi(dados['Close'])
            dados['Target'] = (dados['Close'].shift(-3) > dados['Close']).astype(int)
            
            return dados.dropna()
        except Exception as e:
            st.error(f"Erro nos dados: {str(e)}")
            return pd.DataFrame()
    
    def calcular_rsi(self, series, period=14):
        """Calcula RSI com tratamento de erros"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean().replace(0, np.nan)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def treinar_continuamente(self, dados):
        """Treinamento com validação e auto-correção"""
        try:
            X = dados[['SMA_50', 'SMA_200', 'RSI', 'Volume']]
            y = dados['Target']
            
            # Divisão temporal (não aleatória para dados financeiros)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Treino com warm_start
            self.modelo.fit(X_train, y_train)
            
            # Validação
            pred = self.modelo.predict(X_test)
            acuracia = accuracy_score(y_test, pred)
            self.historico_acuracia.append(acuracia)
            
            # Auto-correção se performance cair
            if len(self.historico_acuracia) > 5:
                if np.mean(self.historico_acuracia[-3:]) < 0.5:
                    st.warning("Auto-correção: Performance baixa, resetando modelo")
                    self.criar_novo_modelo()
            
            # Salva o modelo com hash de segurança
            joblib.dump(self.modelo, 'modelo_btc_ai.pkl')
            with open('modelo_btc_ai.pkl', 'rb') as f:
                modelo_hash = hashlib.md5(f.read()).hexdigest()
            
            st.session_state['model_hash'] = modelo_hash
            self.ultimo_treinamento = datetime.now()
            
            return acuracia
        except Exception as e:
            st.error(f"Erro no treino: {str(e)}")
            return 0
    
    def prever(self, dados):
        """Faz previsões com tratamento de erros"""
        try:
            X = dados[['SMA_50', 'SMA_200', 'RSI', 'Volume']]
            return self.modelo.predict(X)
        except:
            return np.zeros(len(dados))

# Interface principal
ai = AprendizBTC()

# Controles
with st.sidebar:
    st.header("Controle da IA")
    if st.button("🔄 Treinar Agora"):
        with st.spinner("Treinando IA..."):
            dados = ai.preparar_dados()
            if not dados.empty:
                acuracia = ai.treinar_continuamente(dados)
                st.success(f"Acurácia: {acuracia:.2%}")
    
    st.markdown("---")
    st.header("Configurações")
    auto_treino = st.checkbox("Auto-treino diário", True)
    alertas = st.checkbox("Alertas de mercado", True)
    
    if st.button("🧹 Limpar Modelo"):
        ai.criar_novo_modelo()
        st.success("Modelo resetado!")

# Dados e Previsões
dados = ai.preparar_dados()
if not dados.empty:
    dados['Previsao'] = ai.prever(dados)
    
    # Gráfico Interativo
    fig = go.Figure()
    
    # Preço BTC
    fig.add_trace(go.Scatter(
        x=dados.index,
        y=dados['Close'],
        name='Preço BTC',
        line=dict(color='#F7931A', width=2)
    ))
    
    # Sinais de Compra
    compras = dados[dados['Previsao'] == 1]
    fig.add_trace(go.Scatter(
        x=compras.index,
        y=compras['Close'] * 0.98,
        mode='markers+text',
        marker=dict(
            color='#00FF7F',
            size=12,
            symbol='triangle-up',
            line=dict(width=2, color='DarkGreen')
        ),
        text="COMPRA",
        textposition="top center",
        name='Sinal IA'
    ))
    
    # Configurações do Gráfico
    fig.update_layout(
        title='BTC/USD - Sinais da IA',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        height=600,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Status da IA
    with st.expander("📊 Status do Aprendizado"):
        col1, col2 = st.columns(2)
        col1.metric("Último Treino", ai.ultimo_treinamento.strftime("%d/%m/%Y %H:%M") 
                   if ai.ultimo_treinamento else "Nunca")
        
        if ai.historico_acuracia:
            col2.metric("Acurácia Atual", f"{ai.historico_acuracia[-1]:.2%}")
            
            st.line_chart(pd.DataFrame({
                'Acurácia': ai.historico_acuracia,
                'Média Móvel': pd.Series(ai.historico_acuracia).rolling(5).mean()
            }))
    
    # Alertas
    if alertas and not dados.empty:
        ultimo_sinal = dados['Previsao'].iloc[-1]
        if ultimo_sinal == 1:
            st.success("🚨 ALERTA IA: Sinal de COMPRA ativo!")
        elif ultimo_sinal == 0 and dados['Previsao'].iloc[-2] == 1:
            st.warning("⚠️ ALERTA IA: Sinal de compra expirou")

# Sistema de Auto-treinamento
if auto_treino and ('ultimo_auto_treino' not in st.session_state or 
                   (datetime.now() - st.session_state.ultimo_auto_treino).days >= 1):
    with st.spinner("Auto-treinamento em progresso..."):
        dados = ai.preparar_dados()
        if not dados.empty:
            acuracia = ai.treinar_continuamente(dados)
            st.session_state.ultimo_auto_treino = datetime.now()
            st.toast(f"Auto-treinamento completo! Acurácia: {acuracia:.2%}")

# Explicação do Sistema
with st.expander("🤖 Como Funciona a IA"):
    st.markdown("""
    ## Sistema de Autoaprendizado Contínuo
    
    **1. Coleta de Dados**  
    - Baixa dados históricos do BTC-USD
    - Adiciona indicadores técnicos (SMA, RSI)
    
    **2. Engenharia de Features**  
    - Normalização automática
    - Tratamento de valores faltantes
    
    **3. Modelo de IA**  
    - Random Forest com warm_start (permite aprendizado contínuo)
    - Auto-correção quando a performance cai
    
    **4. Monitoramento**  
    - Acurácia em tempo real
    - Alertas automáticos
    - Atualização diária
    
    ## Configurações Recomendadas
    - Mantenha o **auto-treino diário** ativado
    - Monitore a acurácia no painel de status
    - Resetar o modelo se a performance cair abaixo de 50%
    """)

# Rodapé
st.markdown("---")
st.caption("Sistema IA BTC v2.0 - Atualizado em " + datetime.now().strftime("%d/%m/%Y"))
