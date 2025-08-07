import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configuração do app
st.set_page_config(layout="wide")
st.title("🚀 BTC AI Trader - Sistema Autoaprendiz")

class BitcoinAI:
    def __init__(self):
        self.model = None
        self.data = None
        self.initialize_model()
        
    def initialize_model(self):
        """Inicializa ou carrega o modelo com verificação de integridade"""
        model_path = 'btc_ai_model.pkl'
        try:
            if os.path.exists(model_path):
                # Verificação adicional de integridade
                if os.path.getsize(model_path) > 0:
                    self.model = joblib.load(model_path)
                    st.sidebar.success("Modelo carregado com sucesso!")
                    return
            
            # Se falhar ao carregar, cria novo modelo
            self.create_new_model()
            
        except Exception as e:
            st.sidebar.warning(f"Resetando modelo: {str(e)}")
            self.create_new_model()
    
    def create_new_model(self):
        """Cria um novo modelo com parâmetros otimizados"""
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=6,
            random_state=42,
            warm_start=True,
            n_jobs=-1
        )
        st.sidebar.info("Novo modelo criado!")
        self.save_model()
    
    def save_model(self):
        """Salva o modelo com verificação de integridade"""
        try:
            temp_path = 'temp_model.pkl'
            joblib.dump(self.model, temp_path)
            
            # Verifica se o arquivo foi salvo corretamente
            if os.path.getsize(temp_path) > 0:
                os.replace(temp_path, 'btc_ai_model.pkl')
                return True
            return False
        except:
            return False
    
    def load_data(self):
        """Carrega e processa os dados do BTC"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3)
        
        try:
            df = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError("Dados vazios")
                
            # Feature engineering
            df['Returns'] = df['Close'].pct_change()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['Volatility'] = df['Returns'].rolling(7).std()
            
            # Target (1 se o preço subir nos próximos 3 dias)
            df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
            
            self.data = df.dropna()
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def calculate_rsi(self, series, period=14):
        """Calcula o RSI de forma robusta"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
        return rsi.fillna(50)  # Valor neutro para casos indeterminados
    
    def train_model(self):
        """Treina o modelo com validação temporal"""
        if self.data is None:
            st.warning("Dados não carregados!")
            return 0
            
        try:
            features = ['SMA_50', 'SMA_200', 'RSI', 'Volatility', 'Returns']
            X = self.data[features]
            y = self.data['Target']
            
            # Validação cruzada temporal
            tscv = TimeSeriesSplit(n_splits=5)
            accuracies = []
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                self.model.fit(X_train, y_train)
                accuracy = self.model.score(X_test, y_test)
                accuracies.append(accuracy)
            
            mean_accuracy = np.mean(accuracies)
            self.save_model()
            
            return mean_accuracy
            
        except Exception as e:
            st.error(f"Erro no treinamento: {str(e)}")
            return 0
    
    def predict_signals(self):
        """Gera sinais de trading"""
        if self.data is None:
            return None
            
        features = ['SMA_50', 'SMA_200', 'RSI', 'Volatility', 'Returns']
        X = self.data[features]
        
        try:
            return self.model.predict(X)
        except:
            return np.zeros(len(X))

# Interface principal
ai = BitcoinAI()

# Sidebar
with st.sidebar:
    st.header("Controle da IA")
    
    if st.button("🔄 Carregar Dados e Treinar"):
        with st.spinner("Processando..."):
            if ai.load_data():
                accuracy = ai.train_model()
                st.success(f"Treinamento completo! Acurácia: {accuracy:.2%}")
    
    st.markdown("---")
    st.header("Configurações")
    auto_update = st.checkbox("Atualização automática", True)
    show_details = st.checkbox("Mostrar detalhes técnicos", False)
    
    if st.button("🆕 Criar Novo Modelo"):
        ai.create_new_model()
        st.success("Modelo reinicializado!")

# Conteúdo principal
if ai.load_data():
    predictions = ai.predict_signals()
    ai.data['Signal'] = predictions
    
    # Gráfico interativo
    fig = go.Figure()
    
    # Preço do BTC
    fig.add_trace(go.Scatter(
        x=ai.data.index,
        y=ai.data['Close'],
        name='Preço BTC',
        line=dict(color='#F7931A', width=2),
        hovertemplate="<b>Preço:</b> %{y:.2f} USD<extra></extra>"
    ))
    
    # Sinais de Compra
    buy_signals = ai.data[ai.data['Signal'] == 1]
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'] * 0.98,
        mode='markers+text',
        marker=dict(
            color='#00FF7F',
            size=12,
            symbol='triangle-up',
            line=dict(width=2, color='DarkGreen')
        ),
        text="COMPRA",
        textposition="top center",
        name='Sinal de Compra',
        hovertemplate="<b>Sinal de Compra</b><br>%{y:.2f} USD<extra></extra>"
    ))
    
    # Médias Móveis
    fig.add_trace(go.Scatter(
        x=ai.data.index,
        y=ai.data['SMA_50'],
        name='Média 50 Dias',
        line=dict(color='#1E90FF', width=1),
        visible='legendonly'
    ))
    
    fig.add_trace(go.Scatter(
        x=ai.data.index,
        y=ai.data['SMA_200'],
        name='Média 200 Dias',
        line=dict(color='#FF6347', width=1),
        visible='legendonly'
    ))
    
    # Layout
    fig.update_layout(
        title='BTC/USD - Sinais de Trading Inteligente',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        height=700,
        template='plotly_dark',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes técnicos
    if show_details:
        with st.expander("📊 Métricas Detalhadas"):
            st.write("**Últimos Sinais:**")
            st.dataframe(ai.data[['Close', 'SMA_50', 'RSI', 'Signal']].tail(10))
            
            st.write("**Distribuição de Sinais:**")
            st.bar_chart(ai.data['Signal'].value_counts())
            
            if hasattr(ai.model, 'feature_importances_'):
                st.write("**Importância das Features:**")
                features = ['SMA_50', 'SMA_200', 'RSI', 'Volatility', 'Returns']
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': ai.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance.set_index('Feature'))

# Sistema de auto-atualização
if auto_update and ('last_update' not in st.session_state or 
                   (datetime.now() - st.session_state.last_update).days >= 1):
    with st.spinner("Atualizando automaticamente..."):
        if ai.load_data():
            accuracy = ai.train_model()
            st.session_state.last_update = datetime.now()
            st.toast(f"Modelo atualizado! Acurácia: {accuracy:.2%}")

# Documentação
with st.expander("📚 Guia do Usuário"):
    st.markdown("""
    ## Como Funciona o Sistema
    
    1. **Coleta de Dados**: Obtém dados históricos do Bitcoin
    2. **Análise Técnica**: Calcula indicadores como RSI e médias móveis
    3. **Modelo de IA**: Random Forest que aprende padrões de mercado
    4. **Sinais**: Gera recomendações de compra baseadas na análise
    
    ## Configuração Recomendada
    - Mantenha a **atualização automática** ativada
    - Monitore a acurácia do modelo
    - Use em conjunto com outros indicadores
    
    ## Dicas
    - Sinais são mais confiáveis em tendências claras
    - Combine com análise fundamentalista
    - Sempre use gerenciamento de risco
    """)

st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
