import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuração do app
st.set_page_config(layout="wide")
st.title("📊 Indicador Markov-Queue BTC - Versão Premium Completa")

# Sidebar com parâmetros principais e avançados
with st.sidebar:
    st.header("Configurações Principais")
    data_inicio = st.date_input("Data inicial", datetime(2023, 1, 1))
    data_fim = st.date_input("Data final", datetime.today())
    periodo_rsi = st.slider("Período do RSI", 2, 50, 14)
    periodo_sma = st.slider("Período da SMA", 50, 500, 200)
    
    st.markdown("---")
    st.header("Filtros Avançados")
    usar_volume = st.checkbox("Considerar volume", True)
    usar_macd = st.checkbox("Mostrar MACD", True)
    threshold_rsi = st.slider("Limiar RSI para confirmação", 50, 70, 65)
    usar_breakout = st.checkbox("Exigir rompimento das BB", False)
    theme = st.selectbox("Tema do Gráfico", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    
    st.markdown("---")
    st.header("Configurações de Alerta")
    alerta_bull = st.checkbox("Alertar Bull Market", True)
    alerta_bear = st.checkbox("Alertar Bear Market", True)

# Função para baixar dados - VERSÃO CORRIGIDA
@st.cache_data
def carregar_dados():
    try:
        # Baixar dados e converter para estrutura unidimensional
        dados = yf.download("BTC-USD", 
                          start=data_inicio, 
                          end=data_fim + pd.Timedelta(days=1),
                          progress=False)
        
        if dados.empty:
            return pd.DataFrame()
            
        # CORREÇÃO DO ERRO: Converter para Series e garantir estrutura 1D
        if isinstance(dados, pd.DataFrame):
            close_data = dados['Close'].squeeze()  # Converte para Series 1D
            volume_data = dados['Volume'].squeeze() if 'Volume' in dados else None
            
            # Criar DataFrame garantindo estrutura correta
            df = pd.DataFrame({
                'Close': close_data.values if hasattr(close_data, 'values') else close_data
            }, index=dados.index)
            
            if volume_data is not None:
                df['Volume'] = volume_data.values if hasattr(volume_data, 'values') else volume_data
                
            return df
            
        return pd.DataFrame()
    except Exception as erro:
        st.error(f"Erro ao baixar dados: {str(erro)}")
        return pd.DataFrame()

# Carregar dados
dados_btc = carregar_dados()

if not dados_btc.empty:
    # Cálculos técnicos básicos
    dados_btc['SMA'] = dados_btc['Close'].rolling(periodo_sma).mean()
    
    # Cálculo do RSI
    delta = dados_btc['Close'].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo_rsi).mean()
    media_perda = perda.rolling(periodo_rsi).mean().replace(0, np.nan)
    dados_btc['RSI'] = 100 - (100 / (1 + (media_ganho / media_perda)))
    
    # Bollinger Bands
    rolling_mean = dados_btc['Close'].rolling(20).mean()
    rolling_std = dados_btc['Close'].rolling(20).std()
    dados_btc['BB_Upper'] = rolling_mean + 2 * rolling_std
    dados_btc['BB_Lower'] = rolling_mean - 2 * rolling_std
    dados_btc['BB_Width'] = ((dados_btc['BB_Upper'] - dados_btc['BB_Lower']) / rolling_mean) * 100
    
    # Indicadores adicionais (MELHORIAS MANTIDAS)
    if usar_macd:
        ema12 = dados_btc['Close'].ewm(span=12, adjust=False).mean()
        ema26 = dados_btc['Close'].ewm(span=26, adjust=False).mean()
        dados_btc['MACD'] = ema12 - ema26
        dados_btc['Sinal'] = dados_btc['MACD'].ewm(span=9, adjust=False).mean()
    
    if usar_volume and 'Volume' in dados_btc:
        dados_btc['Volume_MA'] = dados_btc['Volume'].rolling(20).mean()
    
    # Remover NaN
    dados_btc = dados_btc.dropna()
    
    # Definir estados com condições aprimoradas (MELHORIAS MANTIDAS)
    condicoes = [
        (dados_btc['Close'] > dados_btc['SMA']) & 
        (dados_btc['RSI'] > threshold_rsi) &
        (dados_btc['Close'] > dados_btc['BB_Upper'] if usar_breakout else True),
        
        (dados_btc['Close'] < dados_btc['SMA']) & 
        (dados_btc['RSI'] < (100 - threshold_rsi)) &
        (dados_btc['Close'] < dados_btc['BB_Lower'] if usar_breakout else True),
        
        (dados_btc['BB_Width'] < 0.5)
    ]
    dados_btc['Estado'] = np.select(condicoes, ['Bull', 'Bear', 'Consolid'], 'Neutro')

    # Gráfico principal (MELHORIAS MANTIDAS)
    fig = go.Figure()
    
    # Preço e SMA
    fig.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['Close'],
        name='Preço BTC',
        line=dict(color='gold')
    ))
    fig.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['SMA'],
        name=f'SMA {periodo_sma}',
        line=dict(color='orange', dash='dot')
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['BB_Upper'],
        name='BB Superior',
        line=dict(color='rgba(70, 130, 180, 0.5)')
    ))
    fig.add_trace(go.Scatter(
        x=dados_btc.index,
        y=dados_btc['BB_Lower'],
        name='BB Inferior',
        line=dict(color='rgba(70, 130, 180, 0.5)')
    ))
    
    # MACD se ativado (MELHORIA MANTIDA)
    if usar_macd:
        fig.add_trace(go.Scatter(
            x=dados_btc.index,
            y=dados_btc['MACD'],
            name='MACD',
            line=dict(color='blue'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=dados_btc.index,
            y=dados_btc['Sinal'],
            name='Sinal MACD',
            line=dict(color='red'),
            yaxis='y2'
        ))
    
    # Volume se ativado (MELHORIA MANTIDA)
    if usar_volume and 'Volume' in dados_btc:
        fig.add_trace(go.Bar(
            x=dados_btc.index,
            y=dados_btc['Volume'],
            name='Volume',
            marker_color='rgba(100, 100, 100, 0.3)',
            yaxis='y3'
        ))
    
    # Áreas coloridas para estados
    cores_estado = {
        'Bull': 'rgba(46,139,87,0.2)',
        'Bear': 'rgba(178,34,34,0.2)',
        'Consolid': 'rgba(30,144,255,0.2)'
    }
    
    for estado, cor in cores_estado.items():
        mask = dados_btc['Estado'] == estado
        changes = mask.astype(int).diff()
        starts = dados_btc.index[changes == 1]
        ends = dados_btc.index[changes == -1]
        
        if len(starts) > 0:
            if len(starts) > len(ends):
                ends = list(ends) + [dados_btc.index[-1]]
            
            for start, end in zip(starts, ends):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=cor, layer="below",
                    line_width=0
                )

    # Layout do gráfico (MELHORIA MANTIDA)
    fig.update_layout(
        title=f'BTC/USD - Markov-Queue Indicator (Último: {dados_btc["Close"].iloc[-1]:.2f} USD)',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        showlegend=True,
        height=700,  # Aumentado para acomodar mais indicadores
        template=theme,
        yaxis2=dict(
            title='MACD',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        yaxis3=dict(
            title='Volume',
            overlaying='y',
            side='left',
            anchor='free',
            position=0.05,
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Alertas de mudança de estado (MELHORIA MANTIDA)
    ultimo_estado = dados_btc['Estado'].iloc[-1]
    penultimo_estado = dados_btc['Estado'].iloc[-2] if len(dados_btc) > 1 else None

    if ultimo_estado != penultimo_estado:
        if alerta_bull and ultimo_estado == 'Bull':
            st.success("🚨 NOVO SINAL: Bull Market detectado!")
        elif alerta_bear and ultimo_estado == 'Bear':
            st.error("🚨 NOVO SINAL: Bear Market detectado!")
    
    # Seção de análise de desempenho (MELHORIA MANTIDA)
    st.subheader("📈 Análise de Desempenho")
    
    if st.checkbox("Mostrar análise de estratégia"):
        dados_btc['Retorno'] = dados_btc['Close'].pct_change()
        dados_btc['Estrategia'] = 0
        dados_btc.loc[dados_btc['Estado'] == 'Bull', 'Estrategia'] = 1
        dados_btc.loc[dados_btc['Estado'] == 'Bear', 'Estrategia'] = -1
        dados_btc['Retorno_Strategy'] = dados_btc['Estrategia'].shift(1) * dados_btc['Retorno']
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=dados_btc.index,
            y=(1 + dados_btc['Retorno']).cumprod(),
            name='Buy & Hold'
        ))
        fig_perf.add_trace(go.Scatter(
            x=dados_btc.index,
            y=(1 + dados_btc['Retorno_Strategy']).cumprod(),
            name='Estratégia Markov-Queue'
        ))
        fig_perf.update_layout(
            title='Desempenho Comparativo',
            yaxis_title='Retorno Acumulado',
            template=theme
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Últimos sinais com formatação condicional (MELHORIA MANTIDA)
    st.subheader("📊 Últimos Sinais")
    st.dataframe(
        dados_btc.tail(10)[['Close', 'SMA', 'RSI', 'BB_Width', 'Estado']].style.format({
            'Close': '{:.2f}', 
            'SMA': '{:.2f}', 
            'RSI': '{:.1f}', 
            'BB_Width': '{:.2f}%'
        }).applymap(
            lambda x: 'background-color: rgba(46,139,87,0.3)' if x == 'Bull' else 
                     'background-color: rgba(178,34,34,0.3)' if x == 'Bear' else 
                     'background-color: rgba(30,144,255,0.3)' if x == 'Consolid' else '',
            subset=['Estado']
        ),
        use_container_width=True
    )
    
    # Exportação de dados (MELHORIA MANTIDA)
    if st.button("📤 Exportar dados para CSV"):
        csv = dados_btc.to_csv(index=True)
        st.download_button(
            label="Baixar CSV",
            data=csv,
            file_name='dados_btc_indicador.csv',
            mime='text/csv'
        )

else:
    st.warning("Não foi possível carregar os dados. Verifique sua conexão e as datas selecionadas.")

# Documentação e ajuda (MELHORIA MANTIDA)
with st.expander("📚 Documentação e Ajuda"):
    st.markdown("""
    ## 📊 Indicador Markov-Queue BTC - Guia Completo
    
    ### 🔍 Como interpretar os sinais:
    - **🟢 Bull Market**:  
      - *Condições*: Preço > SMA + RSI > limiar + (opcional: acima da BB Superior)  
      - *Estratégia*: Considerar posições longas
    
    - **🔴 Bear Market**:  
      - *Condições*: Preço < SMA + RSI < (100 - limiar) + (opcional: abaixo da BB Inferior)  
      - *Estratégia*: Considerar posições short
    
    - **🔵 Consolidação**:  
      - *Condições*: Volatilidade baixa (BB Width < 0.5%)  
      - *Estratégia*: Aguardar rompimento
    
    - **⚪ Neutro**:  
      - *Condições*: Sem sinal claro  
      - *Estratégia*: Analisar outros fatores
    
    ### ⚙️ Configurações recomendadas:
    - **SMA**: 200 períodos para tendências longas  
    - **RSI**: 14 períodos com limiar em 60/40  
    - **Bollinger Bands**: 20 períodos com 2 desvios padrão
    
    ### 📈 Indicadores Adicionais:
    - **MACD**: Mostra convergência/divergência de médias móveis  
    - **Volume**: Confirma força por trás dos movimentos de preço
    
    ### ⚠️ Observações:
    - Este indicador deve ser usado em conjunto com outras análises
    - Configure os parâmetros conforme seu estilo de trading
    - Sempre utilize stop-loss e gerencie seu risco
    """)

# Informações adicionais
st.markdown("---")
st.markdown("""
**ℹ️ Sobre este indicador**:  
O Markov-Queue BTC Indicator combina múltiplos indicadores técnicos para identificar tendências e condições de mercado.  
Desenvolvido para operações de médio/longo prazo com Bitcoin. Atualizado em {}.
""".format(datetime.now().strftime("%d/%m/%Y")))
