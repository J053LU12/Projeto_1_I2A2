import streamlit as st
import pandas as pd
import altair as alt
import json
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

#--------------- Configuração da Página ---------------
st.set_page_config(
    page_title="TributAI",
    page_icon="assets/logo1.png",
    layout="wide"
)

#--------------- Sessão de Histórico de Chat ---------------
if "messages" not in st.session_state:
    st.session_state.messages = []

#--------------- Cabeçalho com Logo e Título ---------------
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open("assets/logo2.png")
        st.image(logo, width=120)
    except FileNotFoundError:
        st.warning("Logo não encontrada em 'assets/logo2.png'.")
with col2:
    st.markdown("""
        <div style="display:flex; align-items:center; height:100%;">
            <h1 style="margin:0;">.:Seu Agente Fiscal:.</h1>
        </div>
    """, unsafe_allow_html=True)

#--------------- Sidebar: Upload e Seleção de Arquivo ---------------
st.sidebar.header("Configuração de Dados")
uploaded_files = st.sidebar.file_uploader(
    "Upload de arquivo(s) CSV",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    @st.cache
    def load_csv(file) -> pd.DataFrame:
        file.seek(0)
        return pd.read_csv(file)

    # Carrega todos e cria dicionário nome→DataFrame
    dfs = {f.name: load_csv(f) for f in uploaded_files}
    arquivos = list(dfs.keys())
    selected_file = st.sidebar.selectbox("Selecione o arquivo para análise", arquivos)
    df = dfs[selected_file]

    # Mostra amostra e estatísticas
    st.header(f"Visão Geral: {selected_file}")
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Estatísticas Descritivas")
    st.write(df.describe(include="all"))

    # Editor de dados inline (Streamlit ≥1.21)
    st.subheader("Editor de Dados")
    df_edit = st.data_editor(df, num_rows="dynamic")
    df_for_analysis = df_edit.copy()

    #--------------- Sidebar: Filtros Dinâmicos ---------------
    st.sidebar.subheader("Filtros Dinâmicos")
    for col, dtype in df_for_analysis.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            mn, mx = float(df_for_analysis[col].min()), float(df_for_analysis[col].max())
            sel = st.sidebar.slider(col, mn, mx, (mn, mx))
            df_for_analysis = df_for_analysis[df_for_analysis[col].between(*sel)]
        else:
            opts = st.sidebar.multiselect(col, df_for_analysis[col].unique(), df_for_analysis[col].unique())
            df_for_analysis = df_for_analysis[df_for_analysis[col].isin(opts)]

    st.subheader("Dados Após Filtros")
    st.dataframe(df_for_analysis, use_container_width=True)

    # Botão para limpar histórico
    if st.sidebar.button("🔄 Reiniciar Chat"):
        st.session_state.messages = []

    # Exibe histórico de chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    #--------------- Configuração da LLM e do Agente ---------------
    api_key = st.secrets["GOOGLE_API_KEY"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    
    INSTRUCTIONS = """
    - Você é um assistente especialista em análise de dados chamado TributAI.
    - Sempre responda em Português do Brasil (pt-BR).
    - Pense passo a passo e forneça respostas claras e amigáveis.
    - Se fizer cálculos, apresente resultados de forma estruturada.
    - Quando relevante, sugira visualizações e retorne em JSON neste formato:
      {
        "chart":"line"|"bar"|"scatter",
        "x":[...],
        "y":[...],
        "x_label":"...",
        "y_label":"..."
      }
    - Se não souber responder, diga: "Desculpe, não consegui processar seu pedido."
    """
    agent = create_pandas_dataframe_agent(
        llm,
        df_for_analysis,
        verbose=True,
        allow_dangerous_code=False,
        include_df_in_prompt=False,
        prefix=INSTRUCTIONS
    )

    #--------------- Chat Input  ---------------
    if prompt := st.chat_input(f"Pergunte sobre '{selected_file}'..."):
        # Usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistente
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    response = agent.run(prompt)
                except Exception:
                    response = "Desculpe, algo deu errado ao processar sua requisição."

            # Tenta extrair JSON de visualização
            try:
                payload = json.loads(response)
                if isinstance(payload, dict) and "chart" in payload:
                    df_chart = pd.DataFrame({"x": payload["x"], "y": payload["y"]})
                    chart_type = payload["chart"]
                    if chart_type == "line":
                        chart = alt.Chart(df_chart).mark_line().encode(
                            x=alt.X("x:T", title=payload.get("x_label", "")),
                            y=alt.Y("y:Q", title=payload.get("y_label", "")),
                            tooltip=["x", "y"]
                        )
                    elif chart_type == "bar":
                        chart = alt.Chart(df_chart).mark_bar().encode(
                            x=alt.X("x:T", title=payload.get("x_label", "")),
                            y=alt.Y("y:Q", title=payload.get("y_label", "")),
                            tooltip=["x", "y"]
                        )
                    elif chart_type == "scatter":
                        chart = alt.Chart(df_chart).mark_circle(size=60).encode(
                            x=alt.X("x:T", title=payload.get("x_label", "")),
                            y=alt.Y("y:Q", title=payload.get("y_label", "")),
                            tooltip=["x", "y"]
                        )
                    else:
                        chart = None

                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.markdown(response)
                else:
                    st.markdown(response)
            except json.JSONDecodeError:
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("📂 Faça upload de pelo menos um arquivo CSV para começar a análise.")