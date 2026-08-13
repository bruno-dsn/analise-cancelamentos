from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.analysis import (
    ROTULOS_SEGMENTOS,
    classificar_risco,
    resumir_segmento,
    simular_campanha,
    taxa_cancelamento,
)
from src.data import (
    CANAIS,
    CONTRATOS,
    PAGAMENTOS,
    PLANOS,
    carregar_dados,
    gerar_dados_sinteticos,
    montar_cenario,
)
from src.model import (
    adicionar_probabilidades,
    avaliar_limiar,
    explicar_previsao,
    prever_probabilidade,
    treinar_e_avaliar,
)


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "data" / "clientes_assinatura_sinteticos.csv"

CORES = {
    "Baixo": "#22C55E",
    "Atenção": "#F2C94C",
    "Alto": "#F97316",
    "Crítico": "#EF4444",
    "Ativo": "#22C55E",
    "Cancelou": "#EF4444",
}

PERFIS = {
    "Cliente engajado": {
        "sim_plano": "Plus",
        "sim_contrato": "Anual",
        "sim_pagamento": "Débito automático",
        "sim_canal": "Indicação",
        "sim_tempo": 28,
        "sim_receita": 79.90,
        "sim_desconto": 10,
        "sim_horas": 7.0,
        "sim_queda": 8.0,
        "sim_dias": 2,
        "sim_satisfacao": 9,
        "sim_chamados": 0,
        "sim_falhas": 0,
        "sim_reajuste": "Não",
    },
    "Cliente em atenção": {
        "sim_plano": "Essencial",
        "sim_contrato": "Mensal",
        "sim_pagamento": "Cartão",
        "sim_canal": "Mídia paga",
        "sim_tempo": 10,
        "sim_receita": 49.90,
        "sim_desconto": 5,
        "sim_horas": 2.5,
        "sim_queda": 42.0,
        "sim_dias": 15,
        "sim_satisfacao": 6,
        "sim_chamados": 2,
        "sim_falhas": 1,
        "sim_reajuste": "Sim",
    },
    "Cliente crítico": {
        "sim_plano": "Premium",
        "sim_contrato": "Mensal",
        "sim_pagamento": "Boleto",
        "sim_canal": "Mídia paga",
        "sim_tempo": 5,
        "sim_receita": 119.90,
        "sim_desconto": 0,
        "sim_horas": 0.8,
        "sim_queda": 78.0,
        "sim_dias": 34,
        "sim_satisfacao": 3,
        "sim_chamados": 5,
        "sim_falhas": 3,
        "sim_reajuste": "Sim",
    },
}


@st.cache_data
def obter_dados() -> pd.DataFrame:
    if ARQUIVO_DADOS.exists():
        return carregar_dados(ARQUIVO_DADOS)
    return gerar_dados_sinteticos()


@st.cache_resource
def obter_resultado():
    return treinar_e_avaliar(obter_dados())


@st.cache_data
def obter_dados_com_risco(_modelo) -> pd.DataFrame:
    dados = adicionar_probabilidades(obter_dados(), _modelo)
    dados["faixa_risco"] = classificar_risco(dados["probabilidade_cancelamento"])
    return dados


def moeda(valor: float) -> str:
    texto = f"{valor:,.2f}"
    return f"R$ {texto.replace(',', 'X').replace('.', ',').replace('X', '.')}"


def percentual(valor: float, casas: int = 1) -> str:
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def faixa_de_risco(probabilidade: float) -> str:
    if probabilidade < 0.15:
        return "Baixo"
    if probabilidade < 0.30:
        return "Atenção"
    if probabilidade < 0.50:
        return "Alto"
    return "Crítico"


def aplicar_perfil() -> None:
    nome = st.session_state["perfil_simulador"]
    if nome in PERFIS:
        for chave, valor in PERFIS[nome].items():
            st.session_state[chave] = valor


def inicializar_simulador() -> None:
    st.session_state.setdefault("perfil_simulador", "Cliente em atenção")
    for chave, valor in PERFIS["Cliente em atenção"].items():
        st.session_state.setdefault(chave, valor)


def medidor_risco(probabilidade: float) -> go.Figure:
    faixa = faixa_de_risco(probabilidade)
    figura = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probabilidade * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Probabilidade estimada de cancelamento em 60 dias"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": CORES[faixa]},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 15], "color": "#163B2B"},
                    {"range": [15, 30], "color": "#4B421C"},
                    {"range": [30, 50], "color": "#512B17"},
                    {"range": [50, 100], "color": "#4A1F24"},
                ],
            },
        )
    )
    figura.update_layout(height=310, margin={"l": 30, "r": 30, "t": 55, "b": 5})
    return figura


st.set_page_config(
    page_title="Laboratório de Retenção de Clientes",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1260px; padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {letter-spacing: -0.025em;}
        [data-testid="stMetricValue"] {font-weight: 760;}
        .hero {
            padding: 1.55rem 1.75rem;
            margin-bottom: 1rem;
            border: 1px solid #273449;
            border-radius: 18px;
            background: linear-gradient(120deg, #111827 0%, #172554 55%, #164E63 100%);
        }
        .hero h1 {margin: 0 0 .45rem 0; font-size: 2.35rem; color: #F8FAFC;}
        .hero p {margin: 0; color: #D6E0EF; font-size: 1.04rem; max-width: 900px;}
        .tag {
            display: inline-block;
            margin-bottom: .65rem;
            padding: .22rem .58rem;
            border-radius: 999px;
            background: #173F4C;
            color: #67E8F9;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .insight {
            padding: 1rem 1.1rem;
            border-left: 4px solid #38BDF8;
            border-radius: 8px;
            background: #111C2E;
            color: #D9E5F3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

dados = obter_dados()
resultado = obter_resultado()
dados_risco = obter_dados_com_risco(resultado.modelo)
inicializar_simulador()

with st.sidebar:
    st.title("Filtros da carteira")
    planos = st.multiselect("Planos", PLANOS, default=PLANOS)
    contratos = st.multiselect("Contratos", CONTRATOS, default=CONTRATOS)
    canais = st.multiselect("Canais de aquisição", CANAIS, default=CANAIS)
    st.caption(
        "Os filtros alteram a visão executiva e os segmentos. "
        "O simulador individual e a campanha usam seus próprios controles."
    )
    st.divider()
    st.subheader("Sobre a base")
    st.write("12.000 clientes fictícios de um serviço brasileiro por assinatura.")
    st.write("Alvo: cancelamento observado nos 60 dias seguintes ao retrato do cliente.")
    st.caption("Nenhum registro corresponde a uma pessoa ou empresa real.")

filtro = dados_risco[
    dados_risco["plano"].isin(planos)
    & dados_risco["contrato"].isin(contratos)
    & dados_risco["canal_aquisicao"].isin(canais)
].copy()

if filtro.empty:
    st.warning("Nenhum cliente atende aos filtros selecionados.")
    st.stop()

st.markdown(
    """
    <section class="hero">
        <div class="tag">Análise de churn e Machine Learning</div>
        <h1>Laboratório de Retenção de Clientes</h1>
        <p>Entenda onde o cancelamento se concentra, simule o risco de um perfil e estime o retorno de uma campanha de retenção.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Projeto educacional com dados sintéticos. As previsões organizam investigação e priorização, "
    "mas não provam que uma ação causará retenção."
)

aba_visao, aba_segmentos, aba_simulador, aba_campanha, aba_modelo = st.tabs(
    [
        "Visão executiva",
        "Segmentos",
        "Simulador de risco",
        "Campanha de retenção",
        "Modelo e dados",
    ]
)

with aba_visao:
    receita_total = filtro["receita_mensal"].sum()
    alto_risco = filtro["probabilidade_cancelamento"].ge(0.30)
    receita_alto_risco = filtro.loc[alto_risco, "receita_mensal"].sum()

    metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)
    metrica_1.metric("Clientes analisados", f"{len(filtro):,}".replace(",", "."))
    metrica_2.metric("Cancelamento em 60 dias", percentual(taxa_cancelamento(filtro)))
    metrica_3.metric("Receita mensal da carteira", moeda(receita_total))
    metrica_4.metric("Receita mensal em alto risco", moeda(receita_alto_risco))

    resumo_contrato = resumir_segmento(filtro, "contrato")
    resumo_plano = resumir_segmento(filtro, "plano")
    coluna_1, coluna_2 = st.columns(2, gap="large")

    with coluna_1:
        grafico_contrato = px.bar(
            resumo_contrato,
            x="contrato",
            y="taxa_cancelamento",
            color="contrato",
            text_auto=".1%",
            title="Cancelamento por tipo de contrato",
            labels={"contrato": "Contrato", "taxa_cancelamento": "Taxa de cancelamento"},
        )
        grafico_contrato.update_yaxes(tickformat=".0%", rangemode="tozero")
        grafico_contrato.update_layout(showlegend=False)
        st.plotly_chart(grafico_contrato, use_container_width=True)

    with coluna_2:
        grafico_plano = px.bar(
            resumo_plano,
            x="plano",
            y="taxa_cancelamento",
            color="plano",
            text_auto=".1%",
            title="Cancelamento por plano",
            labels={"plano": "Plano", "taxa_cancelamento": "Taxa de cancelamento"},
        )
        grafico_plano.update_yaxes(tickformat=".0%", rangemode="tozero")
        grafico_plano.update_layout(showlegend=False)
        st.plotly_chart(grafico_plano, use_container_width=True)

    distribuicao_risco = (
        filtro["faixa_risco"]
        .value_counts(sort=False)
        .rename_axis("Faixa de risco")
        .reset_index(name="Clientes")
    )
    grafico_risco = px.bar(
        distribuicao_risco,
        x="Faixa de risco",
        y="Clientes",
        color="Faixa de risco",
        color_discrete_map=CORES,
        text_auto=True,
        title="Distribuição da carteira por faixa de risco",
    )
    grafico_risco.update_layout(showlegend=False)
    st.plotly_chart(grafico_risco, use_container_width=True)

    maior_segmento = resumo_contrato.iloc[0]
    st.markdown(
        f"""
        <div class="insight">
            <strong>Leitura principal:</strong> o contrato {maior_segmento['contrato'].lower()} apresenta taxa observada de {percentual(maior_segmento['taxa_cancelamento'])}. Esse resultado indica onde investigar primeiro, mas uma campanha precisa de teste com grupo de controle para medir efeito real.
        </div>
        """,
        unsafe_allow_html=True,
    )

with aba_segmentos:
    st.header("Onde o cancelamento se concentra?")
    st.write(
        "Compare volume, taxa de cancelamento, receita mensal e lift. "
        "Lift acima de 1 indica taxa maior que a média do recorte atual."
    )
    nomes_colunas = {rotulo: coluna for coluna, rotulo in ROTULOS_SEGMENTOS.items()}
    escolha = st.selectbox("Dimensão de análise", list(nomes_colunas))
    coluna_segmento = nomes_colunas[escolha]
    resumo = resumir_segmento(filtro, coluna_segmento)

    grafico_segmento = px.scatter(
        resumo,
        x="clientes",
        y="taxa_cancelamento",
        size="receita_mensal",
        color=coluna_segmento,
        hover_data={"lift": ":.2f", "receita_mensal": ":.2f"},
        labels={
            "clientes": "Quantidade de clientes",
            "taxa_cancelamento": "Taxa de cancelamento",
            coluna_segmento: escolha,
        },
        title=f"Volume, taxa e receita por {escolha.lower()}",
    )
    grafico_segmento.update_yaxes(tickformat=".0%")
    st.plotly_chart(grafico_segmento, use_container_width=True)

    resumo_exibicao = resumo.rename(
        columns={
            coluna_segmento: escolha,
            "clientes": "Clientes",
            "cancelamentos": "Cancelamentos",
            "taxa_cancelamento": "Taxa de cancelamento",
            "receita_mensal": "Receita mensal",
            "lift": "Lift",
        }
    ).copy()
    resumo_exibicao["Taxa de cancelamento"] = resumo_exibicao[
        "Taxa de cancelamento"
    ].map(percentual)
    resumo_exibicao["Receita mensal"] = resumo_exibicao["Receita mensal"].map(moeda)
    resumo_exibicao["Lift"] = resumo_exibicao["Lift"].map(lambda valor: f"{valor:.2f}x")
    st.dataframe(resumo_exibicao, hide_index=True, use_container_width=True)

    matriz = (
        filtro.pivot_table(
            index="plano",
            columns="contrato",
            values="cancelou_60d",
            aggfunc="mean",
        )
        .reindex(index=PLANOS, columns=CONTRATOS)
        .fillna(0)
    )
    calor = go.Figure(
        go.Heatmap(
            z=matriz.values * 100,
            x=matriz.columns,
            y=matriz.index,
            colorscale="RdYlGn_r",
            text=[[f"{valor:.1f}%" for valor in linha] for linha in matriz.values * 100],
            texttemplate="%{text}",
            hovertemplate="Plano: %{y}<br>Contrato: %{x}<br>Cancelamento: %{z:.1f}%<extra></extra>",
        )
    )
    calor.update_layout(title="Cancelamento por combinação de plano e contrato", height=380)
    st.plotly_chart(calor, use_container_width=True)

with aba_simulador:
    st.header("Simule um perfil de cliente")
    st.write(
        "Altere as características e veja como o modelo reage. "
        "O resultado é uma demonstração estatística, não uma decisão sobre uma pessoa real."
    )
    st.selectbox(
        "Comece por um exemplo",
        ["Cliente engajado", "Cliente em atenção", "Cliente crítico", "Personalizado"],
        key="perfil_simulador",
        on_change=aplicar_perfil,
    )

    entrada_1, entrada_2 = st.columns(2, gap="large")
    with entrada_1:
        st.selectbox("Plano atual", PLANOS, key="sim_plano")
        st.selectbox("Tipo de contrato", CONTRATOS, key="sim_contrato")
        st.selectbox("Forma de pagamento", PAGAMENTOS, key="sim_pagamento")
        st.selectbox("Canal de aquisição", CANAIS, key="sim_canal")
        st.slider("Tempo como cliente (meses)", 1, 96, key="sim_tempo")
        st.number_input(
            "Receita mensal do cliente",
            min_value=20.0,
            max_value=300.0,
            step=5.0,
            format="%.2f",
            key="sim_receita",
        )
        st.select_slider("Desconto atual", [0, 5, 10, 15, 20], format_func=lambda v: f"{v}%", key="sim_desconto")

    with entrada_2:
        st.slider("Horas de uso por semana", 0.0, 28.0, step=0.5, key="sim_horas")
        st.slider("Queda de uso nos últimos 30 dias", 0.0, 100.0, step=1.0, format="%.0f%%", key="sim_queda")
        st.slider("Dias sem acessar o serviço", 0, 60, key="sim_dias")
        st.slider(
            "Nota de satisfação, de 0 a 10",
            0,
            10,
            key="sim_satisfacao",
            help="0 significa muito insatisfeito e 10 significa muito satisfeito.",
        )
        st.slider("Chamados ao suporte nos últimos 90 dias", 0, 10, key="sim_chamados")
        st.slider("Falhas de pagamento nos últimos 90 dias", 0, 6, key="sim_falhas")
        st.radio("Houve reajuste recente?", ["Não", "Sim"], horizontal=True, key="sim_reajuste")

    cenario = montar_cenario(
        plano=st.session_state["sim_plano"],
        contrato=st.session_state["sim_contrato"],
        forma_pagamento=st.session_state["sim_pagamento"],
        canal_aquisicao=st.session_state["sim_canal"],
        tempo_cliente_meses=st.session_state["sim_tempo"],
        receita_mensal=st.session_state["sim_receita"],
        desconto_pct=st.session_state["sim_desconto"],
        horas_uso_semana=st.session_state["sim_horas"],
        queda_uso_30d_pct=st.session_state["sim_queda"],
        dias_sem_acesso=st.session_state["sim_dias"],
        nota_satisfacao=st.session_state["sim_satisfacao"],
        chamados_90d=st.session_state["sim_chamados"],
        falhas_pagamento_90d=st.session_state["sim_falhas"],
        reajuste_recente=int(st.session_state["sim_reajuste"] == "Sim"),
    )
    probabilidade = prever_probabilidade(resultado.modelo, cenario)
    faixa = faixa_de_risco(probabilidade)

    resultado_1, resultado_2 = st.columns([1, 1.15], gap="large")
    with resultado_1:
        st.plotly_chart(medidor_risco(probabilidade), use_container_width=True)
        st.markdown(
            f"""
            <div class="insight">
                <strong>Faixa {faixa.lower()}.</strong> Em 100 perfis sintéticos semelhantes, o modelo estima aproximadamente {round(probabilidade * 100)} cancelamentos nos próximos 60 dias. Isso não garante o comportamento de um cliente específico.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with resultado_2:
        st.subheader("Principais fatores deste cenário")
        explicacao = explicar_previsao(resultado.modelo, cenario).sort_values("contribuicao")
        grafico_explicacao = px.bar(
            explicacao,
            x="contribuicao",
            y="variavel",
            orientation="h",
            color="efeito",
            color_discrete_map={"Aumenta o risco": "#EF4444", "Reduz o risco": "#22C55E"},
            labels={"contribuicao": "Impacto relativo", "variavel": "", "efeito": "Efeito"},
        )
        grafico_explicacao.add_vline(x=0, line_color="#94A3B8", line_width=1)
        grafico_explicacao.update_layout(height=415, legend_title_text="")
        st.plotly_chart(grafico_explicacao, use_container_width=True)
        st.caption(
            "O gráfico explica a previsão do modelo. Ele mostra associação, não causa."
        )

with aba_campanha:
    st.header("Planeje uma campanha de retenção")
    st.write(
        "A simulação prioriza os clientes com maior risco previsto. "
        "Você escolhe as premissas e o painel estima o retorno financeiro potencial."
    )
    controle_1, controle_2 = st.columns(2, gap="large")
    with controle_1:
        capacidade = st.slider("Quantidade máxima de clientes", 100, 3_000, 800, step=100)
        custo_acao = st.number_input(
            "Custo da ação por cliente",
            min_value=0.0,
            max_value=200.0,
            value=18.0,
            step=2.0,
        )
        efetividade_pct = st.slider(
            "Cancelamentos evitados entre os clientes abordados",
            5,
            50,
            20,
            step=5,
            help="Premissa de cenário. O valor real deve ser medido com teste controlado.",
        )
    with controle_2:
        horizonte = st.slider("Horizonte de receita preservada (meses)", 3, 12, 6)
        margem_pct = st.slider(
            "Margem de contribuição da receita",
            20,
            100,
            70,
            step=5,
        )
        st.caption(
            "A margem de contribuição representa a parcela da receita disponível após custos variáveis."
        )

    campanha = simular_campanha(
        dados_com_risco=dados_risco,
        capacidade=capacidade,
        custo_por_cliente=custo_acao,
        efetividade=efetividade_pct / 100,
        horizonte_meses=horizonte,
        margem_contribuicao=margem_pct / 100,
    )
    camp_1, camp_2, camp_3, camp_4 = st.columns(4)
    camp_1.metric("Clientes abordados", f"{capacidade:,}".replace(",", "."))
    camp_2.metric("Retenções esperadas", f"{campanha['clientes_retidos']:.0f}")
    camp_3.metric("Custo da campanha", moeda(campanha["custo_total"]))
    camp_4.metric("Retorno líquido potencial", moeda(campanha["retorno_liquido"]))

    grafico_retorno = go.Figure(
        go.Waterfall(
            x=["Margem preservada", "Custo da campanha", "Retorno líquido"],
            y=[
                campanha["margem_preservada"],
                -campanha["custo_total"],
                campanha["retorno_liquido"],
            ],
            measure=["relative", "relative", "total"],
            connector={"line": {"color": "#64748B"}},
            increasing={"marker": {"color": "#22C55E"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#38BDF8"}},
        )
    )
    grafico_retorno.update_layout(
        title="Construção do retorno potencial",
        yaxis_title="Valor estimado em R$",
        height=430,
    )
    st.plotly_chart(grafico_retorno, use_container_width=True)

    if campanha["custo_total"]:
        st.write(f"ROI potencial do cenário: **{campanha['roi']:.1%}**")
    st.warning(
        "A efetividade informada é uma hipótese. Para afirmar impacto, a empresa precisaria "
        "comparar grupos equivalentes, com e sem campanha, durante o mesmo período."
    )

    selecionados = campanha["selecionados"].copy()
    selecionados["probabilidade_cancelamento"] = selecionados[
        "probabilidade_cancelamento"
    ].map(lambda valor: f"{valor:.1%}")
    st.download_button(
        "Baixar lista priorizada em CSV",
        data=selecionados.to_csv(index=False).encode("utf-8"),
        file_name="clientes_priorizados_retencao.csv",
        mime="text/csv",
    )

with aba_modelo:
    st.header("Modelo, métricas e dados")
    st.write(
        "O modelo foi treinado em 75% da base e avaliado nos 25% restantes. "
        "Altere o limiar para observar o equilíbrio entre precisão e recall."
    )
    limiar_pct = st.slider("Limiar de classificação", 10, 60, 20, step=5, format="%d%%")
    metricas, matriz = avaliar_limiar(resultado, limiar_pct / 100)

    mod_1, mod_2, mod_3, mod_4 = st.columns(4)
    mod_1.metric("ROC AUC", f"{resultado.metricas['roc_auc']:.3f}")
    mod_2.metric("Acurácia", percentual(metricas["acuracia"]))
    mod_3.metric("Precisão", percentual(metricas["precisao"]))
    mod_4.metric("Recall", percentual(metricas["recall"]))

    grafico_1, grafico_2 = st.columns(2, gap="large")
    with grafico_1:
        matriz_fig = go.Figure(
            go.Heatmap(
                z=matriz,
                x=["Previsto: ativo", "Previsto: cancelamento"],
                y=["Real: ativo", "Real: cancelamento"],
                colorscale=[[0, "#172033"], [1, "#38BDF8"]],
                showscale=False,
                text=matriz,
                texttemplate="%{text}",
                textfont={"size": 20},
            )
        )
        matriz_fig.update_layout(title="Matriz de confusão", height=410)
        st.plotly_chart(matriz_fig, use_container_width=True)

    with grafico_2:
        falso_positivo, verdadeiro_positivo, _ = roc_curve(
            resultado.y_teste,
            resultado.probabilidades_teste,
        )
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=falso_positivo,
                y=verdadeiro_positivo,
                mode="lines",
                name=f"Modelo (AUC {resultado.metricas['roc_auc']:.3f})",
                line={"color": "#22C55E", "width": 3},
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Referência aleatória",
                line={"color": "#94A3B8", "dash": "dash"},
            )
        )
        roc_fig.update_layout(
            title="Curva ROC",
            xaxis_title="Taxa de falsos positivos",
            yaxis_title="Taxa de verdadeiros positivos",
            height=410,
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with st.expander("Como explicar as métricas"):
        st.markdown(
            """
            - **ROC AUC:** capacidade de ordenar clientes de menor e maior risco em vários limiares.
            - **Acurácia:** proporção total de classificações corretas.
            - **Precisão:** entre os clientes sinalizados, quantos realmente cancelaram.
            - **Recall:** entre os clientes que cancelaram, quantos foram identificados.
            - **Brier Score:** qualidade das probabilidades; valores menores são melhores.
            """
        )
        st.write(f"Brier Score no conjunto de teste: **{resultado.metricas['brier']:.3f}**")

    st.subheader("Amostra dos dados")
    amostra = dados.head(100).copy()
    amostra["reajuste_recente"] = amostra["reajuste_recente"].map({0: "Não", 1: "Sim"})
    amostra["cancelou_60d"] = amostra["cancelou_60d"].map({0: "Ativo", 1: "Cancelou"})
    st.dataframe(amostra, hide_index=True, use_container_width=True)
    st.download_button(
        "Baixar base sintética completa",
        data=dados.to_csv(index=False).encode("utf-8"),
        file_name="clientes_assinatura_sinteticos.csv",
        mime="text/csv",
    )

    st.subheader("Limites do projeto")
    st.markdown(
        """
        - Os registros são sintéticos e não representam clientes reais.
        - As probabilidades não foram calibradas em uma empresa.
        - A simulação financeira depende de premissas informadas pelo usuário.
        - Associação não significa causa.
        - Uma aplicação real exigiria validação temporal, monitoramento e teste controlado das campanhas.
        """
    )

st.divider()
st.caption(
    "Projeto de portfólio com dados sintéticos, modelo interpretável e decisões de negócio explicitadas."
)
