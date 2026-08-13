import pandas as pd


ROTULOS_SEGMENTOS = {
    "plano": "Plano",
    "contrato": "Tipo de contrato",
    "forma_pagamento": "Forma de pagamento",
    "canal_aquisicao": "Canal de aquisição",
}


def taxa_cancelamento(dados: pd.DataFrame) -> float:
    if dados.empty:
        return 0.0
    return float(dados["cancelou_60d"].mean())


def resumir_segmento(dados: pd.DataFrame, coluna: str) -> pd.DataFrame:
    taxa_geral = taxa_cancelamento(dados)
    resumo = (
        dados.groupby(coluna, dropna=False)
        .agg(
            clientes=("cliente_id", "count"),
            cancelamentos=("cancelou_60d", "sum"),
            taxa_cancelamento=("cancelou_60d", "mean"),
            receita_mensal=("receita_mensal", "sum"),
        )
        .reset_index()
    )
    resumo["lift"] = (
        resumo["taxa_cancelamento"] / taxa_geral if taxa_geral else 0.0
    )
    return resumo.sort_values("taxa_cancelamento", ascending=False)


def classificar_risco(probabilidade: pd.Series) -> pd.Series:
    return pd.cut(
        probabilidade,
        bins=[-0.001, 0.15, 0.30, 0.50, 1.0],
        labels=["Baixo", "Atenção", "Alto", "Crítico"],
    )


def simular_campanha(
    dados_com_risco: pd.DataFrame,
    capacidade: int,
    custo_por_cliente: float,
    efetividade: float,
    horizonte_meses: int,
    margem_contribuicao: float,
) -> dict[str, float | int | pd.DataFrame]:
    selecionados = (
        dados_com_risco.sort_values("probabilidade_cancelamento", ascending=False)
        .head(capacidade)
        .copy()
    )
    cancelamentos_esperados = float(selecionados["probabilidade_cancelamento"].sum())
    clientes_retidos = cancelamentos_esperados * efetividade
    receita_preservada = float(
        (
            selecionados["receita_mensal"]
            * selecionados["probabilidade_cancelamento"]
            * efetividade
            * horizonte_meses
        ).sum()
    )
    margem_preservada = receita_preservada * margem_contribuicao
    custo_total = capacidade * custo_por_cliente
    retorno_liquido = margem_preservada - custo_total
    roi = retorno_liquido / custo_total if custo_total else 0.0

    return {
        "selecionados": selecionados,
        "cancelamentos_esperados": cancelamentos_esperados,
        "clientes_retidos": clientes_retidos,
        "receita_preservada": receita_preservada,
        "margem_preservada": margem_preservada,
        "custo_total": custo_total,
        "retorno_liquido": retorno_liquido,
        "roi": roi,
    }
