from src.analysis import resumir_segmento, simular_campanha
from src.data import gerar_dados_sinteticos
from src.model import adicionar_probabilidades, treinar_e_avaliar


def test_resumo_de_segmento_fecha_com_a_base():
    dados = gerar_dados_sinteticos(quantidade=800, semente=5)
    resumo = resumir_segmento(dados, "plano")

    assert resumo["clientes"].sum() == len(dados)
    assert resumo["cancelamentos"].sum() == dados["cancelou_60d"].sum()


def test_simulacao_financeira_tem_identidade_consistente():
    dados = gerar_dados_sinteticos(quantidade=1_200, semente=8)
    resultado = treinar_e_avaliar(dados)
    dados_risco = adicionar_probabilidades(dados, resultado.modelo)
    campanha = simular_campanha(
        dados_risco,
        capacidade=200,
        custo_por_cliente=15,
        efetividade=0.20,
        horizonte_meses=6,
        margem_contribuicao=0.70,
    )

    assert len(campanha["selecionados"]) == 200
    assert campanha["custo_total"] == 3_000
    assert campanha["retorno_liquido"] == (
        campanha["margem_preservada"] - campanha["custo_total"]
    )
