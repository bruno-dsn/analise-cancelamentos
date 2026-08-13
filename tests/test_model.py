from src.data import gerar_dados_sinteticos, montar_cenario
from src.model import explicar_previsao, prever_probabilidade, treinar_e_avaliar


def test_modelo_entrega_metricas_validas():
    dados = gerar_dados_sinteticos(quantidade=2_000, semente=42)
    resultado = treinar_e_avaliar(dados)

    assert 0.68 <= resultado.metricas["roc_auc"] <= 1
    assert 0 <= resultado.metricas["brier"] <= 1
    assert len(resultado.matriz_confusao) == 2


def test_previsao_e_explicacao_local():
    dados = gerar_dados_sinteticos(quantidade=1_500, semente=21)
    resultado = treinar_e_avaliar(dados)
    cenario = montar_cenario(
        plano="Essencial",
        contrato="Mensal",
        forma_pagamento="Cartão",
        canal_aquisicao="Mídia paga",
        tempo_cliente_meses=10,
        receita_mensal=49.90,
        desconto_pct=5,
        horas_uso_semana=2.5,
        queda_uso_30d_pct=42,
        dias_sem_acesso=15,
        nota_satisfacao=6,
        chamados_90d=2,
        falhas_pagamento_90d=1,
        reajuste_recente=1,
    )

    probabilidade = prever_probabilidade(resultado.modelo, cenario)
    explicacao = explicar_previsao(resultado.modelo, cenario)

    assert 0 <= probabilidade <= 1
    assert not explicacao.empty
    assert {"variavel", "contribuicao", "efeito"}.issubset(explicacao.columns)
