from src.data import COLUNAS_MODELO, gerar_dados_sinteticos, montar_cenario


def test_dados_sao_reproduziveis_e_completos():
    primeira = gerar_dados_sinteticos(quantidade=400, semente=7)
    segunda = gerar_dados_sinteticos(quantidade=400, semente=7)

    assert primeira.equals(segunda)
    assert primeira.shape == (400, 16)
    assert set(COLUNAS_MODELO).issubset(primeira.columns)
    assert primeira["cliente_id"].is_unique
    assert primeira["cancelou_60d"].isin([0, 1]).all()


def test_cenario_respeita_a_ordem_do_modelo():
    cenario = montar_cenario(
        plano="Plus",
        contrato="Mensal",
        forma_pagamento="Cartão",
        canal_aquisicao="Busca orgânica",
        tempo_cliente_meses=12,
        receita_mensal=79.90,
        desconto_pct=5,
        horas_uso_semana=4.0,
        queda_uso_30d_pct=25,
        dias_sem_acesso=8,
        nota_satisfacao=7,
        chamados_90d=1,
        falhas_pagamento_90d=0,
        reajuste_recente=0,
    )
    assert list(cenario.columns) == COLUNAS_MODELO
