from pathlib import Path

import numpy as np
import pandas as pd


PLANOS = ["Essencial", "Plus", "Premium"]
CONTRATOS = ["Mensal", "Anual"]
PAGAMENTOS = ["Cartão", "Pix", "Débito automático", "Boleto"]
CANAIS = ["Indicação", "Busca orgânica", "Mídia paga", "Parceiro"]

COLUNAS_MODELO = [
    "plano",
    "contrato",
    "forma_pagamento",
    "canal_aquisicao",
    "tempo_cliente_meses",
    "receita_mensal",
    "desconto_pct",
    "horas_uso_semana",
    "queda_uso_30d_pct",
    "dias_sem_acesso",
    "nota_satisfacao",
    "chamados_90d",
    "falhas_pagamento_90d",
    "reajuste_recente",
]

CATEGORICAS = ["plano", "contrato", "forma_pagamento", "canal_aquisicao"]
NUMERICAS = [coluna for coluna in COLUNAS_MODELO if coluna not in CATEGORICAS]


def _sigmoide(valor: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-valor))


def gerar_dados_sinteticos(
    quantidade: int = 12_000,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera retratos fictícios de clientes de um serviço por assinatura."""
    rng = np.random.default_rng(semente)

    plano = rng.choice(PLANOS, quantidade, p=[0.46, 0.36, 0.18])
    contrato = rng.choice(CONTRATOS, quantidade, p=[0.64, 0.36])
    forma_pagamento = rng.choice(PAGAMENTOS, quantidade, p=[0.47, 0.18, 0.24, 0.11])
    canal = rng.choice(CANAIS, quantidade, p=[0.24, 0.31, 0.33, 0.12])

    tempo_cliente = np.clip(rng.gamma(2.2, 13.5, quantidade), 1, 96).round().astype(int)
    desconto = rng.choice([0, 5, 10, 15, 20], quantidade, p=[0.39, 0.19, 0.23, 0.12, 0.07])

    valor_plano = pd.Series(plano).map(
        {"Essencial": 49.90, "Plus": 79.90, "Premium": 119.90}
    ).to_numpy()
    receita_mensal = valor_plano * (1 - desconto / 100)

    horas_base = pd.Series(plano).map(
        {"Essencial": 3.7, "Plus": 5.4, "Premium": 7.2}
    ).to_numpy()
    horas_uso = np.clip(rng.gamma(2.5, horas_base / 2.5), 0, 28)

    queda_uso = np.clip(
        rng.beta(1.45, 3.8, quantidade) * 100
        + (tempo_cliente < 4) * rng.uniform(0, 12, quantidade),
        0,
        100,
    )
    dias_sem_acesso = np.clip(
        rng.poisson(3.8 + queda_uso / 9.5, quantidade),
        0,
        60,
    )

    lambda_chamados = 0.55 + dias_sem_acesso / 34 + (plano == "Premium") * 0.25
    chamados = np.clip(rng.poisson(lambda_chamados), 0, 10)

    risco_pagamento = pd.Series(forma_pagamento).map(
        {"Cartão": 0.44, "Pix": 0.16, "Débito automático": 0.18, "Boleto": 0.62}
    ).to_numpy()
    falhas_pagamento = np.clip(
        rng.poisson(risco_pagamento + (contrato == "Mensal") * 0.12),
        0,
        6,
    )

    reajuste_recente = rng.binomial(1, 0.28, quantidade)
    satisfacao_latente = (
        8.4
        - queda_uso * 0.032
        - chamados * 0.48
        - falhas_pagamento * 0.28
        - reajuste_recente * 0.38
        + (plano == "Premium") * 0.25
        + rng.normal(0, 1.35, quantidade)
    )
    nota_satisfacao = np.clip(np.rint(satisfacao_latente), 0, 10).astype(int)

    efeito_pagamento = pd.Series(forma_pagamento).map(
        {"Cartão": 0.05, "Pix": -0.12, "Débito automático": -0.20, "Boleto": 0.22}
    ).to_numpy()
    efeito_canal = pd.Series(canal).map(
        {"Indicação": -0.24, "Busca orgânica": -0.10, "Mídia paga": 0.18, "Parceiro": 0.03}
    ).to_numpy()

    logit = (
        -4.10
        + (contrato == "Mensal") * 0.82
        + queda_uso * 0.029
        + dias_sem_acesso * 0.040
        + chamados * 0.32
        + falhas_pagamento * 0.66
        + (6 - nota_satisfacao) * 0.32
        + reajuste_recente * 0.46
        - np.minimum(tempo_cliente, 48) * 0.016
        - (desconto >= 10) * 0.14
        + efeito_pagamento * 1.25
        + efeito_canal * 1.25
        + rng.normal(0, 0.40, quantidade)
    )
    probabilidade_cancelamento = _sigmoide(logit)
    cancelou_60d = rng.binomial(1, probabilidade_cancelamento)

    return pd.DataFrame(
        {
            "cliente_id": [f"CLI{numero:05d}" for numero in range(1, quantidade + 1)],
            "plano": plano,
            "contrato": contrato,
            "forma_pagamento": forma_pagamento,
            "canal_aquisicao": canal,
            "tempo_cliente_meses": tempo_cliente,
            "receita_mensal": receita_mensal.round(2),
            "desconto_pct": desconto.astype(int),
            "horas_uso_semana": horas_uso.round(1),
            "queda_uso_30d_pct": queda_uso.round(1),
            "dias_sem_acesso": dias_sem_acesso.astype(int),
            "nota_satisfacao": nota_satisfacao,
            "chamados_90d": chamados.astype(int),
            "falhas_pagamento_90d": falhas_pagamento.astype(int),
            "reajuste_recente": reajuste_recente.astype(int),
            "cancelou_60d": cancelou_60d.astype(int),
        }
    )


def carregar_dados(caminho: str | Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    obrigatorias = set(["cliente_id", "cancelou_60d", *COLUNAS_MODELO])
    faltantes = obrigatorias - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")
    if dados["cliente_id"].duplicated().any():
        raise ValueError("A coluna cliente_id contém valores duplicados.")
    return dados


def montar_cenario(
    plano: str,
    contrato: str,
    forma_pagamento: str,
    canal_aquisicao: str,
    tempo_cliente_meses: int,
    receita_mensal: float,
    desconto_pct: int,
    horas_uso_semana: float,
    queda_uso_30d_pct: float,
    dias_sem_acesso: int,
    nota_satisfacao: int,
    chamados_90d: int,
    falhas_pagamento_90d: int,
    reajuste_recente: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "plano": plano,
                "contrato": contrato,
                "forma_pagamento": forma_pagamento,
                "canal_aquisicao": canal_aquisicao,
                "tempo_cliente_meses": tempo_cliente_meses,
                "receita_mensal": receita_mensal,
                "desconto_pct": desconto_pct,
                "horas_uso_semana": horas_uso_semana,
                "queda_uso_30d_pct": queda_uso_30d_pct,
                "dias_sem_acesso": dias_sem_acesso,
                "nota_satisfacao": nota_satisfacao,
                "chamados_90d": chamados_90d,
                "falhas_pagamento_90d": falhas_pagamento_90d,
                "reajuste_recente": reajuste_recente,
            }
        ],
        columns=COLUNAS_MODELO,
    )
