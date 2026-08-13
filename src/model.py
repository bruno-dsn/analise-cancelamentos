from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAS, COLUNAS_MODELO, NUMERICAS


ROTULOS_VARIAVEIS = {
    "plano": "Plano",
    "contrato": "Tipo de contrato",
    "forma_pagamento": "Forma de pagamento",
    "canal_aquisicao": "Canal de aquisição",
    "tempo_cliente_meses": "Tempo como cliente",
    "receita_mensal": "Receita mensal",
    "desconto_pct": "Desconto aplicado",
    "horas_uso_semana": "Uso semanal",
    "queda_uso_30d_pct": "Queda recente de uso",
    "dias_sem_acesso": "Dias sem acesso",
    "nota_satisfacao": "Nota de satisfação",
    "chamados_90d": "Chamados ao suporte",
    "falhas_pagamento_90d": "Falhas de pagamento",
    "reajuste_recente": "Reajuste recente",
}


@dataclass
class ResultadoModelo:
    modelo: Pipeline
    metricas: dict[str, float]
    matriz_confusao: list[list[int]]
    y_teste: np.ndarray
    probabilidades_teste: np.ndarray


def criar_pipeline() -> Pipeline:
    preparo = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("numericas", StandardScaler(), NUMERICAS),
        ]
    )
    classificador = LogisticRegression(max_iter=2_000, random_state=42)
    return Pipeline([("preparo", preparo), ("modelo", classificador)])


def calcular_metricas(
    y_real: np.ndarray | pd.Series,
    probabilidades: np.ndarray,
    limiar: float = 0.20,
) -> tuple[dict[str, float], list[list[int]]]:
    previsoes = (probabilidades >= limiar).astype(int)
    metricas = {
        "acuracia": accuracy_score(y_real, previsoes),
        "precisao": precision_score(y_real, previsoes, zero_division=0),
        "recall": recall_score(y_real, previsoes, zero_division=0),
        "f1": f1_score(y_real, previsoes, zero_division=0),
        "roc_auc": roc_auc_score(y_real, probabilidades),
        "brier": brier_score_loss(y_real, probabilidades),
    }
    matriz = confusion_matrix(y_real, previsoes, labels=[0, 1]).tolist()
    return metricas, matriz


def treinar_e_avaliar(dados: pd.DataFrame) -> ResultadoModelo:
    x = dados[COLUNAS_MODELO].copy()
    y = dados["cancelou_60d"].copy()
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    modelo_avaliacao = criar_pipeline()
    modelo_avaliacao.fit(x_treino, y_treino)
    probabilidades = modelo_avaliacao.predict_proba(x_teste)[:, 1]
    metricas, matriz = calcular_metricas(y_teste, probabilidades)

    modelo_final = criar_pipeline()
    modelo_final.fit(x, y)
    return ResultadoModelo(
        modelo=modelo_final,
        metricas=metricas,
        matriz_confusao=matriz,
        y_teste=y_teste.to_numpy(),
        probabilidades_teste=probabilidades,
    )


def avaliar_limiar(
    resultado: ResultadoModelo,
    limiar: float,
) -> tuple[dict[str, float], list[list[int]]]:
    return calcular_metricas(
        resultado.y_teste,
        resultado.probabilidades_teste,
        limiar,
    )


def prever_probabilidade(modelo: Pipeline, cenario: pd.DataFrame) -> float:
    return float(modelo.predict_proba(cenario)[0, 1])


def adicionar_probabilidades(dados: pd.DataFrame, modelo: Pipeline) -> pd.DataFrame:
    resultado = dados.copy()
    resultado["probabilidade_cancelamento"] = modelo.predict_proba(
        resultado[COLUNAS_MODELO]
    )[:, 1]
    return resultado


def explicar_previsao(modelo: Pipeline, cenario: pd.DataFrame) -> pd.DataFrame:
    preparo = modelo.named_steps["preparo"]
    classificador = modelo.named_steps["modelo"]
    transformado = preparo.transform(cenario)
    if hasattr(transformado, "toarray"):
        transformado = transformado.toarray()

    nomes = preparo.get_feature_names_out()
    contribuicoes = transformado[0] * classificador.coef_[0]
    agrupadas: dict[str, float] = {}

    for nome, contribuicao in zip(nomes, contribuicoes, strict=True):
        nome_limpo = nome.split("__", maxsplit=1)[1]
        if nome.startswith("categoricas__"):
            variavel = next(
                coluna for coluna in CATEGORICAS if nome_limpo.startswith(f"{coluna}_")
            )
        else:
            variavel = nome_limpo
        agrupadas[variavel] = agrupadas.get(variavel, 0.0) + float(contribuicao)

    explicacao = pd.DataFrame(
        [
            {
                "variavel": ROTULOS_VARIAVEIS.get(variavel, variavel),
                "contribuicao": contribuicao,
                "efeito": "Aumenta o risco" if contribuicao > 0 else "Reduz o risco",
            }
            for variavel, contribuicao in agrupadas.items()
        ]
    )
    explicacao["magnitude"] = explicacao["contribuicao"].abs()
    return explicacao.sort_values("magnitude", ascending=False).head(8)
