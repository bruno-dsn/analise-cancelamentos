from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import classificar_risco, resumir_segmento
from src.data import carregar_dados
from src.model import adicionar_probabilidades, treinar_e_avaliar


RAIZ = Path(__file__).resolve().parents[1]
FUNDO = "#0B1120"
PAINEL = "#111827"
TEXTO = "#F1F5F9"
MUTED = "#94A3B8"
AZUL = "#38BDF8"
VERDE = "#22C55E"
LARANJA = "#F97316"
VERMELHO = "#EF4444"


def moeda_curta(valor: float) -> str:
    if valor >= 1_000_000:
        return f"R$ {valor / 1_000_000:.1f} mi".replace(".", ",")
    if valor >= 1_000:
        return f"R$ {valor / 1_000:.1f} mil".replace(".", ",")
    return f"R$ {valor:.0f}"


def criar_capa_linkedin(dados, resultado, dados_risco) -> Path:
    figura = plt.figure(figsize=(12, 6.27), facecolor=FUNDO)
    eixo = figura.add_axes([0, 0, 1, 1])
    eixo.set_axis_off()

    eixo.add_patch(
        plt.Rectangle((0.055, 0.09), 0.89, 0.82, color=PAINEL, ec="#25324A", lw=1.5)
    )
    eixo.add_patch(plt.Rectangle((0.055, 0.09), 0.012, 0.82, color=AZUL))
    eixo.text(
        0.105,
        0.80,
        "CIÊNCIA DE DADOS APLICADA À RETENÇÃO",
        color=AZUL,
        fontsize=12,
        fontweight="bold",
    )
    eixo.text(
        0.105,
        0.61,
        "Laboratório de\nRetenção de Clientes",
        color=TEXTO,
        fontsize=29,
        fontweight="bold",
        linespacing=1.05,
    )
    eixo.text(
        0.105,
        0.43,
        "Risco de cancelamento, explicabilidade e\nsimulação financeira de campanhas",
        color="#CBD5E1",
        fontsize=14,
        linespacing=1.35,
    )

    receita_risco = dados_risco.loc[
        dados_risco["probabilidade_cancelamento"] >= 0.30,
        "receita_mensal",
    ].sum()
    destaques = [
        ("12.000", "clientes sintéticos"),
        (f"{dados['cancelou_60d'].mean():.1%}".replace(".", ","), "cancelamento em 60 dias"),
        (f"{resultado.metricas['roc_auc']:.3f}", "ROC AUC no teste"),
        (moeda_curta(receita_risco), "receita mensal em alto risco"),
    ]
    posicoes = [0.105, 0.315, 0.525, 0.735]
    for x, (valor, rotulo) in zip(posicoes, destaques, strict=True):
        eixo.text(x, 0.245, valor, color=TEXTO, fontsize=16, fontweight="bold")
        eixo.text(x, 0.185, rotulo, color=MUTED, fontsize=8.7)

    eixo.text(
        0.84,
        0.835,
        "BRUNO NUNES",
        color="#CBD5E1",
        fontsize=9,
        fontweight="bold",
        ha="right",
    )
    destino = RAIZ / "assets" / "capa_linkedin.png"
    figura.savefig(destino, dpi=100, facecolor=FUNDO)
    plt.close(figura)
    return destino


def main() -> None:
    dados = carregar_dados(RAIZ / "data" / "clientes_assinatura_sinteticos.csv")
    resultado = treinar_e_avaliar(dados)
    dados_risco = adicionar_probabilidades(dados, resultado.modelo)
    dados_risco["faixa_risco"] = classificar_risco(
        dados_risco["probabilidade_cancelamento"]
    )

    figura = plt.figure(figsize=(16, 9), facecolor=FUNDO)
    grade = figura.add_gridspec(
        3,
        4,
        height_ratios=[0.72, 1.55, 1.55],
        hspace=0.48,
        wspace=0.42,
    )
    figura.suptitle(
        "Laboratório de Retenção de Clientes",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=TEXTO,
    )
    figura.text(
        0.06,
        0.922,
        "Visão executiva da carteira sintética brasileira",
        color=MUTED,
        fontsize=11,
    )

    metricas = [
        ("Clientes", f"{len(dados):,}".replace(",", ".")),
        ("Cancelamento em 60 dias", f"{dados['cancelou_60d'].mean():.1%}".replace(".", ",")),
        ("Receita mensal", moeda_curta(dados["receita_mensal"].sum())),
        (
            "Receita em alto risco",
            moeda_curta(
                dados_risco.loc[
                    dados_risco["probabilidade_cancelamento"] >= 0.30,
                    "receita_mensal",
                ].sum()
            ),
        ),
    ]
    for indice, (rotulo, valor) in enumerate(metricas):
        eixo = figura.add_subplot(grade[0, indice])
        eixo.set_facecolor(PAINEL)
        eixo.set_xticks([])
        eixo.set_yticks([])
        for borda in eixo.spines.values():
            borda.set_color("#25324A")
        eixo.text(0.06, 0.68, rotulo, color=MUTED, fontsize=10, transform=eixo.transAxes)
        eixo.text(
            0.06,
            0.25,
            valor,
            color=TEXTO,
            fontsize=19,
            fontweight="bold",
            transform=eixo.transAxes,
        )

    contrato = resumir_segmento(dados, "contrato").sort_values("taxa_cancelamento")
    eixo_contrato = figura.add_subplot(grade[1:, 0:2])
    eixo_contrato.set_facecolor(PAINEL)
    barras = eixo_contrato.barh(
        contrato["contrato"],
        contrato["taxa_cancelamento"] * 100,
        color=[VERDE, LARANJA],
        height=0.52,
    )
    eixo_contrato.set_title(
        "Cancelamento por tipo de contrato",
        color=TEXTO,
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    eixo_contrato.set_xlabel("Taxa observada (%)", color=MUTED)
    eixo_contrato.tick_params(colors=TEXTO)
    eixo_contrato.grid(axis="x", alpha=0.15, color=MUTED)
    for barra, valor in zip(barras, contrato["taxa_cancelamento"] * 100, strict=True):
        eixo_contrato.text(
            valor + 0.35,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.1f}%".replace(".", ","),
            va="center",
            color=TEXTO,
            fontsize=11,
        )
    eixo_contrato.set_xlim(0, contrato["taxa_cancelamento"].max() * 125)

    faixas = (
        dados_risco["faixa_risco"]
        .value_counts(sort=False)
        .reindex(["Baixo", "Atenção", "Alto", "Crítico"], fill_value=0)
    )
    eixo_risco = figura.add_subplot(grade[1, 2:4])
    eixo_risco.set_facecolor(PAINEL)
    cores = [VERDE, "#F2C94C", LARANJA, VERMELHO]
    barras_risco = eixo_risco.bar(faixas.index, faixas.values, color=cores)
    eixo_risco.set_title(
        "Clientes por faixa de risco",
        color=TEXTO,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    eixo_risco.tick_params(colors=TEXTO)
    eixo_risco.grid(axis="y", alpha=0.15, color=MUTED)
    for barra, valor in zip(barras_risco, faixas.values, strict=True):
        eixo_risco.text(
            barra.get_x() + barra.get_width() / 2,
            valor + max(faixas.values) * 0.018,
            f"{valor:,}".replace(",", "."),
            ha="center",
            color=TEXTO,
            fontsize=9,
        )

    pagamento = resumir_segmento(dados, "forma_pagamento").sort_values(
        "taxa_cancelamento"
    )
    eixo_pagamento = figura.add_subplot(grade[2, 2:4])
    eixo_pagamento.set_facecolor(PAINEL)
    cores_pagamento = plt.cm.Blues(np.linspace(0.45, 0.9, len(pagamento)))
    barras_pagamento = eixo_pagamento.barh(
        pagamento["forma_pagamento"],
        pagamento["taxa_cancelamento"] * 100,
        color=cores_pagamento,
    )
    eixo_pagamento.set_title(
        "Cancelamento por forma de pagamento",
        color=TEXTO,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    eixo_pagamento.tick_params(colors=TEXTO, labelsize=9)
    eixo_pagamento.grid(axis="x", alpha=0.15, color=MUTED)
    for barra, valor in zip(
        barras_pagamento,
        pagamento["taxa_cancelamento"] * 100,
        strict=True,
    ):
        eixo_pagamento.text(
            valor + 0.25,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.1f}%".replace(".", ","),
            va="center",
            color=TEXTO,
            fontsize=9,
        )

    for eixo in [eixo_contrato, eixo_risco, eixo_pagamento]:
        for borda in eixo.spines.values():
            borda.set_color("#25324A")

    figura.text(
        0.06,
        0.02,
        f"Modelo: regressão logística | ROC AUC {resultado.metricas['roc_auc']:.3f} | Dados sintéticos e reproduzíveis",
        color=MUTED,
        fontsize=9,
    )
    destino = RAIZ / "assets" / "painel_retencao.png"
    figura.savefig(destino, dpi=180, bbox_inches="tight", facecolor=FUNDO)
    plt.close(figura)
    print(f"Imagem criada: {destino}")
    capa_linkedin = criar_capa_linkedin(dados, resultado, dados_risco)
    print(f"Imagem criada: {capa_linkedin}")


if __name__ == "__main__":
    main()
