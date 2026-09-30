"""Compare a fixed logistic-regression experiment with a prior-only baseline."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.data import COLUNAS_MODELO, carregar_dados
from src.model import criar_pipeline, calcular_metricas

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "data" / "clientes_assinatura_sinteticos.csv"
    data = carregar_dados(source)
    x_train, x_test, y_train, y_test = train_test_split(
        data[COLUNAS_MODELO], data["cancelou_60d"],
        test_size=0.25, stratify=data["cancelou_60d"], random_state=42,
    )
    results = {}
    for name, model in [("prior_baseline", DummyClassifier(strategy="prior")),
                        ("logistic_regression", criar_pipeline())]:
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        metrics, matrix = calcular_metricas(y_test, proba, .20)
        results[name] = {"metrics": {key: float(value) for key, value in metrics.items()},
                         "confusion_matrix": matrix}
    report = {
        "dataset": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_kind": "synthetic", "seed": 42, "threshold": .20,
        "train_rows": len(x_train), "test_rows": len(x_test),
        "test_positive_rate": float(y_test.mean()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "results": results,
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Avaliação reproduzível", "",
        "Comparação no mesmo teste estratificado, separado antes do ajuste do pipeline. "
        "Os hiperparâmetros e o limiar foram mantidos fixos; esta rotina não seleciona modelos pelo resultado de teste.", "",
        f"Treino: {len(x_train)} linhas. Teste: {len(x_test)} linhas. Classe positiva no teste: {y_test.mean():.2%}. Limiar: {.20:.0%}.", "",
        "| Métrica | Previsão pela prevalência de treino | Regressão logística |",
        "|---|---:|---:|"]
    for key in results["logistic_regression"]["metrics"]:
        lines.append(f"| {key} | {results['prior_baseline']['metrics'][key]:.4f} | {results['logistic_regression']['metrics'][key]:.4f} |")
    lines += ["", "## Interpretação", "",
        "Acurácia deve ser comparada à referência que classifica todos na classe majoritária. "
        "Um modelo pode recuperar mais casos positivos e ter acurácia menor. "
        "Average precision resume a curva precisão-recall; Brier mede erro probabilístico, "
        "mas isoladamente não demonstra calibração. ROC AUC não é porcentagem de acertos.", "",
        "## Limites e uso do aplicativo", "",
        "Os dados foram gerados artificialmente, com relações definidas pelo próprio gerador. "
        "Os resultados medem este experimento, não generalização para clientes reais. "
        "Uma avaliação empresarial requer dados representativos, validação temporal, custo de erros e monitoramento.", "",
        "O aplicativo refaz o ajuste em toda a base para demonstração. Seus scores nessa base "
        "não constituem avaliação fora da amostra. O simulador de negócio usa hipóteses; ganhos "
        "de receita ou redução de inadimplência/cancelamento não foram medidos.", "",
        "## Reproduzir", "", "```bash", "python -m pip install -r requirements-dev.txt", "python evaluate.py", "```", "",
        "O arquivo evaluation.json registra versões, semente, SHA-256 dos dados, métricas e matrizes de confusão. "
        "Resultados podem variar ligeiramente entre versões das bibliotecas."]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
