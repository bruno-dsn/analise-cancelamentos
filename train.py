from pathlib import Path

import joblib

from src.data import carregar_dados
from src.model import treinar_e_avaliar


RAIZ = Path(__file__).resolve().parent


def main() -> None:
    dados = carregar_dados(RAIZ / "data" / "clientes_assinatura_sinteticos.csv")
    resultado = treinar_e_avaliar(dados)
    destino = RAIZ / "models" / "modelo_cancelamento.joblib"
    destino.parent.mkdir(exist_ok=True)
    joblib.dump(resultado.modelo, destino)
    print(f"Modelo salvo em: {destino}")
    print(f"ROC AUC: {resultado.metricas['roc_auc']:.3f}")
    print(f"Recall no limiar padrão: {resultado.metricas['recall']:.1%}")


if __name__ == "__main__":
    main()
