# Avaliação reproduzível

Comparação no mesmo teste estratificado, separado antes do ajuste do pipeline. Os hiperparâmetros e o limiar foram mantidos fixos; esta rotina não seleciona modelos pelo resultado de teste.

Treino: 9000 linhas. Teste: 3000 linhas. Classe positiva no teste: 11.33%. Limiar: 20%.

| Métrica | Previsão pela prevalência de treino | Regressão logística |
|---|---:|---:|
| acuracia | 0.8867 | 0.8320 |
| precisao | 0.0000 | 0.3392 |
| recall | 0.0000 | 0.5088 |
| f1 | 0.0000 | 0.4071 |
| roc_auc | 0.5000 | 0.7880 |
| average_precision | 0.1133 | 0.3649 |
| brier | 0.1005 | 0.0860 |

## Interpretação

Acurácia deve ser comparada à referência que classifica todos na classe majoritária. Um modelo pode recuperar mais casos positivos e ter acurácia menor. Average precision resume a curva precisão-recall; Brier mede erro probabilístico, mas isoladamente não demonstra calibração. ROC AUC não é porcentagem de acertos.

## Limites e uso do aplicativo

Os dados foram gerados artificialmente, com relações definidas pelo próprio gerador. Os resultados medem este experimento, não generalização para clientes reais. Uma avaliação empresarial requer dados representativos, validação temporal, custo de erros e monitoramento.

O aplicativo refaz o ajuste em toda a base para demonstração. Seus scores nessa base não constituem avaliação fora da amostra. O simulador de negócio usa hipóteses; ganhos de receita ou redução de inadimplência/cancelamento não foram medidos.

## Reproduzir

```bash
python -m pip install -r requirements-dev.txt
python evaluate.py
```

O arquivo evaluation.json registra versões, semente, SHA-256 dos dados, métricas e matrizes de confusão. Resultados podem variar ligeiramente entre versões das bibliotecas.
