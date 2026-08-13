# Decisões do projeto

## O que foi removido

A versão anterior usava uma base sem fonte rastreável e uma regra fixa que chamava vários clientes de prioridade. A taxa de cancelamento chegava a separações quase perfeitas, sinal de que o próprio alvo podia ter sido criado a partir das variáveis analisadas.

Essa base foi substituída por dados sintéticos documentados. A regra fixa foi substituída por um modelo probabilístico avaliado fora da amostra de treino.

## O que o aplicativo responde

1. Qual é a taxa observada de cancelamento?
2. Em quais segmentos ela se concentra?
3. Qual é o risco estimado de um cenário individual?
4. Quais fatores influenciaram a previsão?
5. Qual seria o retorno potencial de uma campanha sob premissas explícitas?
6. Como precisão e recall mudam quando o limiar é alterado?

## Por que regressão logística

O objetivo não era vencer uma competição de acurácia. Era construir um projeto que pudesse ser explicado e auditado. A regressão logística produz probabilidades, aceita variáveis numéricas e categóricas dentro de um pipeline e permite decompor o sentido das contribuições de cada variável.

## Como defender o resultado

O modelo obteve ROC AUC de aproximadamente 0,788 no conjunto de teste. Isso mostra capacidade útil, mas não perfeita, de ordenar clientes de menor e maior risco. No limiar de 20%, o recall fica próximo de 51% e a precisão próxima de 34%.

Esses valores não devem ser escondidos. Em retenção, o limiar depende do custo de contato e do valor do cliente. Se o contato for barato, a empresa pode aceitar mais falsos positivos para encontrar mais cancelamentos. Se for caro, pode exigir maior precisão.

## O que ainda seria necessário em produção

- dados reais com autorização e governança;
- divisão temporal entre treino e teste;
- comparação com modelos de referência;
- calibração das probabilidades;
- monitoramento de desempenho e deriva;
- teste A/B das ações de retenção;
- revisão periódica de vieses e regras de uso.
