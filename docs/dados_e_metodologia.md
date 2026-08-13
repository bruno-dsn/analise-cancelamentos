# Dados e metodologia

## Problema de negócio

O projeto representa um serviço brasileiro por assinatura que deseja identificar sinais associados ao cancelamento. Cada linha descreve o retrato de um cliente em uma data de referência. O alvo `cancelou_60d` informa se houve cancelamento nos 60 dias seguintes.

Essa definição evita misturar no modelo informações conhecidas somente depois do cancelamento.

## Por que os dados são sintéticos

Dados individuais de retenção costumam ser privados e podem conter informações comerciais ou pessoais. A base original do repositório não documentava origem, licença nem processo de coleta. Por isso, ela foi removida.

A nova base possui 12.000 clientes fictícios, gerados com semente fixa. Nenhum identificador representa uma pessoa ou empresa real. O processo completo está em `src/data.py` e pode ser repetido com `python -m scripts.gerar_dados`.

## Variáveis

| Coluna | Uso no projeto |
| --- | --- |
| `cliente_id` | Identificador fictício, usado somente para contagem e exportação |
| `plano` | Essencial, Plus ou Premium |
| `contrato` | Mensal ou anual |
| `forma_pagamento` | Cartão, Pix, débito automático ou boleto |
| `canal_aquisicao` | Indicação, busca orgânica, mídia paga ou parceiro |
| `tempo_cliente_meses` | Tempo de relacionamento até o retrato |
| `receita_mensal` | Receita recorrente do cliente em reais |
| `desconto_pct` | Desconto vigente no plano |
| `horas_uso_semana` | Intensidade média de uso do serviço |
| `queda_uso_30d_pct` | Redução de uso nos 30 dias anteriores |
| `dias_sem_acesso` | Dias desde o último acesso |
| `nota_satisfacao` | Nota de satisfação sintética de 0 a 10 |
| `chamados_90d` | Chamados ao suporte nos 90 dias anteriores |
| `falhas_pagamento_90d` | Falhas de cobrança nos 90 dias anteriores |
| `reajuste_recente` | Indica se houve reajuste de preço recente |
| `cancelou_60d` | Alvo: 1 para cancelamento e 0 para permanência |

Idade, sexo, região, raça, estado civil e outros atributos pessoais não são gerados nem utilizados.

## Geração do alvo

A probabilidade sintética de cancelamento é construída com uma função logística. Queda de uso, muitos dias sem acesso, chamados, falhas de pagamento, baixa satisfação, contrato mensal e reajuste recente elevam a probabilidade. Maior tempo de relacionamento, contrato anual, aquisição por indicação e desconto ativo reduzem a probabilidade.

Um componente aleatório impede que o alvo seja uma regra determinística. Essas relações foram definidas para estudo e não representam estimativas do mercado brasileiro.

## Modelo

O pipeline aplica codificação one-hot às categorias, padronização às variáveis numéricas e regressão logística. A escolha da regressão logística mantém o modelo simples, rápido e explicável.

A base é separada de forma estratificada:

1. 75% para treinamento;
2. 25% para teste;
3. semente 42 para reprodução;
4. avaliação por ROC AUC, acurácia, precisão, recall, F1 e Brier Score.

O limiar inicial de 20% prioriza a identificação de cancelamentos. A aplicação permite alterar esse limite e observar a mudança nos erros.

## Simulação financeira

A campanha ordena clientes pela probabilidade prevista e seleciona os primeiros conforme a capacidade informada. O retorno potencial depende de quatro hipóteses:

1. custo por contato;
2. percentual de cancelamentos evitados;
3. horizonte de receita preservada;
4. margem de contribuição.

As retenções esperadas correspondem à soma das probabilidades dos selecionados multiplicada pela efetividade escolhida. O resultado é um cenário, não uma promessa de retorno.

## Limitações

- dados sintéticos e sem validação em empresa real;
- relações associativas, não causais;
- ausência de validação temporal e monitoramento em produção;
- efetividade de campanha definida pelo usuário;
- necessidade de teste controlado para medir impacto real.
