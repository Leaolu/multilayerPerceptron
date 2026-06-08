# Multilayer Perceptron (MLP) — Trabalho de Inteligência Artificial

## Integrantes do Grupo

| Nome | Número USP |
|------|-----------|
| Lucas Leão Ferreira Barbosa | 15639553 |
| João Victor de Pascale Souza | 15463888 |
| Lion Chen | 15470010 |
| Giovanni Willik Del Piccolo | 15455979 |

## Fonte Teórica

Toda a implementação é baseada no conteúdo da aula:

> **Redes Neurais Artificiais - Perceptron Simples e Multilayer Perceptron**  
> Profa. Dra. Sarajane Marques Peres  
> Disciplina: Inteligência Artificial — Bacharelado em Sistemas de Informação (EACH-USP)

As referências nos comentários do código seguem a numeração das slides desta aula (ex: "slide 72 - Fausett" refere-se à slide 72 do PDF acima).

---

## Descrição dos Arquivos de Código

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Script principal que orquestra o pipeline completo: carregamento de dados, inicialização da rede, treinamento com parada antecipada, avaliação no conjunto de teste e teste com base ruidosa. Gera também: matriz de confusão em PNG |
| `mlp_init.py` | Funções de inicialização: geração de pesos aleatórios (Passo 0 - Fausett), carregamento dos datasets de caracteres (multiclasse e binário), divisão treino/validação/teste, e persistência de pesos. |
| `mlp_forward.py` | Implementação do estágio feedforward (Passos 3, 4 e 5 - Fausett): função sigmoide, propagação pela camada escondida e camada de saída, e predição via index do maior elemento. |
| `mlp_backward.py` | Implementação da backpropagation do erro (Passos 6 e 7 - Fausett): cálculo dos gradientes locais (δ) para camada de saída e escondida, e cálculo das correções de pesos. |
| `mlp_treino.py` | Loop de treinamento completo com atualização de pesos (Passo 8 - Fausett), cálculo do erro quadrático médio, e lógica de parada antecipada monitorando o erro de validação. |
| `mlp_avaliacao.py` | Funções de avaliação incluindo: matriz de confusão, acurácia, plotagem da curva de erro e **visualização em PNG da matriz de confusão com labels de classes** |
| `gerar_ruido.py` | Geração da base ruidosa (130 amostras): cria 5 réplicas de cada letra do alfabeto com ruído aleatório diferente (inversão de 15% dos pixels) para testar a robustez/generalização da rede treinada. |
| `melhor_combinacao.py` | Análise pós-experimentos: varre os resultados de todos os experimentos de um caso, identifica o modelo com melhor acurácia no teste e isola seus arquivos em diretório dedicado. |
| `pior_combinacao.py` | Análise pós-experimentos inversa — identifica o modelo com **pior acurácia** no teste para análise contrastante e isolamento em diretório dedicado. |
| `plot_tempos_3d.py` | Gera 4 gráficos 3D por caso mostrando **tempo acumulado vs épocas vs hiperparâmetro** (neurônios ocultos, alpha, erro mínimo, max épocas). |
| `run_experimentos.sh` | Script shell que automatiza a execução de 972 experimentos, variando hiperparâmetros para os 3 casos. |

---

## Como Executar

### Pré-requisitos

- Python 3.x
- NumPy
- Matplotlib

### Gerar base ruidosa (com 130 amostras)

```bash
python gerar_ruido.py
```

### Executar um treinamento individual

```bash
python main.py --n_escondidos 30 --alpha 0.1 --max_epocas 1500 --erro_minimo 0.01 --paciencia 20 --caso 0
```

### Executar bateria completa de experimentos

```bash
chmod +x run_experimentos.sh
./run_experimentos.sh
```

### Encontrar melhor combinação de um caso

```bash
python melhor_combinacao.py --caso 0
```

### Encontrar pior combinação de um caso

```bash
python pior_combinacao.py --caso 0
```

### Gerar gráficos 3D de análise de tempo

```bash
# Execute após a bateria de experimentos (ou execute individualmente):
python plot_tempos_3d.py
```

---

## Arquivos de Saída Gerados

### Por execução individual (`main.py`)

**Arquivos na raiz do experimento:**

| Arquivo | Descrição |
|---------|-----------|
| `hiperparametros.txt` | Hiperparâmetros da arquitetura (entradas, ocultas, saídas, alpha, paciência, épocas, erro mínimo). |
| `pesos_iniciais.txt` | Pesos e bias iniciais da rede (antes do treinamento). |
| `pesos_finais.txt` | Pesos e bias finais da rede (após o treinamento). |
| `historico_erros_original.csv` | Erro quadrático médio de treino e validação em cada época (dados originais). |
| `historico_erros_ruidoso.csv` | Erro quadrático médio ao testar nós dados ruidosos (130 amostras). |
| `tempo_execucao_original.txt` | Tempo total de treinamento e teste nos dados originais (segundos). |
| `tempo_execucao_ruidoso.txt` | Tempo de teste nos dados ruidosos (segundos). |
| `grafico_analise_completa.png` | Gráficos de convergência de erro e estabilidade dos pesos. |

**Pasta `original/` — Resultados no conjunto de teste original (130 amostras):**

| Arquivo | Descrição |
|---------|-----------|
| `saidas_teste.csv` | Rótulo real e classe predita para cada amostra do conjunto de teste original. |
| `matriz_confusao_teste.csv` | Matriz de confusão em formato CSV. |
| `matriz_confusao_teste.png` | Visualização em PNG da matriz de confusão com labels das classes. |
| `resumo_teste_original.txt` | Erro médio e acurácia no conjunto de teste original. |

**Pasta `ruidoso/` — Resultados no conjunto de teste ruidoso (130 amostras com 5 variações por letra):**

| Arquivo | Descrição |
|---------|-----------|
| `saidas_ruidoso.csv` | Rótulo real e classe predita para cada amostra do conjunto ruidoso. |
| `matriz_confusao_ruidoso.csv` | Matriz de confusão em formato CSV. |
| `matriz_confusao_ruidoso.png` | Visualização em PNG da matriz de confusão com labels das classes. |
| `resumo_teste_ruidoso.txt` | Erro médio e acurácia no conjunto ruidoso. |
| `X_ruidoso.txt` | Dados das amostras ruidosas (cópia para referência). |
| `Y_ruidoso.txt` | Rótulos das amostras ruidosas (cópia para referência). |

### Pela bateria de experimentos (`run_experimentos.sh`)

O script gera a seguinte estrutura:

```
resultados_mlp/
├── resumo_geral_experimentos.csv          # Índice de todos os experimentos executados
├── caso_0/                                # Experimentos do caso multiclasse (A-Z)
│   ├── exp_1_h15_a0.01_e1000_err0.05/
│   │   ├── hiperparametros.txt
│   │   ├── pesos_iniciais.txt
│   │   ├── pesos_finais.txt
│   │   ├── historico_erros_original.csv
│   │   ├── historico_erros_ruidoso.csv
│   │   ├── tempo_execucao_original.txt
│   │   ├── tempo_execucao_ruidoso.txt
│   │   ├── grafico_analise_completa.png
│   │   ├── original/
│   │   │   ├── saidas_teste.csv
│   │   │   ├── matriz_confusao_teste.csv
│   │   │   ├── matriz_confusao_teste.png
│   │   │   └── resumo_teste_original.txt
│   │   └── ruidoso/
│   │       ├── saidas_ruidoso.csv
│   │       ├── matriz_confusao_ruidoso.csv
│   │       ├── matriz_confusao_ruidoso.png
│   │       ├── resumo_teste_ruidoso.txt
│   │       ├── X_ruidoso.txt
│   │       └── Y_ruidoso.txt
│   ├── exp_2_h15_a0.01_e1000_err0.01/
│   │   └── ... (mesma estrutura)
│   └── ...
├── caso_1/                                # Experimentos do caso binário (letras com buraco)
│   └── ...
└── caso_2/                                # Experimentos do caso binário (letras com curva)
    └── ...
```

### Pelo script de melhor combinação (`melhor_combinacao.py`)

| Diretório gerado | Caso |
|-----------------|------|
| `melhor_combinacao_palavra/` | Caso 0 — Multiclasse (A-Z) |
| `melhor_combinacao_letra_com_buraco/` | Caso 1 — Binário (letras com buraco) |
| `melhor_combinacao_letra_curvada/` | Caso 2 — Binário (letras com curva) |

Cada diretório contém os arquivos do experimento campeão + um arquivo `hiperparametros_vencedores_caso_X.txt` com o resumo final.

### Pelo script de pior combinação (`pior_combinacao.py`)

| Diretório gerado | Caso |
|-----------------|------|
| `pior_combinacao_palavra/` | Caso 0 — Multiclasse (A-Z) |
| `pior_combinacao_letra_com_buraco/` | Caso 1 — Binário (letras com buraco) |
| `pior_combinacao_letra_curvada/` | Caso 2 — Binário (letras com curva) |

Cada diretório contém os arquivos do experimento com pior desempenho + um arquivo `hiperparametros_pior_caso_X.txt` com o resumo de análise contrastante.

### Pelo script de gerar tabelas de tempo médio de execução (`gerar_tabelas_tempo.py`)

Arquivo `Tabelas_Tempo.md` gerado na raiz contendo tabelas relacionando o tempo de execução com cada uma das variações de hiperparâmetros.

---

## Mapeamento: Requisitos da Entrega × Arquivos de Saída

A especificação do trabalho exige os seguintes artefatos de saída. Abaixo, cada requisito é associado ao(s) arquivo(s) correspondente(s):

| # | Requisito da Especificação | Arquivo(s) de Saída |
|---|---------------------------|---------------------|
| 1 | Hiperparâmetros finais da arquitetura e de inicialização | `hiperparametros.txt` |
| 2 | Pesos iniciais da rede | `pesos_iniciais.txt` |
| 3 | Pesos finais da rede | `pesos_finais.txt` |
| 4 | Erro cometido pela rede em cada iteração do treinamento | `historico_erros_original.csv` + `historico_erros_ruidoso.csv` |
| 5 | Saídas produzidas pela rede para cada dado de teste | `original/saidas_teste.csv` |
| 6 | Gráfico de comportamento de erros | `grafico_analise_completa.png` |
| 7 | Matriz de confusão | `original/matriz_confusao_teste.png` + `ruidoso/matriz_confusao_ruidoso.png`  (em PNG com labels) |
| 8 | Teste com variação ruidosa dos dados | `ruidoso/resumo_teste_ruidoso.txt` + `ruidoso/saidas_ruidoso.csv` |
| 9 | Base de dados ruidosa (com ruído) | `X_ruidoso.txt` + `Y_ruidoso.txt` |
| 10 | Código bem documentado com nomenclatura da aula | Comentários em todos os `.py` referenciando slides |

---

## Estrutura da Rede

```
Entrada (120 neurônios)  →  Camada Escondida (p neurônios)  →  Saída (m neurônios)
       Xi, i=1..120              Zj, j=1..p                     Yk, k=1..m
       
       Pesos: v, v0             Pesos: w, w0
       (entrada→escondida)      (escondida→saída)
```

- **Caso 0 (Multiclasse):** m = 26 neurônios de saída (um por letra A-Z)
- **Caso 1 (Buraco):** m = 2 neurônios de saída (tem buraco / não tem)
- **Caso 2 (Curva):** m = 2 neurônios de saída (tem curva / não tem)
