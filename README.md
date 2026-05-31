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
| `main.py` | Script principal que orquestra o pipeline completo: carregamento de dados, inicialização da rede, treinamento com parada antecipada, avaliação no conjunto de teste e teste com base autoral. |
| `mlp_init.py` | Funções de inicialização: geração de pesos aleatórios (Passo 0 - Fausett), carregamento dos datasets de caracteres (multiclasse e binário), divisão treino/validação/teste, e persistência de pesos. |
| `mlp_forward.py` | Implementação do estágio feedforward (Passos 3, 4 e 5 - Fausett): função sigmoide, propagação pela camada escondida e camada de saída, e predição via index do maior elemento. |
| `mlp_backward.py` | Implementação da backpropagation do erro (Passos 6 e 7 - Fausett): cálculo dos gradientes locais (δ) para camada de saída e escondida, e cálculo das correções de pesos. |
| `mlp_treino.py` | Loop de treinamento completo com atualização de pesos (Passo 8 - Fausett), cálculo do erro quadrático médio, e lógica de parada antecipada monitorando o erro de validação. |
| `mlp_avaliacao.py` | Funções de avaliação: matriz de confusão, cálculo de acurácia, plotagem da curva de erro (treino vs validação) e exportação dos resultados de teste em CSV. |
| `gerar_ruido.py` | Geração da base autoral: aplica ruído aleatório (inversão de 15% dos pixels) sobre um alfabeto completo para testar a robustez/generalização da rede treinada. |
| `melhor_combinacao.py` | Análise pós-experimentos: varre os resultados de todos os experimentos de um caso, identifica o modelo com melhor acurácia no teste e isola seus arquivos em diretório dedicado. |
| `run_experimentos.sh` | Script shell que automatiza a execução de todos os experimentos, variando hiperparâmetros (camadas ocultas, alpha, épocas, erro mínimo) para os 3 casos de classificação. |

---

## Como Executar

### Pré-requisitos

- Python 3.x
- NumPy
- Matplotlib

### Gerar base autoral (ruído)

```bash
python3 gerar_ruido.py
```

### Executar um treinamento individual

```bash
python3 main.py --n_escondidos 30 --alpha 0.1 --max_epocas 1500 --erro_minimo 0.01 --paciencia 20 --caso 0
```

### Executar bateria completa de experimentos

```bash
chmod +x run_experimentos.sh
./run_experimentos.sh
```

### Encontrar melhor combinação de um caso

```bash
python3 melhor_combinacao.py --caso 0
```

---

## Arquivos de Saída Gerados

### Por execução individual (`main.py`)

| Arquivo | Descrição |
|---------|-----------|
| `hiperparametros.txt` | Hiperparâmetros da arquitetura (entradas, ocultas, saídas, alpha, paciência, épocas, erro mínimo). |
| `pesos_iniciais.txt` | Pesos e bias iniciais da rede (antes do treinamento). |
| `pesos_finais.txt` | Pesos e bias finais da rede (após o treinamento). |
| `historico_erros.csv` | Erro quadrático médio de treino e validação em cada época. |
| `saidas_teste.csv` | Rótulo real e classe predita para cada amostra do conjunto de teste. |
| `resumo_teste_autoral.txt` | Erro médio e acurácia no conjunto autoral. |
| `saidas_letras_autoral.csv` | Rótulo real e classe predita para cada amostra do conjunto autoral. |
| `curva_erro.png` | Gráfico da curva de erro (treino vs validação) ao longo das épocas. |

### Pela bateria de experimentos (`run_experimentos.sh`)

O script gera a seguinte estrutura:

```
resultados_mlp/
├── resumo_geral_experimentos.csv          # Índice de todos os experimentos executados
├── caso_0/                                # Experimentos do caso multiclasse (A-Z)
│   ├── exp_1_h15_a0.01_e1000_err0.05/
│   │   ├── hiperparametros.txt
│   │   ├── pesos_iniciais.npz
│   │   ├── pesos_finais.npz
│   │   ├── historico_erros.csv
│   │   ├── saidas_teste.csv
│   │   ├── resumo_teste_autoral.txt
│   │   ├── saidas_letras_autoral.csv
│   │   ├── curva_erro_exp_1.png
│   │   ├── X_autoral.txt
│   │   └── Y_autoral.txt
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

---

## Mapeamento: Requisitos da Entrega × Arquivos de Saída

A especificação do trabalho exige os seguintes artefatos de saída. Abaixo, cada requisito é associado ao(s) arquivo(s) correspondente(s):

| # | Requisito da Especificação | Arquivo(s) de Saída |
|---|---------------------------|---------------------|
| 1 | Hiperparâmetros finais da arquitetura e de inicialização | `hiperparametros.txt` |
| 2 | Pesos iniciais da rede | `pesos_iniciais.txt` |
| 3 | Pesos finais da rede | `pesos_finais.txt` |
| 4 | Erro cometido pela rede em cada iteração do treinamento | `historico_erros.csv` |
| 5 | Saídas produzidas pela rede para cada dado de teste | `saidas_teste.csv` |
| 6 | Gráfico de comportamento de erros (vídeo) | `curva_erro.png` / `curva_erro_exp_N.png` |
| 7 | Matriz de confusão (vídeo) | Gerada via `mlp_avaliacao.py → matriz_confusao()` |
| 8 | Teste com variação autoral dos dados | `resumo_teste_autoral.txt` + `saidas_letras_autoral.csv` |
| 9 | Base de dados autoral (com ruído) | `X_autoral.txt` + `Y_autoral.txt` |
| 10 | Código bem documentado com nomenclatura da aula | Comentários em todos os `.py` referenciando slides do slide |

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
