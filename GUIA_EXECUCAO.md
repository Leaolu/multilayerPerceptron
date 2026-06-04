# 🚀 Guia Completo de Execução - Multilayer Perceptron (MLP)

Este documento fornece instruções passo a passo para executar o projeto **do zero** e gerar todos os arquivos de saída atualizados.

---

## 📋 Pré-requisitos

- **Python 3.7+** (recomendado 3.9 ou superior)
- **Git** (para clonar o repositório, se aplicável)
- **pip** (gerenciador de pacotes Python)
- **Bash/Shell** (para executar scripts .sh no Linux/macOS ou WSL no Windows)

---

## 🛠️ Passo 1: Preparar o Ambiente

### 1.1 Verificar Instalação do Python

```bash
python --version
```

**Saída esperada:** `Python 3.x.x` (version 3.7 or higher)

### 1.2 Clonar/Acessar o Repositório

Se estiver usando Git:
```bash
git clone <url-do-repositorio>
cd multilayerPerceptron
```

Ou simplesmente acesse a pasta do projeto:
```bash
cd /caminho/para/multilayerPerceptron
```

---

## 📦 Passo 2: Instalar Dependências

### 2.1 Criar Ambiente Virtual (Recomendado)

#### No Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

#### No Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### No Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2.2 Instalar Dependências do requirements.txt

```bash
pip install -r requirements.txt
```

**Saída esperada:**
```
Successfully installed numpy-1.xx.x matplotlib-3.xx.x seaborn-0.xx.x
```

### 2.3 Verificar Instalação

```bash
python -c "import numpy, matplotlib, seaborn; print('✓ Todas as dependências instaladas com sucesso!')"
```

---

## 🗂️ Passo 3: Verificar Estrutura do Projeto

Confirme que os seguintes arquivos existem:

```
multilayerPerceptron/
├── main.py                          # Script principal
├── mlp_init.py                      # Inicialização
├── mlp_forward.py                   # Feedforward
├── mlp_backward.py                  # Backpropagation
├── mlp_treino.py                    # Loop de treinamento
├── mlp_avaliacao.py                 # Avaliação e visualização
├── gerar_ruido.py                   # Geração de base ruidosa
├── melhor_combinacao.py             # Extração do melhor modelo
├── pior_combinacao.py               # Extração do pior modelo
├── plot_tempos_3d.py                # Gráficos 3D
├── analise_geral_graficos.py        # Análise geral
├── run_experimentos.sh              # Script de automação
├── requirements.txt                 # Dependências
├── X.txt                            # Dados de entrada (features)
├── Y_letra.txt                      # Rótulos
├── README.md                        # Documentação
└── [outros arquivos]
```

---

## 🎯 Passo 4: Gerar Base Ruidosa

Este passo **deve ser executado ANTES** de qualquer treinamento, pois a base ruidosa é necessária:

```bash
python gerar_ruido.py
```

**Saída esperada:**
```
╔══════════════════════════════════════════════════════════════════════╗
║              Gerando Base de Dados Ruidosa                          ║
╚══════════════════════════════════════════════════════════════════════╝

1️⃣  Carregando base original...
2️⃣  Configuração:
    - Letras únicas: 26
    - Réplicas por letra: 5
    - Taxa de ruído por instância: 15.0%
    - Total de amostras ruidosas: 130
3️⃣  Gerando 5 instâncias ruidosas de cada letra (A-Z)...
4️⃣  Salvando dados ruidosos...
    ✓ X_ruidoso.txt salvo
    ✓ Y_ruidoso.txt salvo
✅ Base de dados ruidosa gerada com sucesso!
```

**Arquivos criados:**
- `X_ruidoso.txt` → 130 amostras (5 réplicas × 26 letras) com 15% ruído
- `Y_ruidoso.txt` → Rótulos das amostras ruidosas (A-Z, cada uma 5×)

---

## 🏃 Passo 5: Executar um Treinamento Individual (Opcional)

Se desejar testar com uma configuração específica:

### 5.1 Treinamento do Caso 0 (Multiclasse A-Z)

```bash
python main.py --n_escondidos 50 --alpha 0.01 --max_epocas 1500 --erro_minimo 0.005 --paciencia 20 --caso 0
```

**Argumentos:**
- `--n_escondidos`: Quantidade de neurônios na camada oculta (padrão: 20)
- `--alpha`: Taxa de aprendizado (padrão: 0.1)
- `--max_epocas`: Número máximo de épocas (padrão: 1500)
- `--erro_minimo`: Erro mínimo desejado (padrão: 0.01)
- `--paciencia`: Épocas de paciência para early stopping (padrão: 20)
- `--caso`: Tipo de problema (0=Multiclasse, 1=Binário Buraco, 2=Binário Curva) (padrão: 0)

### 5.2 Saídas do Treinamento Individual

```
📁 Arquivos gerados no diretório raiz:
├── hiperparametros.txt                  # Configuração usada
├── pesos_iniciais.txt                   # Pesos antes do treinamento
├── pesos_finais.txt                     # Pesos após treinamento
├── historico_erros_original.csv         # Erro de treino/validação por época
├── historico_erros_ruidoso.csv          # Erro no conjunto ruidoso por época
├── tempo_execucao_original.txt          # Tempo: treino + validação + teste original
├── tempo_execucao_ruidoso.txt           # Tempo: apenas inferência/avaliação ruidosa
├── grafico_analise_completa.png         # Análise completa (convergência + pesos)
├── original/                            # Resultados do teste ORIGINAL
│   ├── saidas_teste.csv                 # Predições vs reais (teste)
│   ├── matriz_confusao_teste.csv        # Matriz confusão (CSV)
│   ├── matriz_confusao_teste.png        # Matriz confusão (IMAGEM)
│   └── resumo_teste_original.txt        # Resumo de performance original
└── ruidoso/                             # Resultados do teste RUIDOSO
    ├── saidas_ruidoso.csv               # Predições vs reais (ruidoso)
    ├── matriz_confusao_ruidoso.csv      # Matriz confusão (CSV)
    ├── matriz_confusao_ruidoso.png      # Matriz confusão (IMAGEM)
    └── resumo_teste_ruidoso.txt         # Resumo de performance ruidosa
```

---

## 🚀 Passo 6: Executar Bateria Completa de Experimentos

Este é o comando **principal** que executa TODOS os 972 experimentos:

### 6.1 No Linux/macOS/WSL:

```bash
chmod +x run_experimentos.sh
./run_experimentos.sh
```

### 6.2 No Windows (com WSL):

```bash
# Se estiver usando WSL
./run_experimentos.sh
```

### 6.3 No Windows (sem WSL, apenas Python):

```bash
# Execute manualmente em Python (mais lento):
python analise_geral_graficos.py
python plot_tempos_3d.py
```

Ou crie um script batch equivalente.

### 6.4 O que Acontece Durante a Execução

O `run_experimentos.sh` automaticamente:

1. ✅ **Executa 972 treinamentos** com diferentes hiperparâmetros
   - 3 casos × 3 neurônios × 4 alphas × 3 épocas × 3 erros = 972 experimentos

2. ✅ **Cria estrutura de resultados:**
   ```
   resultados_mlp/
   ├── resumo_geral_experimentos.csv   # Índice de TODOS os experimentos
   ├── caso_0/                         # Caso Multiclasse
   │   ├── exp_1_h15_a0.01_e1000_err0.05/
   │   │   ├── hiperparametros.txt
   │   │   ├── pesos_iniciais.txt / pesos_finais.txt
   │   │   ├── historico_erros_original.csv
   │   │   ├── historico_erros_ruidoso.csv
   │   │   ├── tempo_execucao_original.txt
   │   │   ├── tempo_execucao_ruidoso.txt
   │   │   ├── grafico_analise_completa.png
   │   │   ├── original/
   │   │   │   ├── saidas_teste.csv
   │   │   │   ├── matriz_confusao_teste.csv / .png
   │   │   │   └── resumo_teste_original.txt
   │   │   └── ruidoso/
   │   │       ├── saidas_ruidoso.csv
   │   │       ├── matriz_confusao_ruidoso.csv / .png
   │   │       ├── resumo_teste_ruidoso.txt
   │   │       ├── X_ruidoso.txt
   │   │       └── Y_ruidoso.txt
   │   ├── exp_2_h15_a0.01_e1000_err0.01/
   │   └── ... (324 experimentos)
   ├── caso_1/                         # Caso Binário (Buraco)
   │   └── ... (324 experimentos)
   └── caso_2/                         # Caso Binário (Curva)
       └── ... (324 experimentos)
   ```

3. ✅ **Extrai melhores e piores modelos:**
   ```
   melhor_combinacao_palavra/              # Caso 0 - MELHOR
   melhor_combinacao_letra_com_buraco/     # Caso 1 - MELHOR
   melhor_combinacao_letra_curvada/        # Caso 2 - MELHOR
   pior_combinacao_palavra/                # Caso 0 - PIOR
   pior_combinacao_letra_com_buraco/       # Caso 1 - PIOR
   pior_combinacao_letra_curvada/          # Caso 2 - PIOR
   ```

4. ✅ **Gera gráficos de análise:**
   ```
   tempos_execucao/
   ├── caso_0/
   │   ├── tempo_3d_neurons_vs_epochs.png
   │   ├── tempo_3d_alpha_vs_epochs.png
   │   ├── tempo_3d_erro_vs_epochs.png
   │   └── tempo_3d_maxepocas_vs_epochs.png
   ├── caso_1/ (mesma estrutura)
   └── caso_2/ (mesma estrutura)
   ```

### 6.5 Tempo Esperado de Execução

**Tempo aproximado por máquina:**
- CPU moderna (4+ cores): **30-60 minutos**
- CPU mais lenta: **1-2 horas**

---

## 📊 Passo 7: Analisar Resultados

### 7.1 Verificar Resumo Geral

```bash
# No Linux/macOS:
head -20 resultados_mlp/resumo_geral_experimentos.csv

# No Windows:
type resultados_mlp\resumo_geral_experimentos.csv | more
```

**Estrutura:**
```
Experimento,Caso,Camada_Oculta,Alpha,Max_Epocas,Erro_Minimo,Tempo_Original_Segundos,Tempo_Ruidoso_Segundos
exp_1,0,15,0.01,1000,0.05,31.5021,0.0342
exp_2,0,15,0.01,1000,0.01,37.6267,0.0289
...
```

### 7.2 Comparar Melhor vs Pior

```bash
# Melhor desempenho - Caso 0
cat melhor_combinacao_palavra/hiperparametros_vencedores_caso_0.txt

# Pior desempenho - Caso 0
cat pior_combinacao_palavra/hiperparametros_pior_caso_0.txt
```

### 7.3 Visualizar Gráficos 3D

Abra os arquivos PNG da pasta `tempos_execucao/`:
```
tempos_execucao/caso_0/
├── tempo_3d_neurons_vs_epochs.png       # Neurônios vs Épocas
├── tempo_3d_alpha_vs_epochs.png         # Alpha vs Épocas
├── tempo_3d_erro_vs_epochs.png          # Erro Mínimo vs Épocas
└── tempo_3d_maxepocas_vs_epochs.png     # Max Épocas vs Épocas
```

---

## 🔄 Passo 8: Executar Apenas Análises Específicas (Pós-Experimentos)

Se os experimentos já foram executados e você quer apenas gerar análises:

### 8.1 Extrair Melhor Combinação para um Caso

```bash
python melhor_combinacao.py --caso 0    # Caso 0: Multiclasse
python melhor_combinacao.py --caso 1    # Caso 1: Binário (Buraco)
python melhor_combinacao.py --caso 2    # Caso 2: Binário (Curva)
```

### 8.2 Extrair Pior Combinação para um Caso

```bash
python pior_combinacao.py --caso 0      # Caso 0: Multiclasse
python pior_combinacao.py --caso 1      # Caso 1: Binário (Buraco)
python pior_combinacao.py --caso 2      # Caso 2: Binário (Curva)
```

### 8.3 Gerar Gráficos 3D

```bash
python plot_tempos_3d.py
# Gera 12 gráficos PNG em tempos_execucao/caso_X/
```

### 8.4 Gerar Gráficos Gerais de Análise

```bash
python analise_geral_graficos.py
# Gera gráficos de tempo vs hiperparâmetros em resultados_mlp/
```

---

## 🧹 Passo 9: Limpar e Recomeçar (Opcional)

Se desejar executar o projeto do zero novamente, limpe os resultados antigos:

### 9.1 No Linux/macOS:

```bash
# Remove todos os resultados anteriores
rm -rf resultados_mlp/
rm -rf melhor_combinacao_*/
rm -rf pior_combinacao_*/
rm -rf tempos_execucao/
rm -f X_ruidoso.txt Y_ruidoso.txt
rm -f *.png historico_erros_*.csv tempo_execucao_*.txt
rm -rf original/ ruidoso/
```

### 9.2 No Windows (CMD):

```cmd
rmdir /s /q resultados_mlp
rmdir /s /q melhor_combinacao_*
rmdir /s /q pior_combinacao_*
rmdir /s /q tempos_execucao
del X_ruidoso.txt Y_ruidoso.txt
del *.png
```

Depois, retorne ao **Passo 4** (Gerar Base Ruidosa).

---

## 📁 Estrutura Final de Saídas

Após executar `run_experimentos.sh`, você terá:

```
multilayerPerceptron/
├── [arquivos de código]
├── [arquivos de dados originais]
├── X_ruidoso.txt                           # Base ruidosa gerada
├── Y_ruidoso.txt
│
├── resultados_mlp/                         # Resultados de todos os 972 experimentos
│   ├── resumo_geral_experimentos.csv
│   ├── caso_0/
│   │   ├── exp_1_h15_a0.01_e1000_err0.05/
│   │   │   ├── hiperparametros.txt
│   │   │   ├── pesos_iniciais.txt
│   │   │   ├── pesos_finais.txt
│   │   │   ├── historico_erros_original.csv
│   │   │   ├── historico_erros_ruidoso.csv
│   │   │   ├── tempo_execucao_original.txt
│   │   │   ├── tempo_execucao_ruidoso.txt
│   │   │   ├── grafico_analise_completa.png
│   │   │   ├── original/
│   │   │   │   ├── saidas_teste.csv
│   │   │   │   ├── matriz_confusao_teste.csv
│   │   │   │   ├── matriz_confusao_teste.png
│   │   │   │   └── resumo_teste_original.txt
│   │   │   └── ruidoso/
│   │   │       ├── saidas_ruidoso.csv
│   │   │       ├── matriz_confusao_ruidoso.csv
│   │   │       ├── matriz_confusao_ruidoso.png
│   │   │       ├── resumo_teste_ruidoso.txt
│   │   │       ├── X_ruidoso.txt
│   │   │       └── Y_ruidoso.txt
│   │   ├── exp_2_h15_a0.01_e1000_err0.01/
│   │   └── ... (321 mais)
│   ├── caso_1/ (324 experimentos)
│   └── caso_2/ (324 experimentos)
│
├── melhor_combinacao_palavra/              # MELHOR Caso 0
│   ├── [mesma estrutura do experimento]
│   └── hiperparametros_vencedores_caso_0.txt
│
├── melhor_combinacao_letra_com_buraco/     # MELHOR Caso 1
├── melhor_combinacao_letra_curvada/        # MELHOR Caso 2
│
├── pior_combinacao_palavra/                # PIOR Caso 0
│   ├── [mesma estrutura do experimento]
│   └── hiperparametros_pior_caso_0.txt
│
├── pior_combinacao_letra_com_buraco/       # PIOR Caso 1
├── pior_combinacao_letra_curvada/          # PIOR Caso 2
│
└── tempos_execucao/                        # Gráficos 3D
    ├── caso_0/
    │   ├── tempo_3d_neurons_vs_epochs.png
    │   ├── tempo_3d_alpha_vs_epochs.png
    │   ├── tempo_3d_erro_vs_epochs.png
    │   └── tempo_3d_maxepocas_vs_epochs.png
    ├── caso_1/ (mesma estrutura)
    └── caso_2/ (mesma estrutura)
```

---

## ✅ Checklist Final

- [ ] Python 3.7+ instalado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Base ruidosa gerada (`python gerar_ruido.py`)
- [ ] Experimentos executados (`./run_experimentos.sh`)
- [ ] Gráficos 3D atualizados (`python plot_tempos_3d.py`)
- [ ] Análises extraídas (`python melhor_combinacao.py --caso X` e `python pior_combinacao.py --caso X`)
- [ ] Resultados verificados em `resultados_mlp/` e `melhor_combinacao_*/` e `pior_combinacao_*/`

---

## 🆘 Resolução de Problemas

### Problema: "matplotlib não encontrado"
```bash
pip install matplotlib
```

### Problema: "bash: ./run_experimentos.sh: Permission denied"
```bash
chmod +x run_experimentos.sh
./run_experimentos.sh
```

### Problema: "FileNotFoundError: X_ruidoso.txt não encontrado"
```bash
# Execute antes:
python gerar_ruido.py
```

### Problema: "ModuleNotFoundError: No module named 'seaborn'"
```bash
pip install seaborn
# Nota: seaborn é opcional; matplotlib fornecerá fallback automático
```

### Problema: Ambiente virtual não ativado
```bash
# Linux/macOS:
source venv/bin/activate

# Windows CMD:
venv\Scripts\activate
```

---

## 📞 Suporte

Para questões específicas sobre cada módulo, veja:
- `README.md` — Documentação geral do projeto
- `IMPLEMENTACOES_FEATURES.md` — Detalhes das features implementadas
- Comentários nos arquivos `.py` — Referências aos slides da aula

---

**Desenvolvido por:** Grupo de IA - EACH-USP  
**Data:** 2026  
**Versão:** 3.0 (com separação original/ruidoso e múltiplas réplicas)
