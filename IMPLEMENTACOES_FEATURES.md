# RESUMO DAS IMPLEMENTAÇÕES DE FEATURES

## 1️⃣ Visualização de Matriz de Confusão em PNG

### Modificações em `mlp_avaliacao.py`:
- ✅ Adicionado import opcional de `seaborn` para melhor aparência
- ✅ Criada função `gerar_imagem_matriz_confusao(matriz, caminho_png, caso)`
  - Gera heatmap colorido com valores anotados
  - Labels automáticos das classes (A-Z para caso 0, 0-1 para binários)
  - Suporta fallback para matplotlib se seaborn não estiver disponível
  - Tamanho adaptativo baseado no número de classes
  - Salva com DPI 150 para boa qualidade

### Integração em `main.py`:
- ✅ Importada nova função `gerar_imagem_matriz_confusao`
- ✅ Chamada para matriz de teste: `gerar_imagem_matriz_confusao(matriz_teste, "original/matriz_confusao_teste.png", caso=caso)`
- ✅ Chamada para matriz ruidosa: `gerar_imagem_matriz_confusao(matriz_ruidoso, "ruidoso/matriz_confusao_ruidoso.png", caso=caso)`

### Saídas:
- `original/matriz_confusao_teste.png` → Matriz de confusão do teste original
- `ruidoso/matriz_confusao_ruidoso.png` → Matriz de confusão do teste ruidoso

---

## 2️⃣ Gráficos 3D de Análise de Tempo

### Novo arquivo: `plot_tempos_3d.py`
- ✅ Coleta dados do `resultados_mlp/resumo_geral_experimentos.csv`
- ✅ Gera 4 gráficos 3D **por caso** (12 gráficos no total)

#### Gráficos Gerados:
1. **Neurônios Ocultos vs Épocas vs Tempo** → `tempo_3d_neurons_vs_epochs.png`
2. **Taxa Aprendizado vs Épocas vs Tempo** → `tempo_3d_alpha_vs_epochs.png`
3. **Limiar Erro Mínimo vs Épocas vs Tempo** → `tempo_3d_erro_vs_epochs.png`
4. **Max Épocas vs Épocas vs Tempo** → `tempo_3d_maxepocas_vs_epochs.png`

### Estrutura de Saída:
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

### Características dos Gráficos:
- 📊 Scatter 3D com cores baseadas na intensidade do tempo (escala viridis)
- 🎨 Colorbar indicando escala de tempo
- 📐 Ângulo de visualização otimizado (elev=20, azim=45)
- 💾 Resolução 150 DPI para boa qualidade de impressão

---

## 3️⃣ Extração da Pior Combinação de Hiperparâmetros

### Novo arquivo: `pior_combinacao.py`
- ✅ Funcionalidade espelhada de `melhor_combinacao.py`
- ✅ **Diferença**: Procura pela MENOR acurácia em vez da maior
- ✅ Critério de seleção: `(acurácia_teste crescente, erro_final decrescente)`

### Pastas Geradas (por caso):
1. **`pior_combinacao_palavra`** → Caso 0 (Multiclasse A-Z)
2. **`pior_combinacao_letra_com_buraco`** → Caso 1 (Binário)
3. **`pior_combinacao_letra_curvada`** → Caso 2 (Binário)

### Conteúdo de cada pasta (mesma estrutura dos experimentos):
- `hiperparametros.txt`
- `pesos_iniciais.txt` / `pesos_finais.txt`
- `historico_erros_original.csv` / `historico_erros_ruidoso.csv`
- `tempo_execucao_original.txt` / `tempo_execucao_ruidoso.txt`
- `grafico_analise_completa.png`
- `original/` (saidas_teste.csv, matriz_confusao_teste.csv/.png, resumo_teste_original.txt)
- `ruidoso/` (saidas_ruidoso.csv, matriz_confusao_ruidoso.csv/.png, resumo_teste_ruidoso.txt, X_ruidoso.txt, Y_ruidoso.txt)
- `hiperparametros_pior_caso_X.txt` (novo arquivo de resumo)

---

## 4️⃣ Separação de Dados Originais vs Ruidosos

### Nova estrutura por experimento:
```
exp_N_hX_aY_eZ_errW/
├── hiperparametros.txt              # Escopo geral
├── pesos_iniciais.txt               # Escopo geral
├── pesos_finais.txt                 # Escopo geral
├── historico_erros_original.csv     # Erro por época (treino/validação)
├── historico_erros_ruidoso.csv      # Erro ruidoso por época
├── tempo_execucao_original.txt      # Tempo: treino + validação + teste
├── tempo_execucao_ruidoso.txt       # Tempo: apenas inferência ruidosa
├── grafico_analise_completa.png     # Convergência + pesos (substitui curva_erro)
├── original/                        # Resultados do teste ORIGINAL
│   ├── saidas_teste.csv
│   ├── matriz_confusao_teste.csv
│   ├── matriz_confusao_teste.png
│   └── resumo_teste_original.txt
└── ruidoso/                         # Resultados do teste RUIDOSO
    ├── saidas_ruidoso.csv
    ├── matriz_confusao_ruidoso.csv
    ├── matriz_confusao_ruidoso.png
    ├── resumo_teste_ruidoso.txt
    ├── X_ruidoso.txt
    └── Y_ruidoso.txt
```

### Mudanças de nomenclatura:
- `autoral` → `ruidoso` (mais claro e sem ambiguidade)
- `historico_erros.csv` → `historico_erros_original.csv` + `historico_erros_ruidoso.csv`
- `tempo_execucao.txt` → `tempo_execucao_original.txt` + `tempo_execucao_ruidoso.txt`
- `curva_erro.png` → removido (já incluído no `grafico_analise_completa.png`)
- `saidas_letras_autoral.csv` → `ruidoso/saidas_ruidoso.csv`
- `resumo_teste_autoral.txt` → `ruidoso/resumo_teste_ruidoso.txt`
- `matriz_confusao_autoral.*` → `ruidoso/matriz_confusao_ruidoso.*`

---

## 5️⃣ Múltiplas Réplicas Ruidosas (130 amostras)

### Modificações em `gerar_ruido.py`:
- ✅ Gera **5 réplicas** de cada letra com ruído aleatório diferente
- ✅ Total: 26 letras × 5 réplicas = **130 amostras** (compatível com teste original)
- ✅ Taxa de ruído: 15% dos pixels invertidos por amostra
- ✅ Cada réplica tem ruído independente para variabilidade
- ✅ Salva como `X_ruidoso.txt` e `Y_ruidoso.txt`

### Vantagens:
- Tamanho igual ao conjunto de teste original (130 amostras)
- Comparação justa entre acurácia original e ruidosa
- Melhor representatividade estatística

---

## 6️⃣ Separação de Tempos de Execução

### Tempo Original (`tempo_execucao_original.txt`):
- Inclui: treinamento + validação (early stopping) + teste original
- Representa o tempo real de construção do modelo

### Tempo Ruidoso (`tempo_execucao_ruidoso.txt`):
- Inclui: apenas inferência e avaliação no conjunto ruidoso
- Separado pois é um teste "extra" (não faz parte do pipeline de treinamento)

### No resumo geral:
- CSV agora tem duas colunas: `Tempo_Original_Segundos` e `Tempo_Ruidoso_Segundos`

---

## 7️⃣ Fluxo Completo de Execução

```
run_experimentos.sh
├── Loop: 972 experimentos
│   ├── main.py (treino + avaliação + teste original + teste ruidoso)
│   ├── Organiza em subpastas original/ e ruidoso/
│   └── Separa tempos de execução
├── Loop por caso:
│   ├── melhor_combinacao.py (copia estrutura completa com subpastas)
│   └── pior_combinacao.py (copia estrutura completa com subpastas)
├── analise_geral_graficos.py (gráficos 2D de tempo)
└── plot_tempos_3d.py (12 gráficos 3D)
```

---

## 8️⃣ Dependências

- `numpy` (obrigatório)
- `matplotlib` (obrigatório)
- `seaborn` (opcional, para melhor visualização de matrizes)

---

## 📋 Checklist de Validação

- ✅ Todos os `.py` compilam sem erros
- ✅ `run_experimentos.sh` sintaxe válida
- ✅ Estrutura original/ e ruidoso/ criada corretamente
- ✅ Tempos separados (original vs ruidoso)
- ✅ Históricos de erro separados
- ✅ Nomenclatura "autoral" substituída por "ruidoso"
- ✅ `curva_erro.png` removido (substituído por `grafico_analise_completa.png`)
- ✅ 130 amostras ruidosas (5 réplicas × 26 letras)
- ✅ `resumo_teste_original.txt` criado para dados originais
- ✅ Scripts de análise (melhor/pior) compatíveis com nova e antiga estrutura
- ✅ Documentação (README.md, GUIA_EXECUCAO.md) atualizada
