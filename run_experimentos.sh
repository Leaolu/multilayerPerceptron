#!/bin/bash
ROXO='\033[0;35m'
VERMELHO='\033[0;31m'
AMARELO='\033[1;33m'
VERDE='\033[0;32m'
BRANCO='\033[1;37m'
NC='\033[0m' # Sem Cor (Reset)

# Nome do script python principal
PYTHON_SCRIPT="main.py"

# Intervalos coerentes para os testes
CASOS=(0 1 2)
CAMADAS_ESCONDIDAS=(15 30 50)
ALPHAS=(0.01 0.05 0.1 0.2)
MAX_EPOCAS=(1000 1500 2000)
ERROS_MINIMOS=(0.05 0.01 0.005)

echo -e "${ROXO}==========================================================${NC}"
echo -e "${ROXO}    Iniciando a Bateria de Experimentos Automatizados     ${NC}"
echo -e "${ROXO}==========================================================${NC}"

# Cria o diretório principal de resultados
DIRETORIO_RAIZ_RESULTADOS="resultados_mlp"
mkdir -p "$DIRETORIO_RAIZ_RESULTADOS"

# Criar o cabeçalho do arquivo de resumo global (ficará na raiz)
ARQUIVO_RESUMO="${DIRETORIO_RAIZ_RESULTADOS}/resumo_geral_experimentos.csv"
echo "Experimento,Caso,Camada_Oculta,Alpha,Max_Epocas,Erro_Minimo,Tempo_Original_Segundos,Tempo_Ruidoso_Segundos" > "$ARQUIVO_RESUMO"

TOTAL_EXP=$(( ${#CASOS[@]} * ${#CAMADAS_ESCONDIDAS[@]} * ${#ALPHAS[@]} * ${#MAX_EPOCAS[@]} * ${#ERROS_MINIMOS[@]} ))
CONTADOR=1

# NOVO LOOP: Itera sobre os Casos primeiro
for c in "${CASOS[@]}"; do

    # CRIACAO DO DIRETORIO ESPECIFICO DO CASO
    DIRETORIO_CASO="${DIRETORIO_RAIZ_RESULTADOS}/caso_${c}"
    mkdir -p "$DIRETORIO_CASO"

    for h in "${CAMADAS_ESCONDIDAS[@]}"; do
        for a in "${ALPHAS[@]}"; do
            for e in "${MAX_EPOCAS[@]}"; do
                for err in "${ERROS_MINIMOS[@]}"; do

                    # Identifica o nome legivel do caso para imprimir na tela
                    if [ "$c" -eq 0 ]; then NOME_CASO="Multiclasse (26)"
                    elif [ "$c" -eq 1 ]; then NOME_CASO="Binario (Buracos)"
                    elif [ "$c" -eq 2 ]; then NOME_CASO="Binario (Curvas)"
                    fi

                    echo ""
                    echo -e "${ROXO}--------------------------------------------------------${NC}"
                    echo -e "${ROXO} Experimento ${VERDE}${CONTADOR}${ROXO} de ${VERDE}${TOTAL_EXP}${NC} | ${AMARELO}Caso: $NOME_CASO${NC}"
                    echo -e "${BRANCO} Run: Caso=$c | Hidden=$h | Alpha=$a | MaxEpocas=$e | ErroMin=$err${NC}"
                    echo -e "${ROXO}--------------------------------------------------------${NC}"

                    # Nome unico para a pasta deste experimento DENTRO da pasta do caso
                    PASTA_EXP="${DIRETORIO_CASO}/exp_${CONTADOR}_h${h}_a${a}_e${e}_err${err}"
                    mkdir -p "$PASTA_EXP"

                    # Executa o Python passando as variacoes atuais E O ARGUMENTO --caso
                    python "$PYTHON_SCRIPT" --n_escondidos "$h" --alpha "$a" --max_epocas "$e" --erro_minimo "$err" --caso "$c"

                    # Resgata os tempos para o resumo geral
                    if [ -f "tempo_execucao_original.txt" ]; then
                        TEMPO_ORIGINAL=$(cat "tempo_execucao_original.txt")
                    else
                        TEMPO_ORIGINAL="0.0"
                    fi

                    if [ -f "tempo_execucao_ruidoso.txt" ]; then
                        TEMPO_RUIDOSO=$(cat "tempo_execucao_ruidoso.txt")
                    else
                        TEMPO_RUIDOSO="0.0"
                    fi

                    # Alimenta o arquivo de resumo geral
                    echo "exp_${CONTADOR},${c},${h},${a},${e},${err},${TEMPO_ORIGINAL},${TEMPO_RUIDOSO}" >> "$ARQUIVO_RESUMO"

                    echo -e "${AMARELO}-> Isolando arquivos gerados na pasta do experimento...${NC}"

                    # Criar subpastas original e ruidoso
                    mkdir -p "$PASTA_EXP/original"
                    mkdir -p "$PASTA_EXP/ruidoso"

                    # 1. MOVE arquivos na raiz da pasta de experimento (escopo geral)
                    [ -f "hiperparametros.txt" ] && mv "hiperparametros.txt" "$PASTA_EXP/"
                    [ -f "pesos_iniciais.txt" ]  && mv "pesos_iniciais.txt" "$PASTA_EXP/"
                    [ -f "pesos_finais.txt" ]    && mv "pesos_finais.txt" "$PASTA_EXP/"
                    [ -f "historico_erros_original.csv" ] && mv "historico_erros_original.csv" "$PASTA_EXP/"
                    [ -f "historico_erros_ruidoso.csv" ] && mv "historico_erros_ruidoso.csv" "$PASTA_EXP/"
                    [ -f "tempo_execucao_original.txt" ] && mv "tempo_execucao_original.txt" "$PASTA_EXP/"
                    [ -f "tempo_execucao_ruidoso.txt" ] && mv "tempo_execucao_ruidoso.txt" "$PASTA_EXP/"
                    [ -f "grafico_analise_completa.png" ] && mv "grafico_analise_completa.png" "$PASTA_EXP/"

                    # 2. MOVE arquivos do teste ORIGINAL para subpasta original/
                    [ -f "original/saidas_teste.csv" ] && mv "original/saidas_teste.csv" "$PASTA_EXP/original/"
                    [ -f "original/matriz_confusao_teste.csv" ] && mv "original/matriz_confusao_teste.csv" "$PASTA_EXP/original/"
                    [ -f "original/matriz_confusao_teste.png" ] && mv "original/matriz_confusao_teste.png" "$PASTA_EXP/original/"
                    [ -f "original/resumo_teste_original.txt" ] && mv "original/resumo_teste_original.txt" "$PASTA_EXP/original/"

                    # 3. MOVE arquivos do teste RUIDOSO para subpasta ruidoso/
                    [ -f "ruidoso/saidas_ruidoso.csv" ] && mv "ruidoso/saidas_ruidoso.csv" "$PASTA_EXP/ruidoso/"
                    [ -f "ruidoso/matriz_confusao_ruidoso.csv" ] && mv "ruidoso/matriz_confusao_ruidoso.csv" "$PASTA_EXP/ruidoso/"
                    [ -f "ruidoso/matriz_confusao_ruidoso.png" ] && mv "ruidoso/matriz_confusao_ruidoso.png" "$PASTA_EXP/ruidoso/"
                    [ -f "ruidoso/resumo_teste_ruidoso.txt" ] && mv "ruidoso/resumo_teste_ruidoso.txt" "$PASTA_EXP/ruidoso/"

                    # 4. COPIA os arquivos da base de dados ruidosa para referencia
                    [ -f "X_ruidoso.txt" ] && cp "X_ruidoso.txt" "$PASTA_EXP/ruidoso/"
                    [ -f "Y_ruidoso.txt" ] && cp "Y_ruidoso.txt" "$PASTA_EXP/ruidoso/"

                    CONTADOR=$((CONTADOR + 1))
                done
            done
        done
    done
    echo -e "${VERDE}-> Extraindo a melhor combinacao para o Caso ${c}...${NC}"
    python melhor_combinacao.py --caso "$c"

    echo -e "${VERMELHO}-> Extraindo a pior combinacao para o Caso ${c}...${NC}"
    python pior_combinacao.py --caso "$c"
done

echo ""
echo -e "${AMARELO}==========================================================${NC}"
echo -e "${AMARELO} Gerando graficos finais de analise para a apresentacao...${NC}"
echo -e "${AMARELO}==========================================================${NC}"
python analise_geral_graficos.py

echo ""
echo -e "${AMARELO}==========================================================${NC}"
echo -e "${AMARELO} Gerando graficos 3D de analise de tempo...${NC}"
echo -e "${AMARELO}==========================================================${NC}"
python plot_tempos_3d.py

echo ""
echo -e "${VERDE}==========================================================${NC}"
echo -e "${VERDE} Concluido! O arquivo historico global esta em: ${BRANCO}$ARQUIVO_RESUMO${NC}"
echo -e "${VERDE}==========================================================${NC}"
