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
echo "Experimento,Caso,Camada_Oculta,Alpha,Max_Epocas,Erro_Minimo" > "$ARQUIVO_RESUMO"

TOTAL_EXP=$(( ${#CASOS[@]} * ${#CAMADAS_ESCONDIDAS[@]} * ${#ALPHAS[@]} * ${#MAX_EPOCAS[@]} * ${#ERROS_MINIMOS[@]} ))
CONTADOR=1

# NOVO LOOP: Itera sobre os Casos primeiro
for c in "${CASOS[@]}"; do
    
    # CRIAÇÃO DO DIRETÓRIO ESPECÍFICO DO CASO
    DIRETORIO_CASO="${DIRETORIO_RAIZ_RESULTADOS}/caso_${c}"
    mkdir -p "$DIRETORIO_CASO"

    for h in "${CAMADAS_ESCONDIDAS[@]}"; do
        for a in "${ALPHAS[@]}"; do
            for e in "${MAX_EPOCAS[@]}"; do
                for err in "${ERROS_MINIMOS[@]}"; do
                    
                    # Identifica o nome legível do caso para imprimir na tela
                    if [ "$c" -eq 0 ]; then NOME_CASO="Multiclasse (26)"
                    elif [ "$c" -eq 1 ]; then NOME_CASO="Binário (Buracos)"
                    elif [ "$c" -eq 2 ]; then NOME_CASO="Binário (Curvas)"
                    fi

                    echo ""
                    echo -e "${ROXO}--------------------------------------------------------${NC}"
                    echo -e "${ROXO} Experimento ${VERDE}${CONTADOR}${ROXO} de ${VERDE}${TOTAL_EXP}${NC} | ${AMARELO}Caso: $NOME_CASO${NC}"
                    echo -e "${BRANCO} Run: Caso=$c | Hidden=$h | Alpha=$a | MaxEpocas=$e | ErroMin=$err${NC}"
                    echo -e "${ROXO}--------------------------------------------------------${NC}"
                    
                    # Nome único para a pasta deste experimento DENTRO da pasta do caso
                    PASTA_EXP="${DIRETORIO_CASO}/exp_${CONTADOR}_h${h}_a${a}_e${e}_err${err}"
                    mkdir -p "$PASTA_EXP"
                    
                    # Alimenta o arquivo de resumo geral adicionando a coluna do Caso
                    echo "exp_${CONTADOR},${c},${h},${a},${e},${err}" >> "$ARQUIVO_RESUMO"
                    
                    # Executa o Python passando as variações atuais E O NOVO ARGUMENTO --caso
                    python3 "$PYTHON_SCRIPT" --n_escondidos "$h" --alpha "$a" --max_epocas "$e" --erro_minimo "$err" --caso "$c"
                    
                    echo -e "${AMARELO}-> Isolando arquivos gerados na pasta do experimento...${NC}"
                    
                    # 1. MOVE os arquivos obrigatórios normais
                    [ -f "hiperparametros.txt" ] && mv "hiperparametros.txt" "$PASTA_EXP/"
                    [ -f "pesos_iniciais.txt" ]  && mv "pesos_iniciais.txt" "$PASTA_EXP/"
                    [ -f "pesos_finais.txt" ]    && mv "pesos_finais.txt" "$PASTA_EXP/"
                    [ -f "historico_erros.csv" ] && mv "historico_erros.csv" "$PASTA_EXP/"
                    [ -f "saidas_teste.csv" ]    && mv "saidas_teste.csv" "$PASTA_EXP/"
                    
                    # 2. MOVE os resultados do teste autoral
                    [ -f "resumo_teste_autoral.txt" ] && mv "resumo_teste_autoral.txt" "$PASTA_EXP/"
                    [ -f "saidas_letras_autoral.csv" ] && mv "saidas_letras_autoral.csv" "$PASTA_EXP/"

                    # 3. COPIA os arquivos da base de dados autoral
                    [ -f "X_autoral.txt" ] && cp "X_autoral.txt" "$PASTA_EXP/"
                    [ -f "Y_autoral.txt" ] && cp "Y_autoral.txt" "$PASTA_EXP/"
                    
                    # 4. Solução do gráfico
                    if [ -f "curva_erro.png" ]; then
                        mv "curva_erro.png" "${PASTA_EXP}/curva_erro_exp_${CONTADOR}.png"
                    fi

                    CONTADOR=$((CONTADOR + 1))
                done
            done
        done
    done
    echo -e "${VERDE}-> Extraindo a melhor combinação para o Caso ${c}...${NC}"
    python3 melhor_combinacao.py --caso "$c"
done

echo ""
echo -e "${VERDE}==========================================================${NC}"
echo -e "${VERDE} Concluído! O arquivo histórico global está em: ${BRANCO}$ARQUIVO_RESUMO${NC}"
echo -e "${VERDE}==========================================================${NC}"