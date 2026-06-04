# codigo para análise dos resultados de múltiplos experimentos de treinamento
# do MLP. Identifica a PIOR combinação de hiperparâmetros com base na
# acurácia no conjunto de teste e isola os arquivos do modelo com pior desempenho.
import os
import re
import shutil
import argparse


def analisar_experimentos(caso):
    # busca os resultados dos experimentos no diretório correspondente ao caso
    diretorio_busca = os.path.join("resultados_mlp", f"caso_{caso}")

    if not os.path.exists(diretorio_busca):
        print(f"❌ Erro: O diretório '{diretorio_busca}' não foi encontrado.")
        return

    # Mapeia o caso para nome do problema e pasta de destino
    if caso == 0:
        nome_problema = "Multiclasse (Alfabeto A-Z)"
        pasta_destino = "pior_combinacao_palavra"
    elif caso == 1:
        nome_problema = "Binário (Letras com Buraco)"
        pasta_destino = "pior_combinacao_letra_com_buraco"
    elif caso == 2:
        nome_problema = "Binário (Letras com Curva)"
        pasta_destino = "pior_combinacao_letra_curvada"
    else:
        print("Caso inválido!")
        return

    # padrão de nome das pastas de experimento: exp_{id}_h{hidden}_a{alpha}_e{epocas}_err{erro}
    padrao_pasta = re.compile(r"exp_(\d+)_h(\d+)_a([\d.]+)_e(\d+)_err([\d.]+)")
    experimentos = []

    print(f"\n🔍 Analisando os resultados do Caso {caso}: {nome_problema}...")

    for nome_pasta in os.listdir(diretorio_busca):
        caminho_pasta = os.path.join(diretorio_busca, nome_pasta)
        if not os.path.isdir(caminho_pasta):
            continue

        match = padrao_pasta.search(nome_pasta)
        if not match:
            continue

        id_exp, h, alpha, max_epocas, erro_min = match.groups()

        # calcula acurácia no conjunto de teste a partir do CSV de saídas
        # Tenta primeiro na nova estrutura (original/), depois na raiz (legado)
        acc_teste = 0.0
        caminho_teste = os.path.join(caminho_pasta, "original", "saidas_teste.csv")
        if not os.path.exists(caminho_teste):
            caminho_teste = os.path.join(caminho_pasta, "saidas_teste.csv")
        if os.path.exists(caminho_teste):
            with open(caminho_teste, 'r', encoding='utf-8') as f:
                linhas = f.readlines()[1:]
                total, acertos = 0, 0
                for linha in linhas:
                    partes = linha.strip().split(',')
                    if len(partes) >= 2:
                        total += 1
                        if partes[0].upper() == partes[1].upper():
                            acertos += 1
                if total > 0:
                    acc_teste = acertos / total

        # calcula acurácia no conjunto ruidoso
        # Tenta primeiro na nova estrutura (ruidoso/), depois na raiz (legado)
        acc_ruidoso = 0.0
        caminho_ruidoso = os.path.join(caminho_pasta, "ruidoso", "saidas_ruidoso.csv")
        if not os.path.exists(caminho_ruidoso):
            caminho_ruidoso = os.path.join(caminho_pasta, "saidas_letras_autoral.csv")
        if os.path.exists(caminho_ruidoso):
            with open(caminho_ruidoso, 'r', encoding='utf-8') as f:
                linhas = f.readlines()[1:]
                total, acertos = 0, 0
                for linha in linhas:
                    partes = linha.strip().split(',')
                    if len(partes) >= 2:
                        total += 1
                        if partes[0].upper() == partes[1].upper():
                            acertos += 1
                if total > 0:
                    acc_ruidoso = acertos / total

        # extrai épocas executadas e erro final do histórico de erros
        erro_final = float('inf')
        epocas_executadas = 0
        caminho_erros = os.path.join(caminho_pasta, "historico_erros_original.csv")
        if not os.path.exists(caminho_erros):
            caminho_erros = os.path.join(caminho_pasta, "historico_erros.csv")
        if os.path.exists(caminho_erros):
            with open(caminho_erros, 'r', encoding='utf-8') as f:
                linhas = [l.strip() for l in f.readlines() if l.strip()]
                if linhas and not linhas[0].replace(',','').replace('.','').replace('-','').isdigit():
                    dados = linhas[1:]
                else:
                    dados = linhas

                if dados:
                    epocas_executadas = len(dados)
                    ultima_linha = dados[-1].split(',')
                    try:
                        erro_final = float(ultima_linha[-1])
                    except ValueError:
                        pass

        experimentos.append({
            "id": int(id_exp),
            "pasta": nome_pasta,
            "h": int(h),
            "alpha": float(alpha),
            "max_epocas": int(max_epocas),
            "erro_minimo": float(erro_min),
            "acc_teste": acc_teste,
            "acc_ruidoso": acc_ruidoso,
            "erro_final": erro_final,
            "epocas": epocas_executadas
        })

    if not experimentos:
        print(f"⚠️ Nenhum experimento válido com os arquivos gerados foi encontrado no diretório {diretorio_busca}.")
        return

    # Critério de seleção: MENOR acurácia no teste, desempate pelo MAIOR erro final (PIOR desempenho)
    experimentos.sort(key=lambda x: (x["acc_teste"], -x["erro_final"]))

    pior = experimentos[0]

    # Isola os arquivos da rede com pior desempenho em diretório dedicado
    if os.path.exists(pasta_destino):
        shutil.rmtree(pasta_destino)
    os.makedirs(pasta_destino)

    pasta_origem = os.path.join(diretorio_busca, pior['pasta'])

    print(f"📉 Pior exp_{pior['id']} (Acc: {pior['acc_teste']*100:.2f}%). Salvando em '{pasta_destino}'...")

    # Copia toda a estrutura do experimento (incluindo subpastas original/ e ruidoso/)
    for item in os.listdir(pasta_origem):
        origem_completa = os.path.join(pasta_origem, item)
        destino_completo = os.path.join(pasta_destino, item)
        if os.path.isdir(origem_completa):
            shutil.copytree(origem_completa, destino_completo)
        elif os.path.isfile(origem_completa):
            shutil.copy2(origem_completa, destino_completo)

    # gera arquivo resumo com os hiperparâmetros do pior caso
    caminho_salvamento = os.path.join(pasta_destino, f"hiperparametros_pior_caso_{caso}.txt")
    with open(caminho_salvamento, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write(f"  HIPERPARÂMETROS - PIOR DESEMPENHO - {nome_problema.upper()} \n")
        f.write("===================================================================\n\n")
        f.write(f"Pior Experimento: exp_{pior['id']}\n")
        f.write("--- Hiperparâmetros da Configuração ---\n")
        f.write(f"  - Neurônios Ocultos (Hidden)     : {pior['h']}\n")
        f.write(f"  - Taxa de Aprendizado (Alpha)    : {pior['alpha']}\n")
        f.write(f"  - Limite Máximo de Épocas        : {pior['max_epocas']}\n")
        f.write(f"  - Erro Mínimo Estipulado         : {pior['erro_minimo']}\n\n")
        f.write("--- Métricas Alcançadas (Pior Performance) ---\n")
        f.write(f"  - Acurácia Alcançada (Teste)     : {pior['acc_teste']*100:.2f}%\n")
        f.write(f"  - Acurácia Alcançada (Ruidoso)   : {pior['acc_ruidoso']*100:.2f}%\n")
        f.write(f"  - Épocas Executadas até Parada   : {pior['epocas']}\n")
        f.write(f"  - Erro Quadrático Médio Final    : {pior['erro_final']}\n")
        f.write("\n[NOTA] Este modelo representa a pior combinação de hiperparâmetros,\n")
        f.write("    útil para análise contrastante com a melhor combinação.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encontrar pior MLP do caso")
    parser.add_argument("--caso", type=int, required=True, help="Número do caso (0, 1 ou 2)")
    args = parser.parse_args()

    analisar_experimentos(args.caso)
