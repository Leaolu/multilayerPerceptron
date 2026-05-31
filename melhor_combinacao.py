# codigo para análise dos resultados de múltiplos experimentos de treinamento
# do MLP. Identifica a melhor combinação de hiperparâmetros com base na
# acurácia no conjunto de teste e isola os arquivos do modelo campeão.
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
        pasta_destino = "melhor_combinacao_palavra"
    elif caso == 1:
        nome_problema = "Binário (Letras com Buraco)"
        pasta_destino = "melhor_combinacao_letra_com_buraco"
    elif caso == 2:
        nome_problema = "Binário (Letras com Curva)"
        pasta_destino = "melhor_combinacao_letra_curvada"
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
        acc_teste = 0.0
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

        # calcula acurácia no conjunto autoral
        acc_autoral = 0.0
        caminho_autoral = os.path.join(caminho_pasta, "saidas_letras_autoral.csv")
        if os.path.exists(caminho_autoral):
            with open(caminho_autoral, 'r', encoding='utf-8') as f:
                linhas = f.readlines()[1:]
                total, acertos = 0, 0
                for linha in linhas:
                    partes = linha.strip().split(',')
                    if len(partes) >= 2:
                        total += 1
                        if partes[0].upper() == partes[1].upper():
                            acertos += 1
                if total > 0:
                    acc_autoral = acertos / total

        # extrai épocas executadas e erro final do histórico de erros
        erro_final = float('inf')
        epocas_executadas = 0
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
            "acc_autoral": acc_autoral,
            "erro_final": erro_final,
            "epocas": epocas_executadas
        })

    if not experimentos:
        print(f"⚠️ Nenhum experimento válido com os arquivos gerados foi encontrado no diretório {diretorio_busca}.")
        return

    # Critério de seleção: maior acurácia no teste, desempate pelo menor erro final
    experimentos.sort(key=lambda x: (-x["acc_teste"], x["erro_final"]))

    campeao = experimentos[0]

    # Isola os arquivos da rede campeã em diretório dedicado
    if os.path.exists(pasta_destino):
        shutil.rmtree(pasta_destino)
    os.makedirs(pasta_destino)

    pasta_origem = os.path.join(diretorio_busca, campeao['pasta'])

    print(f"🏆 Melhor exp_{campeao['id']} (Acc: {campeao['acc_teste']*100:.2f}%). Salvando em '{pasta_destino}'...")

    arquivos_copiados = []
    for arquivo in os.listdir(pasta_origem):
        origem_completa = os.path.join(pasta_origem, arquivo)
        destino_completo = os.path.join(pasta_destino, arquivo)
        if os.path.isfile(origem_completa):
            shutil.copy2(origem_completa, destino_completo)
            arquivos_copiados.append(arquivo)

    # gera arquivo resumo com os hiperparâmetros vencedores do caso
    caminho_salvamento = os.path.join(pasta_destino, f"hiperparametros_vencedores_caso_{caso}.txt")
    with open(caminho_salvamento, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write(f"  HIPERPARÂMETROS FINAIS DA ARQUITETURA - {nome_problema.upper()} \n")
        f.write("===================================================================\n\n")
        f.write(f"Melhor Experimento: exp_{campeao['id']}\n")
        f.write("--- Hiperparâmetros Selecionados ---\n")
        f.write(f"  - Neurônios Ocultos (Hidden)     : {campeao['h']}\n")
        f.write(f"  - Taxa de Aprendizado (Alpha)    : {campeao['alpha']}\n")
        f.write(f"  - Limite Máximo de Épocas        : {campeao['max_epocas']}\n")
        f.write(f"  - Erro Mínimo Estipulado         : {campeao['erro_minimo']}\n\n")
        f.write("--- Métricas Finais Alcançadas ---\n")
        f.write(f"  - Acurácia Final (Conjunto Teste) : {campeao['acc_teste']*100:.2f}%\n")
        f.write(f"  - Acurácia Final (Base Autoral)   : {campeao['acc_autoral']*100:.2f}%\n")
        f.write(f"  - Épocas Executadas até a Parada  : {campeao['epocas']}\n")
        # Previne quebra de string caso o erro seja infinito (inf)
        f.write(f"  - Erro Quadrático Médio Final     : {campeao['erro_final']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encontrar melhor MLP do caso")
    parser.add_argument("--caso", type=int, required=True, help="Número do caso (0, 1 ou 2)")
    args = parser.parse_args()

    analisar_experimentos(args.caso)