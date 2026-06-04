# Script para análise pós-experimentos em lote (Geração de gráficos para Slides)
# Extrai as métricas de tempo vs hiperparâmetros e seleciona as matrizes
# de confusão representativas (Pior, Média e Melhor performance).
import csv
import os
import matplotlib.pyplot as plt

def analisar_resultados():
    arquivo_csv = "resultados_mlp/resumo_geral_experimentos.csv"
    if not os.path.exists(arquivo_csv):
        print("Arquivo de resumo geral não encontrado. Rode o script .sh primeiro.")
        return

    dados = []
    with open(arquivo_csv, "r") as f:
        reader = csv.DictReader(f)
        for linha in reader:
            # Compatibilidade: aceita tanto o formato novo quanto o legado
            if 'Tempo_Original_Segundos' in linha:
                tempo = float(linha['Tempo_Original_Segundos'])
            else:
                tempo = float(linha['Tempo_Segundos'])
            dados.append({
                'exp': linha['Experimento'],
                'caso': linha['Caso'],
                'hidden': int(linha['Camada_Oculta']),
                'epocas': int(linha['Max_Epocas']),
                'tempo': tempo
            })

    # Analisando apenas o Caso 0 (Completo) para os gráficos de tempo
    dados_caso0 = [d for d in dados if d['caso'] == '0']
    
    if not dados_caso0:
        return

    gerar_grafico_tempo(dados_caso0, 'hidden', 'Quantidade de Neurônios Ocultos', 'Tempo_vs_Neuronios.png')
    gerar_grafico_tempo(dados_caso0, 'epocas', 'Limite Máximo de Épocas', 'Tempo_vs_Epocas.png')

def gerar_grafico_tempo(dados, chave_agrupamento, rotulo_x, nome_arquivo):
    agrupado = {}
    for d in dados:
        val = d[chave_agrupamento]
        if val not in agrupado:
            agrupado[val] = []
        agrupado[val].append(d['tempo'])
    
    x_vals = sorted(list(agrupado.keys()))
    y_vals = [sum(agrupado[x])/len(agrupado[x]) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.bar([str(x) for x in x_vals], y_vals, color='royalblue')
    plt.title(f'Tempo Médio de Treinamento vs {rotulo_x}')
    plt.xlabel(rotulo_x)
    plt.ylabel('Tempo de Execução Médio (segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    caminho_salvar = os.path.join("resultados_mlp", nome_arquivo)
    plt.savefig(caminho_salvar)
    plt.close()
    print(f"Gráfico gerado: {caminho_salvar}")

if __name__ == "__main__":
    analisar_resultados()