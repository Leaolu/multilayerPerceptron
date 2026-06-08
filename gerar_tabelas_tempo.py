# Script para gerar tabelas de análise de tempo de execução médio
# em função de cada hiperparâmetro, substituindo os antigos gráficos 3D.
# O tempo analisado é o tempo total (treino + validação + teste).
# Gera a saída em formato Markdown estrito e alinhado para exportação limpa.

import os
import csv
from collections import defaultdict

def calcular_media_dicionario(dicionario):
    # Calcula a média dos valores para cada chave em um dicionário
    medias = {}
    for chave, valores in dicionario.items():
        medias[chave] = sum(valores) / len(valores)
    return medias


def gerar_tabelas_para_caso(dados_caso, caso):
    # Gera e exibe as 4 tabelas de tempo médio para um caso específico em formato Markdown.
    # Garante alinhamento correto e delimitadores apropriados para renderização visual.
    
    nomes_casos = {0: "Multiclasse (26 Letras)", 1: "Binário (Letras com Buraco)", 2: "Binário (Letras com Curva)"}
    nome_atual = nomes_casos.get(caso, f"Caso {caso}")
    
    print(f"\n## Análise de Tempo Médio - {nome_atual}")
    print(f"Resultados agregados considerando o tempo total de execução (treino + validação + teste).\n")

    tempos_por_hidden = defaultdict(list)
    tempos_por_alpha = defaultdict(list)
    tempos_por_max_epocas = defaultdict(list)
    tempos_por_erro_min = defaultdict(list)

    # Agrupa os tempos pelo valor do hiperparâmetro
    for exp in dados_caso:
        tempos_por_hidden[exp['hidden']].append(exp['tempo_total'])
        tempos_por_alpha[exp['alpha']].append(exp['tempo_total'])
        tempos_por_max_epocas[exp['max_epocas']].append(exp['tempo_total'])
        tempos_por_erro_min[exp['erro_min']].append(exp['tempo_total'])

    # Calcula as médias
    media_hidden = calcular_media_dicionario(tempos_por_hidden)
    media_alpha = calcular_media_dicionario(tempos_por_alpha)
    media_max_epocas = calcular_media_dicionario(tempos_por_max_epocas)
    media_erro_min = calcular_media_dicionario(tempos_por_erro_min)

    # Função auxiliar para imprimir a tabela estritamente formatada em Markdown
    def imprimir_tabela_markdown(nome_parametro, dicionario_medias):
        print(f"### Tempo Médio por {nome_parametro}\n")
        print(f"| {nome_parametro:<30} | Tempo Médio (Segundos) |")
        print(f"| :---" + " " * 26 + " | :---" + " " * 18 + " |")
        for chave in sorted(dicionario_medias.keys()):
            # Formata chaves flutuantes ou inteiras mantendo alinhamento limpo
            str_chave = f"{chave}"
            print(f"| {str_chave:<30} | {dicionario_medias[chave]:<22.4f} |")
        print("\n")

    imprimir_tabela_markdown("Neurônios Ocultos", media_hidden)
    imprimir_tabela_markdown("Taxa de Aprendizado (Alpha)", media_alpha)
    imprimir_tabela_markdown("Limite Máximo de Épocas", media_max_epocas)
    imprimir_tabela_markdown("Limiar de Erro Mínimo", media_erro_min)


def main():
    arquivo_csv = "resultados_mlp/resumo_geral_experimentos.csv"
    
    if not os.path.exists(arquivo_csv):
        print("# Arquivo de Resumo Não Encontrado\n")
        print("Arquivo `resultados_mlp/resumo_geral_experimentos.csv` não foi detectado. Rode a bateria de testes primeiro.")
        return
    
    dados = defaultdict(list)
    
    # Coleta e organiza os dados do CSV de resultados globais
    with open(arquivo_csv, 'r') as f:
        reader = csv.DictReader(f)
        for linha in reader:
            try:
                caso = int(linha['Caso'])
                hidden = int(linha['Camada_Oculta'])
                alpha = float(linha['Alpha'])
                max_epocas = int(linha['Max_Epocas'])
                erro_min = float(linha['Erro_Minimo'])
                
                if 'Tempo_Original_Segundos' in linha:
                    tempo_total = float(linha['Tempo_Original_Segundos'])
                else:
                    tempo_total = float(linha['Tempo_Segundos'])
                
                dados[caso].append({
                    'hidden': hidden,
                    'alpha': alpha,
                    'max_epocas': max_epocas,
                    'erro_min': erro_min,
                    'tempo_total': tempo_total
                })
            except (ValueError, KeyError):
                continue

    print("# Relatório de Tempos Médios de Execução do MLP")
    print("Este relatório apresenta o impacto isolado de cada hiperparâmetro no tempo computacional.\n")

    # Gera as tabelas para cada caso processado
    for caso in sorted(dados.keys()):
        gerar_tabelas_para_caso(dados[caso], caso)


if __name__ == "__main__":
    main()