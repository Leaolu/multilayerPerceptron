# Script para gerar gráficos 3D de análise de tempo de execução
# em função de hiperparâmetros e épocas de treinamento.
# X: hiperparâmetro (neurônios ocultos, alpha, erro mínimo, max épocas)
# Y: número da época (1, 2, 3, ..., N)
# Z: tempo acumulado até aquela época

import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def contar_epocas_experimento(caso, numero_exp, hidden, alpha, max_epocas, erro_min):
    """
    Conta quantas épocas foram executadas lendo o arquivo historico_erros_original.csv
    do experimento específico.
    """
    padrao_pasta = re.compile(r"exp_(\d+)_h(\d+)_a([\d.]+)_e(\d+)_err([\d.]+)")
    pasta_exp = f"exp_{numero_exp}_h{hidden}_a{alpha}_e{max_epocas}_err{erro_min}"
    
    # Tenta primeiro o novo nome, depois o legado
    caminho_erros = os.path.join("resultados_mlp", f"caso_{caso}", pasta_exp, "historico_erros_original.csv")
    if not os.path.exists(caminho_erros):
        caminho_erros = os.path.join("resultados_mlp", f"caso_{caso}", pasta_exp, "historico_erros.csv")
    
    if not os.path.exists(caminho_erros):
        return 0
    
    with open(caminho_erros, 'r') as f:
        linhas = [l.strip() for l in f.readlines() if l.strip()]
        # Skip header if exists
        if linhas and not linhas[0][0].isdigit():
            dados = linhas[1:]
        else:
            dados = linhas
        return len(dados)


def coletar_dados_tempo():
    """
    Coleta dados de tempo e hiperparâmetros, integrando com informações
    de épocas de cada experimento.
    Retorna estrutura organizada por caso com dados por época.
    """
    arquivo_csv = "resultados_mlp/resumo_geral_experimentos.csv"
    
    if not os.path.exists(arquivo_csv):
        print("[ERRO] Arquivo de resumo geral não encontrado.")
        return {}
    
    dados = defaultdict(list)
    
    with open(arquivo_csv, 'r') as f:
        reader = csv.DictReader(f)
        for linha in reader:
            try:
                caso = int(linha['Caso'])
                hidden = int(linha['Camada_Oculta'])
                alpha = float(linha['Alpha'])
                max_epocas = int(linha['Max_Epocas'])
                erro_min = float(linha['Erro_Minimo'])
                # Compatibilidade: aceita tanto o formato novo quanto o legado
                if 'Tempo_Original_Segundos' in linha:
                    tempo_total = float(linha['Tempo_Original_Segundos'])
                else:
                    tempo_total = float(linha['Tempo_Segundos'])
                numero_exp = int(linha['Experimento'].split('_')[1])
                
                # Contar épocas executadas
                num_epocas = contar_epocas_experimento(caso, numero_exp, hidden, alpha, max_epocas, erro_min)
                
                if num_epocas == 0:
                    continue  # Pular se não conseguir ler as épocas
                
                # Calcular tempo por época
                tempo_por_epoca = tempo_total / num_epocas if num_epocas > 0 else 0
                
                exp_data = {
                    'caso': caso,
                    'hidden': hidden,
                    'alpha': alpha,
                    'max_epocas': max_epocas,
                    'erro_min': erro_min,
                    'tempo_total': tempo_total,
                    'num_epocas': num_epocas,
                    'tempo_por_epoca': tempo_por_epoca
                }
                dados[caso].append(exp_data)
            except (ValueError, KeyError) as e:
                print(f"[AVISO] Erro ao processar linha: {e}")
                continue
    
    return dados



def plotar_grafico_3d(hiperparametro_values, epocas_values, tempo_acumulado_values, 
                     hiperparametro_label, titulo, nome_arquivo, caso):
    """
    Cria um gráfico 3D com:
    - X: valores do hiperparâmetro
    - Y: número da época (1, 2, 3, ..., N)
    - Z: tempo acumulado até aquela época
    """
    pasta_saida = f"tempos_execucao/caso_{caso}"
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    caminho_completo = os.path.join(pasta_saida, nome_arquivo)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot com cores baseadas no tempo
    scatter = ax.scatter(hiperparametro_values, epocas_values, tempo_acumulado_values,
                        c=tempo_acumulado_values, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(hiperparametro_label, fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Época de Treinamento', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Tempo Acumulado (segundos)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=20)
    
    # Adicionar colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Tempo Acumulado (s)', fontsize=10)
    
    # Ajustar ângulo de visualização
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(caminho_completo, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Gráfico salvo: {caminho_completo}")


def gerar_graficos_3d_para_caso(dados_caso, caso):
    """
    Gera 4 gráficos 3D para um caso específico.
    Cada gráfico tem um hiperparâmetro diferente no eixo X,
    as épocas no eixo Y, e o tempo acumulado no eixo Z.
    """
    print(f"\n📊 Gerando gráficos 3D para Caso {caso}...")
    
    if not dados_caso:
        print(f"  [AVISO] Sem dados para caso {caso}")
        return
    
    # Preparar dados para cada gráfico
    
    # Gráfico 1: Neurônios Ocultos vs Épocas vs Tempo
    print("  → Preparando gráfico 1: Neurônios Ocultos vs Épocas...")
    x1, y1, z1 = [], [], []
    for exp in dados_caso:
        for epoca in range(1, exp['num_epocas'] + 1):
            x1.append(exp['hidden'])
            y1.append(epoca)
            z1.append(exp['tempo_por_epoca'] * epoca)
    
    if x1:
        plotar_grafico_3d(x1, y1, z1,
                         'Neurônios Ocultos',
                         'Tempo Acumulado: Neurônios Ocultos vs Épocas',
                         'tempo_3d_neurons_vs_epochs.png', caso)
    
    # Gráfico 2: Alpha vs Épocas vs Tempo
    print("  → Preparando gráfico 2: Alpha vs Épocas...")
    x2, y2, z2 = [], [], []
    for exp in dados_caso:
        for epoca in range(1, exp['num_epocas'] + 1):
            x2.append(exp['alpha'])
            y2.append(epoca)
            z2.append(exp['tempo_por_epoca'] * epoca)
    
    if x2:
        plotar_grafico_3d(x2, y2, z2,
                         'Taxa Aprendizado (Alpha)',
                         'Tempo Acumulado: Alpha vs Épocas',
                         'tempo_3d_alpha_vs_epochs.png', caso)
    
    # Gráfico 3: Erro Mínimo vs Épocas vs Tempo
    print("  → Preparando gráfico 3: Erro Mínimo vs Épocas...")
    x3, y3, z3 = [], [], []
    for exp in dados_caso:
        for epoca in range(1, exp['num_epocas'] + 1):
            x3.append(exp['erro_min'])
            y3.append(epoca)
            z3.append(exp['tempo_por_epoca'] * epoca)
    
    if x3:
        plotar_grafico_3d(x3, y3, z3,
                         'Limiar Erro Mínimo',
                         'Tempo Acumulado: Erro Mínimo vs Épocas',
                         'tempo_3d_erro_vs_epochs.png', caso)
    
    # Gráfico 4: Max Épocas vs Épocas vs Tempo
    print("  → Preparando gráfico 4: Max Épocas vs Épocas...")
    x4, y4, z4 = [], [], []
    for exp in dados_caso:
        for epoca in range(1, exp['num_epocas'] + 1):
            x4.append(exp['max_epocas'])
            y4.append(epoca)
            z4.append(exp['tempo_por_epoca'] * epoca)
    
    if x4:
        plotar_grafico_3d(x4, y4, z4,
                         'Limite Máximo de Épocas',
                         'Tempo Acumulado: Max Épocas vs Épocas',
                         'tempo_3d_maxepocas_vs_epochs.png', caso)


def main():
    print("\n" + "="*60)
    print(" Gerador de Gráficos 3D: Tempo de Execução vs Épocas")
    print("="*60)
    
    # Coletar dados
    dados = coletar_dados_tempo()
    
    if not dados:
        print("[ERROR] Nenhum dado foi coletado. Verifique o arquivo de resumo.")
        return
    
    # Gerar gráficos para cada caso
    for caso in sorted(dados.keys()):
        gerar_graficos_3d_para_caso(dados[caso], caso)
    
    print("\n" + "="*60)
    print(" ✅ Gráficos 3D gerados com sucesso!")
    print(" 📁 Localização: tempos_execucao/caso_X/")
    print(" 📊 Estrutura:")
    print("    - tempo_3d_neurons_vs_epochs.png")
    print("    - tempo_3d_alpha_vs_epochs.png")
    print("    - tempo_3d_erro_vs_epochs.png")
    print("    - tempo_3d_maxepocas_vs_epochs.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
