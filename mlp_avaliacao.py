# Funções de avaliação do modelo MLP: métricas de desempenho e visualização
# da curva de erro durante o treinamento.
import numpy as np
import matplotlib.pyplot as plt
import csv


def matriz_confusao(T_real, T_predito, n_classes):
    # constrói a matriz de confusão n_classes x n_classes.
    # Cada posição (i,j) indica quantas amostras da classe real i
    # foram classificadas como classe j pela rede neural
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(T_real, T_predito):
        matriz[real][pred] += 1
    return matriz


def acuracia(T_real, T_predito):
    # calcula a acurácia: proporção de classificações corretas sobre o total.
    corretos = np.sum(np.array(T_real) == np.array(T_predito))
    return corretos / len(T_real)


def plotar_curva_erro(historico_erros_treino, historico_erros_val=None):
    # Plota a curva de erro quadrático médio ao longo das épocas.
    # slide 82, Haykin: a parada antecipada monitora o erro de
    # validação.
    # A visualização das duas curvas (treino vs validação) permite
    # identificar esse ponto de divergência.
    plt.figure(figsize=(8, 6))

    plt.plot(range(1, len(historico_erros_treino) + 1), historico_erros_treino, color='b', label='Treino')

    if historico_erros_val is not None:
        plt.plot(range(1, len(historico_erros_val) + 1), historico_erros_val, color='r', label='Validação')

    plt.title('Curva de Erro do Treinamento vs Validação (Early Stopping)')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio')
    plt.legend()
    plt.grid(True)

    plt.savefig('curva_erro.png', bbox_inches='tight')
    plt.close()


def salvar_saidas_teste(X_teste, T_real, T_predito, caminho):
    # salva as saídas do teste em CSV para análise posterior.
    # Cada linha contém o rótulo real (esquerda) e a classe predita pela rede (direita),
    # permitindo verificar acertos e erros individuais.
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rotulo_Real', 'Classe_Predita'])
        for r, p in zip(T_real, T_predito):
            writer.writerow([r, p])


def salvar_matriz_confusao(matriz, caminho):
    # Salva a matriz de confusão em um arquivo CSV.
    # Auxilia na avaliação de generalização da rede descrita na especificação
    # do trabalho e no slide 80 - Fausett (aplicação da rede neural).
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        for linha in matriz:
            writer.writerow(linha)


def plotar_analise_completa_slides(erros_treino, erros_val, erros_teste, erros_autoral, pesos_v, pesos_w):
    # Função para gerar gráficos exclusivos da Apresentação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epocas = range(1, len(erros_treino) + 1)
    
    # Gráfico 1: Evolução dos Erros em Todos os Conjuntos
    ax1.plot(epocas, erros_treino, 'b-', label='Treino')
    if erros_val: ax1.plot(epocas, erros_val, 'r--', label='Validação')
    if erros_teste: ax1.plot(epocas, erros_teste, 'g:', label='Teste')
    if erros_autoral and erros_autoral[0] is not None: 
        ax1.plot(epocas, erros_autoral, 'm-.', label='Ruído/Autoral')
        
    ax1.set_title('Convergência do Erro (Parada Antecipada)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Erro Quadrático Médio')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico 2: Estabilidade Numérica dos Pesos
    ax2.plot(epocas, pesos_v, 'c-', label='Magnitude Média (Oculta - V)')
    ax2.plot(epocas, pesos_w, 'y-', label='Magnitude Média (Saída - W)')
    ax2.set_title('Estabilização dos Pesos Sinápticos')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Média Absoluta dos Pesos')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('grafico_analise_completa.png', bbox_inches='tight', dpi=300)
    plt.close()