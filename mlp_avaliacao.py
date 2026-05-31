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


def salvar_matriz_confusao(T_real, T_predito, n_classes, rotulos, caminho_csv, caminho_png, titulo="Matriz de Confusão"):
    # Persiste a matriz de confusão em dois formatos:
    #  - CSV: para inspeção textual e processamento posterior.
    #  - PNG: heatmap para visualização rápida (usado no vídeo de apresentação).
    # rotulos: lista de strings com o nome de cada classe (mesmo tamanho de n_classes).
    matriz = matriz_confusao(T_real, T_predito, n_classes)

    # Salva o CSV com cabeçalho indicando linhas = rótulo real, colunas = predito.
    with open(caminho_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Real \\ Predito'] + list(rotulos))
        for i, linha in enumerate(matriz):
            writer.writerow([rotulos[i]] + list(linha))

    # Gera heatmap com matplotlib (apenas I/O e plot, sem biblioteca de RNA).
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.4), max(5, n_classes * 0.4)))
    im = ax.imshow(matriz, cmap='Blues', aspect='auto')
    ax.set_title(titulo)
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(rotulos, rotation=45, ha='right')
    ax.set_yticklabels(rotulos)

    # Anota o valor de cada célula para leitura direta.
    limite = matriz.max() / 2.0 if matriz.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            cor = 'white' if matriz[i, j] > limite else 'black'
            ax.text(j, i, str(matriz[i, j]), ha='center', va='center', color=cor, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(caminho_png, bbox_inches='tight')
    plt.close(fig)

    return matriz


def salvar_saidas_teste(X_teste, T_real, T_predito, caminho):
    # salva as saídas do teste em CSV para análise posterior.
    # Cada linha contém o rótulo real (esquerda) e a classe predita pela rede (direita),
    # permitindo verificar acertos e erros individuais.
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rotulo_Real', 'Classe_Predita'])
        for r, p in zip(T_real, T_predito):
            writer.writerow([r, p])
