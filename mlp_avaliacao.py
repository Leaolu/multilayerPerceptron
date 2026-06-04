# Funções de avaliação do modelo MLP: métricas de desempenho e visualização
# da curva de erro durante o treinamento.
import numpy as np
import matplotlib.pyplot as plt
import csv
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


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


def gerar_imagem_matriz_confusao(matriz, caminho_png, caso=0):
    # Gera uma visualização em PNG da matriz de confusão com labels legíveis.
    # caso: 0 para multiclasse (A-Z), 1-2 para binário (0-1)
    n_classes = matriz.shape[0]
    
    # Criar labels conforme o caso
    if caso == 0:
        labels = [chr(ord('A') + i) for i in range(n_classes)]
    else:
        labels = [str(i) for i in range(n_classes)]
    
    # Criar figura com tamanho apropriado
    fig_size = max(8, n_classes * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Usar seaborn se disponível para melhor aparência
    if HAS_SEABORN:
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, 
                   cbar_kws={'label': 'Quantidade'}, ax=ax)
    else:
        # Fallback: usar matplotlib com imshow
        im = ax.imshow(matriz, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax, label='Quantidade')
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Adicionar anotações manualmente se não houver seaborn
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, matriz[i, j], ha="center", va="center", 
                             color="white" if matriz[i, j] > matriz.max() / 2 else "black",
                             fontsize=max(6, 12 - n_classes // 5))
    
    ax.set_xlabel('Classe Predita', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
    ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(caminho_png, bbox_inches='tight', dpi=150)
    plt.close()


def plotar_analise_completa_slides(erros_treino, erros_val, erros_teste, erros_ruidoso, pesos_v, pesos_w):
    # Função para gerar gráficos exclusivos da Apresentação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epocas = range(1, len(erros_treino) + 1)
    
    # Gráfico 1: Evolução dos Erros em Todos os Conjuntos
    ax1.plot(epocas, erros_treino, 'b-', label='Treino')
    if erros_val: ax1.plot(epocas, erros_val, 'r--', label='Validação')
    if erros_teste: ax1.plot(epocas, erros_teste, 'g:', label='Teste')
    if erros_ruidoso and erros_ruidoso[0] is not None: 
        ax1.plot(epocas, erros_ruidoso, 'm-.', label='Ruidoso')
        
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