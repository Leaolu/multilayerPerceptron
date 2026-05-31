# Implementação do loop de treinamento do MLP com parada antecipada.
import numpy as np
import copy
from mlp_forward import forward
from mlp_backward import backward


def calcular_erro(t, y):
    # Cálculo do erro instantâneo para um padrão de treinamento.
    # E(n) = 1/2 * sum_k((tk - yk)^2)
    # slide 55, Haykin: o valor instantâneo do erro é 1/2 * e²j(n),
    # e o erro total E(n) é a soma sobre todos os neurônios de saída.
    return 0.5 * np.sum((t - y) ** 2)


def atualizar_pesos(v, v0, w, w0, delta_v, delta_v0, delta_w, delta_w0):
    # Passo 8, Fausett (slide 73):
    # Cada unidade de saída (Yk) e cada unidade escondida (Zj) atualizam
    # seus pesos e bias:
    #   wjk(new) = wjk(old) + Δwjk
    #   vij(new) = vij(old) + Δvij
    w += delta_w
    w0 += delta_w0
    v += delta_v
    v0 += delta_v0
    return v, v0, w, w0


def calcular_erro_conjunto(X, T_one_hot, v, v0, w, w0):
    # Calcula o erro quadrático médio para um conjunto de dados (validação ou teste)
    # sem atualizar pesos.
    # Eav = (1/N) * sum_n(E(n))
    # slide 55, Haykin: o erro quadratico médio é a função custo para
    # o conjunto de treinamento e é uma medida de desempenho do aprendizado.
    erro_total = 0.0
    N = len(X)
    for i in range(N):
        _, _, _, y = forward(X[i], v, v0, w, w0)
        erro_total += calcular_erro(T_one_hot[i], y)
    return erro_total / N


def treinar(X_treino, T_treino_one_hot, X_val, T_val_one_hot, v, v0, w, w0, alpha, max_epocas, erro_minimo, paciencia=20):
    # Loop principal de treinamento do MLP.
    # slide 68, Fausett:
    #   Passo 0: Inicializa pesos, bias, taxa de aprendizado, épocas.
    #   Passo 1: Enquanto condição de parada é falsa, execute mais uma época.
    #   Passo 9: Teste condição de parada (erro, taxa de aprendizado ou épocas).
    # slide 82, Haykin: parada antecipada para evitar
    # overfitting, monitorando o erro no conjunto de validação (implementação bônus).
    historico_erros_treino = []
    historico_erros_val = []
    epoca = 0
    N = len(X_treino)

    melhor_erro_val = float('inf')
    epocas_sem_melhoria = 0

    # Salva os pesos iniciais como os melhores até agora
    melhor_v, melhor_v0 = copy.deepcopy(v), copy.deepcopy(v0)
    melhor_w, melhor_w0 = copy.deepcopy(w), copy.deepcopy(w0)

    while epoca < max_epocas:
        erro_epoca = 0.0

        # Passo de Treinamento: para cada par de treinamento (Passo 2, Fausett),
        # executa feedforward (Passos 3-5), backpropagation (Passos 6-7) e
        # atualização de pesos (Passo 8).
        # slide 56, Haykin: alteração de pesos padrão a padrão,
        # uma época é a apresentação completa do conjunto de treinamento.
        for i in range(N):
            Xi = X_treino[i]
            ti = T_treino_one_hot[i]

            z_in, z, y_in, y = forward(Xi, v, v0, w, w0)
            delta_v, delta_v0, delta_w, delta_w0 = backward(Xi, ti, z, z_in, y, y_in, w, alpha)
            v, v0, w, w0 = atualizar_pesos(v, v0, w, w0, delta_v, delta_v0, delta_w, delta_w0)

            erro_epoca += calcular_erro(ti, y)

        erro_medio_treino = erro_epoca / N
        historico_erros_treino.append(erro_medio_treino)

        # Avaliação no conjunto de validação (sem atualizar pesos)
        erro_medio_val = calcular_erro_conjunto(X_val, T_val_one_hot, v, v0, w, w0)
        historico_erros_val.append(erro_medio_val)

        epoca += 1

        # Lógica de Parada Antecipada
        # Caso o número de epocas sem melhoria seja igual ou maior à paciência, temos uma parada antecipada
        if erro_medio_val < melhor_erro_val:
            melhor_erro_val = erro_medio_val
            epocas_sem_melhoria = 0
            melhor_v, melhor_v0 = copy.deepcopy(v), copy.deepcopy(v0)
            melhor_w, melhor_w0 = copy.deepcopy(w), copy.deepcopy(w0)
        else:
            epocas_sem_melhoria += 1

        if epoca == 1 or epoca % 100 == 0:
            print(f"Época {epoca:4d}/{max_epocas} | Erro Treino: {erro_medio_treino:.5f} | Erro Val: {erro_medio_val:.5f} | Paciência: {epocas_sem_melhoria}/{paciencia}")

        # Condições de Parada (Passo 9, slide 68, Fausett)
        if erro_medio_treino <= erro_minimo:
            print(f"-> Parada por Erro Mínimo atingido no treino (Época {epoca})")
            break

        if epocas_sem_melhoria >= paciencia:
            print(f"-> PARADA ANTECIPADA ativada! A validação parou de melhorar na época {epoca}.")
            print("-> Restaurando os pesos da melhor época...")
            v, v0, w, w0 = melhor_v, melhor_v0, melhor_w, melhor_w0
            break

    return v, v0, w, w0, historico_erros_treino, historico_erros_val
