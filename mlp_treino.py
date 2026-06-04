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
    erro_total = 0.0
    for i in range(len(X)):
        _, _, _, y = forward(X[i], v, v0, w, w0)
        erro_total += calcular_erro(T_one_hot[i], y)
    return erro_total / len(X)


def treinar(X_treino, T_treino_one_hot, X_val, T_val_one_hot, X_teste, T_teste_one_hot, X_aut, T_aut_one_hot, v, v0, w, w0, alpha, max_epocas, erro_minimo, paciencia):
    # Executa o loop principal (Passos 1 ao 9 do algoritmo, slide 68 - Fausett).
    historico_erros_treino = []
    historico_erros_val = []
    
    # Adicionadas variáveis para a geração dos gráficos da apresentação
    historico_erros_teste = []
    historico_erros_aut = []
    convergencia_pesos_v = []
    convergencia_pesos_w = []

    melhor_erro_val = float('inf')
    epocas_sem_melhoria = 0
    epoca = 0
    N = len(X_treino)
    
    melhor_v, melhor_v0 = copy.deepcopy(v), copy.deepcopy(v0)
    melhor_w, melhor_w0 = copy.deepcopy(w), copy.deepcopy(w0)

    while epoca < max_epocas:
        erro_epoca_treino = 0.0

        for i in range(N):
            Xi = X_treino[i]
            ti = T_treino_one_hot[i]
            z_in, z, y_in, y = forward(Xi, v, v0, w, w0)
            delta_v, delta_v0, delta_w, delta_w0 = backward(Xi, ti, z, z_in, y, y_in, w, alpha)
            v, v0, w, w0 = atualizar_pesos(v, v0, w, w0, delta_v, delta_v0, delta_w, delta_w0)

        erro_medio_treino = calcular_erro_conjunto(X_treino, T_treino_one_hot, v, v0, w, w0)
        historico_erros_treino.append(erro_medio_treino)

        # Avaliação no conjunto de validação (sem atualizar pesos)
        erro_medio_val = calcular_erro_conjunto(X_val, T_val_one_hot, v, v0, w, w0)
        historico_erros_val.append(erro_medio_val)
        
        # Avaliação nos conjuntos de teste e autoral para análise dos slides
        erro_medio_teste = calcular_erro_conjunto(X_teste, T_teste_one_hot, v, v0, w, w0)
        historico_erros_teste.append(erro_medio_teste)
        
        if X_aut is not None:
            historico_erros_aut.append(calcular_erro_conjunto(X_aut, T_aut_one_hot, v, v0, w, w0))
        else:
            historico_erros_aut.append(None)
            
        convergencia_pesos_v.append(np.mean(np.abs(v)))
        convergencia_pesos_w.append(np.mean(np.abs(w)))

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
            print(f"-> Parada por Erro Mínimo atingido no treino (Época {epoca}).")
            break
            
        if epocas_sem_melhoria >= paciencia:
            print(f"-> Parada Antecipada (Early Stopping) acionada na época {epoca}.")
            break

    return melhor_v, melhor_v0, melhor_w, melhor_w0, historico_erros_treino, historico_erros_val, historico_erros_teste, historico_erros_aut, convergencia_pesos_v, convergencia_pesos_w