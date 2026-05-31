# Implementação do estágio de retropropagação do erro (Passos 6 e 7 do
# algoritmo de treinamento descrito por Fausett).
import numpy as np
from mlp_forward import sigmoid_derivada


def calcular_delta_saida(t, y, y_in):
    # Passo 6: Fausett (slide 72):
    # Cada unidade de saída (Yk) computa o termo de informação de erro δk.
    #   δk = (tk - yk) * f'(y_ink)
    # onde tk é a saída desejada, yk é a saída obtida, e f' é a derivada
    # da função de ativação avaliada no campo local induzido y_ink.
    # slide 61, Haykin: o gradiente local para o neurônio de saída
    # é o produto do erro pela derivada da função de ativação.
    return (t - y) * sigmoid_derivada(y_in)


def calcular_correcao_pesos_saida(delta_k, z, alpha):
    # Passo 6: Fausett (slide 72):
    # calcula a correção de pesos e bias da camada de saída:
    #   Δwjk = α * δk * zj
    #   Δw0k = α * δk
    # onde α é a taxa de aprendizado e zj é a saída da camada escondida.
    # slide 60 - Haykin: Δwji(n) = η * δj(n) * yi(n) (regra Delta).
    delta_w = alpha * np.outer(z, delta_k)
    delta_w0 = alpha * delta_k
    return delta_w, delta_w0


def calcular_delta_escondida(delta_k, w, z_in):
    # Passo 7: Fausett (slide 72):
    # Cada unidade escondida (Zj) soma as informações de erro vindas da
    # camada de saída e computa seu próprio gradiente local δj.
    #   δ_inj = sum_k(δk * wjk)
    #   δj = δ_inj * f'(z_inj)
    # slide 62-66, Haykin: o neurônio escondido não possui saída
    # desejada direta; seu erro é determinado recursivamente a partir dos
    # sinais de erro dos neurônios aos quais está conectado.
    delta_in = np.dot(w, delta_k)
    delta_j = delta_in * sigmoid_derivada(z_in)
    return delta_j


def calcular_correcao_pesos_escondida(delta_j, X, alpha):
    # Passo 7: Fausett (slide 72):
    # Calcula a correção de pesos e bias da camada escondida.
    #   Δvij = α * δj * xi
    #   Δv0j = α * δj
    # onde xi é o valor da entrada correspondente.
    delta_v = alpha * np.outer(X, delta_j)
    delta_v0 = alpha * delta_j
    return delta_v, delta_v0


def backward(X, t, z, z_in, y, y_in, w, alpha):
    # Executa os Passos 6 e 7 completos da backpropagation (slides 71-72).
    # Primeiro calcula o delta da camada de saída (Passo 6), depois
    # passa o erro para a camada escondida (Passo 7).
    # O bias w0 não participa da backpropagation do erro para camadas anteriores.
    delta_k = calcular_delta_saida(t, y, y_in)
    delta_w, delta_w0 = calcular_correcao_pesos_saida(delta_k, z, alpha)

    delta_j = calcular_delta_escondida(delta_k, w, z_in)
    delta_v, delta_v0 = calcular_correcao_pesos_escondida(delta_j, X, alpha)

    return delta_v, delta_v0, delta_w, delta_w0
