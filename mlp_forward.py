# Implementação da fase feedforward do MLP (Passos 3, 4 e 5 do algoritmo
# de treinamento descrito por Fausett).
# slides 68-70 do slide "PerceptronSimplesMLP.pdf"
import numpy as np


def sigmoid(x):
    # Função de ativação sigmoide.
    # f(x) = 1 / (1 + exp(-x))
    # slide 52, Fausett: função logistica comumente usada no MLP,
    # não linear e diferenciável em todos os seus pontos.
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivada(x):
    # Derivada da função sigmoide.
    # f'(x) = f(x) * (1 - f(x))
    # slide 52, Fausett: propriedade essencial para o cálculo do
    # gradiente local durante a retropropagação do erro.
    f = sigmoid(x)
    return f * (1.0 - f)


def forward_camada_escondida(X, v, v0):
    # Passo 4, Fausett (slide 70):
    # Cada unidade escondida (Zj, j=1..p) soma suas entradas ponderadas
    # e aplica a função de ativação para computar seu sinal de saída.
    #   z_inj = v0j + sum_i(xi * vij)
    #   zj = f(z_inj)
    # v0 é o bias da camada escondida, v são os pesos entrada para a escondida.
    z_in = v0 + np.dot(X, v)
    z = sigmoid(z_in)
    return z_in, z


def forward_camada_saida(z, w, w0):
    # Passo 5, Fausett (slide 70):
    # Cada unidade de saída (Yk, k=1..m) soma suas entradas ponderadas
    # e aplica a função de ativação para computar seu sinal de saída.
    #   y_ink = w0k + sum_j(zj * wjk)
    #   yk = f(y_ink)
    # w0 é o bias da camada de saída, w são os pesos escondida para a saída.
    y_in = w0 + np.dot(z, w)
    y = sigmoid(y_in)
    return y_in, y


def forward(X, v, v0, w, w0):
    # Executa o estágio feedforward completo (Passos 3, 4 e 5).
    # slide 68 - Fausett: o treinamento envolve três estágios,
    # sendo o primeiro a passagem feedforward dos dados.
    # Passo 3: a unidade de entrada Xi recebe xi e dissipa para a próxima camada
    # Passo 4: camada escondida processa
    # Passo 5: camada de saída processa
    z_in, z = forward_camada_escondida(X, v, v0)
    y_in, y = forward_camada_saida(z, w, w0)
    return z_in, z, y_in, y


def inferencia(X, v, v0, w, w0):
    # Procedimento de inferencia da rede treinada (slide 80 - Fausett):
    # Admite os pesos e bias obtidos no treinamento e, para cada padrão
    # de entrada, executa o feedforward e retorna a classe predita via index do maior valor.
    predicoes = []
    for xi in X:
        _, _, _, y = forward(xi, v, v0, w, w0)
        predicoes.append(np.argmax(y))
    return np.array(predicoes)
