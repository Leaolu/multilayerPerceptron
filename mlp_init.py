# Funções de inicialização da rede MLP: pesos, carregamento de dados e
# divisão dos conjuntos de treino/validação/teste.
import numpy as np
import csv


def inicializar_pesos(n_entradas, n_escondidos, n_saidas):
    # Passo 0, Fausett (slide 68):
    # Inicializa pesos e bias com números aleatórios no intervalo (-1, 1).
    # slide 31 - Fausett: "Inicialize os pesos e bias – por simplicidade,
    # inicialize-os com 0 ou com números aleatórios no intervalo (0,1) ou (-1,1)"
    # v: pesos da camada de entrada para a camada escondida (n_entradas x n_escondidos)
    # v0: bias da camada escondida (vetor de tamanho n_escondidos)
    # w: pesos da camada escondida para a camada de saída (n_escondidos x n_saidas)
    # w0: bias da camada de saída (vetor de tamanho n_saidas)
    v = np.random.uniform(-1.0, 1.0, (n_entradas, n_escondidos))
    v0 = np.random.uniform(-1.0, 1.0, n_escondidos)
    w = np.random.uniform(-1.0, 1.0, (n_escondidos, n_saidas))
    w0 = np.random.uniform(-1.0, 1.0, n_saidas)
    return v, v0, w, w0


def carregar_dataset_csv(caminho, index_rotulo=-1):
    # Carrega um dataset genérico em csv
    X, T = [], []
    with open(caminho, 'r') as f:
        reader = csv.reader(f)
        for linha in reader:
            if not linha: continue
            atributos = [float(val) for val in linha[:index_rotulo]]
            rotulo = int(linha[index_rotulo])
            X.append(atributos)
            T.append(rotulo)
    return np.array(X), np.array(T)


def carregar_caracteres_completo(caminho_X, caminho_Y):
    # Carrega o dataset de reconhecimento de caracteres (26 classes, A até Z).
    # slides 41-43, Fausett: reconhecimento de caracteres como problema
    # de classificação resolvido com rede neural. Cada caractere é representado
    # por um vetor de 120 atributos (grid 10x12 de pixels bipolares).
    # X: matriz de padrões de entrada (cada linha = 1 caractere com 120 pixels)
    # T: vetor de rótulos inteiros (0=A, 1=B, ..., 25=Z)
    X = []
    with open(caminho_X, 'r') as f:
        for linha in f:
            valores = linha.replace(',', ' ').split()
            if valores:
                X.append([float(v) for v in valores])

    T = []
    with open(caminho_Y, 'r') as f:
        for linha in f:
            letra = linha.strip().upper()
            if letra:
                T.append(ord(letra) - ord('A'))

    return np.array(X), np.array(T)


def carregar_letras_com_buraco(caminho_X, caminho_Y):
    # Carrega o dataset para classificação binária: letras com buraco vs sem buraco.
    # Classe 1: letras que possuem região fechada (A, B, D, O, P, Q, R).
    # Classe 0: demais letras.
    # Problema de classificação binária com 2 neurônios de saída.
    X = []
    with open(caminho_X, 'r') as f:
        for linha in f:
            valores = linha.replace(',', ' ').split()
            if valores:
                X.append([float(v) for v in valores])

    T = []
    letras_com_buraco = {'A', 'B', 'D', 'O', 'P', 'Q', 'R'}

    with open(caminho_Y, 'r') as f:
        for linha in f:
            letra = linha.strip().upper()
            if letra:
                if letra in letras_com_buraco:
                    T.append(1)
                else:
                    T.append(0)

    return np.array(X), np.array(T)


def carregar_letras_com_curva(caminho_X, caminho_Y):
    # Carrega o dataset para classificação binária: letras com curva vs sem curva.
    # Classe 1: letras que tem curva (B, C, D, G, J, O, P, Q, R, S, U).
    # Classe 0: demais letras.
    # Problema de classificação binária com 2 neurônios de saída.
    X = []
    with open(caminho_X, 'r') as f:
        for linha in f:
            valores = linha.replace(',', ' ').split()
            if valores:
                X.append([float(v) for v in valores])

    T = []
    letras_com_curva = {'B', 'C', 'D', 'G', 'J', 'O', 'P', 'Q', 'R', 'S', 'U'}

    with open(caminho_Y, 'r') as f:
        for linha in f:
            letra = linha.strip().upper()
            if letra:
                if letra in letras_com_curva:
                    T.append(1)
                else:
                    T.append(0)

    return np.array(X), np.array(T)


def dividir_dados(X, T, qtd_teste=130, qtd_val=130):
    # Divide o dataset em três conjuntos: treino, validação e teste.
    # A validação é usada para monitorar o erro durante o treinamento
    # e implementar a parada antecipada (slide 82, Haykin) [técnica extra implementada].
    # O teste é usado apenas para avaliação final do modelo treinado.
    limite_teste = len(X) - qtd_teste
    limite_val = limite_teste - qtd_val

    X_treino, T_treino = X[:limite_val], T[:limite_val]
    X_val, T_val = X[limite_val:limite_teste], T[limite_val:limite_teste]
    X_teste, T_teste = X[limite_teste:], T[limite_teste:]

    return X_treino, T_treino, X_val, T_val, X_teste, T_teste


def salvar_pesos(v, v0, w, w0, caminho):
    with open(caminho, 'w') as f:
        f.write("Pesos v (Entrada -> Oculta):\n")
        np.savetxt(f, v, fmt="%.6f")
        f.write("\nBias v0 (Oculta):\n")
        np.savetxt(f, v0, fmt="%.6f")
        f.write("\nPesos w (Oculta -> Saída):\n")
        np.savetxt(f, w, fmt="%.6f")
        f.write("\nBias w0 (Saída):\n")
        np.savetxt(f, w0, fmt="%.6f")


def carregar_pesos(caminho):
    # Carrega pesos e bias previamente salvos para aplicação da rede.
    # slide 80 - Fausett: Admita os pesos e bias obtidos no
    # procedimento de treinamento.
    dados = np.load(caminho)
    return dados['v'], dados['v0'], dados['w'], dados['w0']


def salvar_historico_erro(erros_treino, caminho, erros_val=None):
    # Salva o histórico de erro quadrático médio por época em CSV.
    # Permite análise da curva de aprendizado e verificação
    # da convergência do treinamento.
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        if erros_val is not None:
            writer.writerow(["Epoca", "Erro_Treino", "Erro_Validacao"])
            for i, (et, ev) in enumerate(zip(erros_treino, erros_val)):
                writer.writerow([i+1, et, ev])
        else:
            writer.writerow(["Epoca", "Erro_Quadratico_Medio"])
            for i, erro in enumerate(erros_treino):
                writer.writerow([i+1, erro])
