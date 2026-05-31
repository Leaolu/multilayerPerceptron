# gerar_ruido.py
# Gera uma variação autoral do dataset original aplicando ruído aleatório de acordo com a taxa de ruido (15% no caso)
# aos pixels dos caracteres.
# Ref: slide 42, Fausett: conjunto de teste com variações dos caracteres
# originais para avaliar a capacidade de generalização da rede.
import numpy as np
from mlp_init import carregar_caracteres_completo


def gerar_base_autoral():
    print("Carregando base original...")
    X, T = carregar_caracteres_completo("X.txt", "Y_letra.txt")

    # Seleciona as últimas 26 amostras do arquivo X.txt, fornecido no edisciplinas, (um alfabeto completo) como base
    X_base = X[-26:].copy()
    T_base = T[-26:]

    # variavel da taxa de ruido que implementa o percentual de pixels que serão corrompidos (invertidos)
    taxa_ruido = 0.15

    print(f"Aplicando {taxa_ruido*100}% de ruído autoral...")
    for i in range(len(X_base)):
        for j in range(len(X_base[i])):
            # para cada pixel, sorteia se será corrompido com base na taxa
            if np.random.rand() < taxa_ruido:
                # inverte o sinal do pixel bipolar (+1 -> -1 ou -1 -> +1)
                X_base[i][j] *= -1

    # salva o conjunto de atributos com ruido no formato original (CSV)
    with open("X_autoral.txt", "w") as f:
        for linha in X_base:
            linha_formatada = ",".join([str(int(val)) for val in linha])
            f.write(linha_formatada + "\n")

    # Salva os rotulos em formato de letras (A-Z)
    with open("Y_autoral.txt", "w") as f:
        for rotulo in T_base:
            letra = chr(int(rotulo) + ord('A'))
            f.write(letra + "\n")

    print("Arquivos 'X_autoral.txt' e 'Y_autoral.txt' gerados com sucesso!")
    print("Formato mantido idêntico ao original (separado por vírgulas e em inteiros).")

if __name__ == "__main__":
    gerar_base_autoral()
