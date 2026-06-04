# codigo principal para treinamento e avaliação do MLP
# Orquestra o pipeline completo: carregamento de dados, inicialização,
# treinamento com parada antecipada, avaliação no teste e teste autoral.
# slide 68 - Fausett: visão geral do algoritmo de treinamento do MLP.
import numpy as np
import argparse
import os
import time
from mlp_init import (inicializar_pesos, carregar_dataset_csv, carregar_caracteres_completo,
                      dividir_dados, salvar_historico_erro, salvar_pesos,
                      carregar_letras_com_buraco, carregar_letras_com_curva)
from mlp_forward import inferencia
from mlp_treino import treinar, calcular_erro_conjunto
from mlp_avaliacao import matriz_confusao, acuracia, plotar_curva_erro, salvar_saidas_teste, salvar_matriz_confusao, plotar_analise_completa_slides


def converter_para_one_hot(T, n_classes):
    # converte rótulos inteiros para codificação one-hot.
    # necessário pois a camada de saída do MLP possui um neurônio por classe
    # (slide 67 - Fausett: arquitetura com m unidades de saída Yk, k=1..m).
    T_one_hot = np.zeros((len(T), n_classes))
    for i, rotulo in enumerate(T):
        T_one_hot[i][rotulo] = 1.0
    return T_one_hot


def mapear_saida_para_texto(T_ints, caso):
    # mapeia os índices de classe para representação textual legível.
    # Caso multiclasse (26): índice 0 para A, 1 para B, ..., 25 para Z.
    # Caso binário (2): retorna 0 ou 1.
    if caso == 0:
        return [chr(int(val) + ord('A')) for val in T_ints]
    else:
        return [str(int(val)) for val in T_ints]


def main():
    parser = argparse.ArgumentParser(description="Treinamento de MLP")
    parser.add_argument("--n_escondidos", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_epocas", type=int, default=1500)
    parser.add_argument("--erro_minimo", type=float, default=0.01)
    parser.add_argument("--paciencia", type=int, default=20, help="Épocas de paciência para parada antecipada")
    parser.add_argument("--caso", type=int, default=0, help="0: Completo(26) | 1: Buraco(2) | 2: Curva(2)")
    args = parser.parse_args()

    print("\n--- Inicializando Rede Neural MLP ---")
    caso = args.caso

    # Carregamento do dataset conforme o caso selecionado.
    # Ref: slides 41-43 - Fausett: reconhecimento de caracteres como
    # problema de classificação com rede neural.
    if caso == 1:
        X, T = carregar_letras_com_buraco("X.txt", "Y_letra.txt")
    elif caso == 2:
        X, T = carregar_letras_com_curva("X.txt", "Y_letra.txt")
    else:
        X, T = carregar_caracteres_completo("X.txt", "Y_letra.txt")

    X_treino, T_treino, X_val, T_val, X_teste, T_teste = dividir_dados(X, T, qtd_teste=130, qtd_val=130)

    # Definição da arquitetura da rede (slide 67 - Fausett):
    # n_entradas: número de unidades fixo de entrada Xi (i=1..n), correspondente aos 120 pixels.
    # n_escondidos: número de unidades escondidas Zj (j=1..p), hiperparamêtro.
    # n_saidas: número de unidades de saída Yk (k=1..m), ajustável a depender do caso.
    n_entradas = 120
    n_escondidos = args.n_escondidos
    n_saidas = 26 if caso == 0 else 2
    alpha = args.alpha
    max_epocas = args.max_epocas
    erro_minimo = args.erro_minimo
    paciencia = args.paciencia

    T_treino_one_hot = converter_para_one_hot(T_treino, n_saidas)
    T_val_one_hot = converter_para_one_hot(T_val, n_saidas)
    T_teste_one_hot = converter_para_one_hot(T_teste, n_saidas)

    # Carregamento antecipado do Autoral para geração do Gráfico de Slide
    X_aut, T_aut_one_hot = None, None
    try:
        if caso == 1: X_aut_temp, T_aut_temp = carregar_letras_com_buraco("X_autoral.txt", "Y_autoral.txt")
        elif caso == 2: X_aut_temp, T_aut_temp = carregar_letras_com_curva("X_autoral.txt", "Y_autoral.txt")
        else: X_aut_temp, T_aut_temp = carregar_caracteres_completo("X_autoral.txt", "Y_autoral.txt")
        X_aut = X_aut_temp
        T_aut_one_hot = converter_para_one_hot(T_aut_temp, n_saidas)
    except FileNotFoundError:
        pass

    print(f"Configuração: Hidden={n_escondidos} | Alpha={alpha} | Max Épocas={max_epocas} | Paciência={paciencia}")
    print(f"Problema    : {'Multiclasse (26)' if caso == 0 else 'Binário (2)'}")
    print(f"Conjuntos   -> Treino: {len(X_treino)} | Validação: {len(X_val)} | Teste: {len(X_teste)}")

    # Passo 0: Fausett (slide 68): inicialização dos pesos e bias
    v, v0, w, w0 = inicializar_pesos(n_entradas, n_escondidos, n_saidas)
    salvar_pesos(v, v0, w, w0, "pesos_iniciais.txt")

    with open("hiperparametros.txt", "w", encoding="utf-8") as f:
        f.write("--- Hiperparametros da Rede ---\n")
        f.write(f"Entradas: {n_entradas} | Ocultas: {n_escondidos} | Saidas: {n_saidas}\n")
        f.write(f"Taxa de Aprendizado: {alpha} | Paciencia (Early Stop): {paciencia}\n")
        f.write(f"Maximo Epocas: {max_epocas} | Erro Minimo: {erro_minimo}\n")

    # Início da Cronometragem e Execução do treinamento
    tempo_inicio = time.time()
    print("Iniciando treinamento...")
    v, v0, w, w0, erros_treino, erros_val, erros_teste, erros_aut, p_v, p_w = treinar(X_treino, T_treino_one_hot, X_val, T_val_one_hot, X_teste, T_teste_one_hot, X_aut, T_aut_one_hot, v, v0, w, w0, alpha, max_epocas, erro_minimo, paciencia)
    
    tempo_execucao = time.time() - tempo_inicio
    with open("tempo_execucao.txt", "w") as f_tempo:
        f_tempo.write(f"{tempo_execucao:.4f}")

    plotar_analise_completa_slides(erros_treino, erros_val, erros_teste, erros_aut, p_v, p_w)

    salvar_pesos(v, v0, w, w0, "pesos_finais.txt")
    salvar_historico_erro(erros_treino, "historico_erros.csv", erros_val)

    # uso da rede treinada no conjunto de teste (slide 80 - Fausett):
    # pega os pesos obtidos e executa feedforward para cada padrão de teste
    T_predito_ints = inferencia(X_teste, v, v0, w, w0)
    acc = acuracia(T_teste, T_predito_ints)

    # Gera e salva a matriz de confusão do conjunto de teste principal
    matriz_teste = matriz_confusao(T_teste, T_predito_ints, n_saidas)
    salvar_matriz_confusao(matriz_teste, "matriz_confusao_teste.csv")

    epocas_executadas = len(erros_treino)
    erro_final = erros_treino[-1]

    print(f"\nResultados: Épocas={epocas_executadas} | Erro Treino Final={erro_final:.4f} | Acurácia no Teste={acc*100:.2f}%")
    print("\nMatriz de Confusão (Conjunto de Teste):")
    print(matriz_teste)

    T_teste_letras = mapear_saida_para_texto(T_teste, caso)
    T_predito_letras = mapear_saida_para_texto(T_predito_ints, caso)
    salvar_saidas_teste(X_teste, T_teste_letras, T_predito_letras, "saidas_teste.csv")

    # TESTE EXTRA: VARIAÇÃO AUTORAL
    # Avalia a rede treinada em um conjunto de dados com ruído feito por nós,
    # verificando a capacidade de generalização do modelo em um caso mais real
    try:
        print("\n--- Teste de Variação Autoral ---")

        if caso == 1:
            X_aut_final, T_aut_final = carregar_letras_com_buraco("X_autoral.txt", "Y_autoral.txt")
        elif caso == 2:
            X_aut_final, T_aut_final = carregar_letras_com_curva("X_autoral.txt", "Y_autoral.txt")
        else:
            X_aut_final, T_aut_final = carregar_caracteres_completo("X_autoral.txt", "Y_autoral.txt")

        T_aut_final_one_hot = converter_para_one_hot(T_aut_final, n_classes=n_saidas)

        erro_medio_aut_final = calcular_erro_conjunto(X_aut_final, T_aut_final_one_hot, v, v0, w, w0)

        T_predito_aut_ints = inferencia(X_aut_final, v, v0, w, w0)
        acc_autoral = acuracia(T_aut_final, T_predito_aut_ints)

        # Gera e salva a matriz de confusão do conjunto autoral
        matriz_autoral = matriz_confusao(T_aut_final, T_predito_aut_ints, n_saidas)
        salvar_matriz_confusao(matriz_autoral, "matriz_confusao_autoral.csv")

        print(f"Erro Médio no conjunto AUTORAL: {erro_medio_aut_final:.6f}")
        print(f"Acurácia no conjunto AUTORAL: {acc_autoral*100:.2f}%")
        print("\nMatriz de Confusão (Conjunto Autoral):")
        print(matriz_autoral)

        with open("resumo_teste_autoral.txt", "w") as f_resumo:
            f_resumo.write("=== Resultado do Teste Autoral ===\n")
            f_resumo.write(f"Erro Quadrático Médio: {erro_medio_aut_final:.6f}\n")
            f_resumo.write(f"Acurácia Alcançada   : {acc_autoral*100:.2f}%\n")
            f_resumo.write(f"Quantidade de Amostras: {len(X_aut_final)}\n")

        T_aut_letras = mapear_saida_para_texto(T_aut_final, caso)
        T_predito_aut_letras = mapear_saida_para_texto(T_predito_aut_ints, caso)
        salvar_saidas_teste(X_aut_final, T_aut_letras, T_predito_aut_letras, "saidas_letras_autoral.csv")

        print("Resultados gravados em 'resumo_teste_autoral.txt' e 'saidas_letras_autoral.csv'")

    except FileNotFoundError:
        print("\n[Aviso] Arquivos autorais não encontrados. Rode o 'gerar_ruido.py' primeiro se desejar este teste.")

    plotar_curva_erro(erros_treino, erros_val)

if __name__ == "__main__":
    main()