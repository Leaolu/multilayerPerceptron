# gerar_ruido.py
# Gera uma variação ruidosa do dataset original aplicando ruído aleatório (15% dos pixels)
# aos pixels dos caracteres.
# Ref: slide 42, Fausett: conjunto de teste com variações dos caracteres
# originais para avaliar a capacidade de generalização da rede.
# 
# Estratégia:
# - Extrai um alfabeto completo (26 letras) da base original
# - Gera MÚLTIPLAS instâncias ruidosas de cada letra (5 réplicas cada)
# - Total: 26 × 5 = 130 amostras
# - Objetivo: Tamanho comparável ao conjunto de teste (130 amostras)
# - Cada réplica tem ruído aleatório DIFERENTE para representar variabilidade
import numpy as np
from mlp_init import carregar_caracteres_completo


def gerar_base_ruidosa():
    """
    Gera uma base ruidosa com múltiplas instâncias de cada letra.
    
    Estrutura:
    - Seleciona um alfabeto completo (26 letras únicas)
    - Para cada letra: gera 5 réplicas com ruído aleatório diferente
    - Total: 26 × 5 = 130 amostras (compatível com tamanho do teste)
    - Taxa de ruído: 15% dos pixels invertidos por amostra
    
    Vantagens:
    ✓ Tamanho igual ao conjunto de teste original (130 amostras)
    ✓ Variabilidade de ruído: cada letra tem 5 versões diferentes
    ✓ Teste mais representativo de generalização
    """
    print("╔" + "="*70 + "╗")
    print("║" + " Gerando Base de Dados Ruidosa ".center(70) + "║")
    print("╚" + "="*70 + "╝\n")
    
    print("1️⃣  Carregando base original...")
    X, T = carregar_caracteres_completo("X.txt", "Y_letra.txt")

    # Seleciona as últimas 26 amostras: um alfabeto completo (A-Z)
    X_base = X[-26:].copy()
    T_base = T[-26:]
    
    total_letras = len(X_base)
    replicas_por_letra = 5
    taxa_ruido = 0.15

    print(f"\n2️⃣  Configuração:")
    print(f"    - Letras únicas: {total_letras}")
    print(f"    - Réplicas por letra: {replicas_por_letra}")
    print(f"    - Taxa de ruído por instância: {taxa_ruido*100}%")
    print(f"    - Total de amostras ruidosas: {total_letras * replicas_por_letra}")
    
    print(f"\n3️⃣  Gerando {replicas_por_letra} instâncias ruidosas de cada letra (A-Z)...")

    X_ruidoso = []
    T_ruidoso = []
    
    # Para cada letra do alfabeto completo
    for letra_idx in range(total_letras):
        letra_original = X_base[letra_idx].copy()
        rotulo = T_base[letra_idx]
        
        # Gerar 5 réplicas com ruído aleatório diferente
        for replica in range(replicas_por_letra):
            letra_com_ruido = letra_original.copy()
            
            # Aplicar ruído: invertir pixels aleatoriamente
            for pixel_idx in range(len(letra_com_ruido)):
                if np.random.rand() < taxa_ruido:
                    # Invertir pixel bipolar: +1 → -1 ou -1 → +1
                    letra_com_ruido[pixel_idx] *= -1
            
            X_ruidoso.append(letra_com_ruido)
            T_ruidoso.append(rotulo)

    # Converter para numpy arrays
    X_ruidoso = np.array(X_ruidoso)
    T_ruidoso = np.array(T_ruidoso)

    # Salvar X_ruidoso.txt
    print(f"\n4️⃣  Salvando dados ruidosos...")
    with open("X_ruidoso.txt", "w") as f:
        for linha in X_ruidoso:
            linha_formatada = ",".join([str(int(val)) for val in linha])
            f.write(linha_formatada + "\n")

    # Salvar Y_ruidoso.txt
    with open("Y_ruidoso.txt", "w") as f:
        for rotulo in T_ruidoso:
            letra = chr(int(rotulo) + ord('A'))
            f.write(letra + "\n")

    print("    ✓ X_ruidoso.txt salvo")
    print("    ✓ Y_ruidoso.txt salvo")

    # Informações finais
    print("\n" + "="*70)
    print("✅ Base de dados ruidosa gerada com sucesso!")
    print("="*70)
    print(f"\n📊 Resumo:")
    print(f"   Total de amostras ruidosas: {len(X_ruidoso)}")
    print(f"   Distribuição: {replicas_por_letra} réplicas × {total_letras} letras")
    print(f"   Cada letra (A-Z) tem 5 versões diferentes com ruído")
    print(f"\n📏 Comparação com conjunto de teste:")
    print(f"   Conjunto de teste original:  130 amostras")
    print(f"   Conjunto de teste ruidoso:   {len(X_ruidoso)} amostras")
    print(f"   ✓ Agora são COMPARÁVEIS! Teste justo e representativo.")
    print("="*70 + "\n")


if __name__ == "__main__":
    gerar_base_ruidosa()
