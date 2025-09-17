import numpy as np
import math

# Constante da permeabilidade magnética do vácuo
mu_0 = 4 * np.pi * 1e-7  # V·s/(A·m)

def log_GMD2_func(k_idx, w, s, h, correction):
    # Usando kappa (κ) no lugar de k para evitar conflito com o índice do somatório
    kappa_local = (k_idx * w) / s
    return np.log(s + h) + np.log(kappa_local / 2) - correction

def AMSD2_sq_func(k_idx, w):
    # A função AMSD2_sq_func será usada dentro do somatório
    return (k_idx * w)**2

def AMD2_func(k_idx, w, s, h, correction):
    # A função AMD2_func será usada dentro do somatório
    return np.exp(log_GMD2_func(k_idx, w, s, h, correction))

def calculate_L_c(c, AMSD_L_sq, log_GMD_L, AMD_L, mu_0):
    # Usando a Equação (6) 
    term1 = np.log(np.sqrt(c**2 + AMSD_L_sq) + c)
    term2 = log_GMD_L
    term3 = np.sqrt(1 + (AMSD_L_sq / c**2))
    term4 = AMD_L / c
    return (mu_0 * c / (2 * np.pi)) * (term1 - term2 - term3 + term4)

def calculate_mutual_means(c_bar, w, N):
    # Cálculo das distâncias médias compostas para Indutância Mútua (M_a, M_b)
    # Aproximação dos filamentos centrais
    k_vals_mutual = np.arange(-(N - 1), N) # k vai de -(N-1) a N-1
    # log(GMD_c_bar) --- Equação (27) 
    sum_log_GMD = np.sum((N - np.abs(k_vals_mutual)) * np.log(c_bar + k_vals_mutual * w))
    log_GMD_c_bar = (1 / N**2) * sum_log_GMD
    # AMSD_c_bar^2 --- Equação (28) 
    sum_AMSD_sq = np.sum((N - np.abs(k_vals_mutual)) * (c_bar + k_vals_mutual * w)**2)
    AMSD_c_bar_sq = (1 / N**2) * sum_AMSD_sq
    # AMD_c_bar --- Equação (29) 
    sum_AMD = np.sum((N - np.abs(k_vals_mutual)) * (c_bar + k_vals_mutual * w))
    AMD_c_bar = (1 / N**2) * sum_AMD
    return log_GMD_c_bar, AMSD_c_bar_sq, AMD_c_bar

def calculate_M_c(c, log_GMD, AMSD_sq, AMD, mu_0):
    # Usando a Equação (24) 
    # Nota: No artigo original, a equação para a mútua é nomeada como vartheta_c.
    # A fórmula é a mesma da Equação (6), mas com sinais diferentes nos últimos dois termos.
    term1 = np.log(np.sqrt(c**2 + AMSD_sq) + c)
    term2 = log_GMD
    term3 = np.sqrt(1 + (AMSD_sq / c**2))
    term4 = AMD / c
    return (mu_0 * c / (2 * np.pi)) * (term1 - term2 - term3 + term4)

def calculate_rectangular_spiral_inductance(A, B, N, s, g, h,mu_0):
    """
    Calcula a indutância de uma bobina espiral planar retangular com base no artigo
    "Inductance Formula for Rectangular Planar Spiral Inductors".

    Args:
        A (float): Comprimento do lado externo mais longo (em metros).
        B (float): Comprimento do lado externo mais curto (em metros).
        N (int): Número de voltas ou enrolamentos.
        s (float): Largura do condutor (em metros).
        g (float): Espaçamento entre os condutores (em metros).
        h (float): Altura/espessura do condutor (em metros).

    Returns:
        float: A indutância calculada (em Henrys).
    """
    # Garantindo que A>B
    if B > A:
        A, B = B, A

    if N < 2:
        raise ValueError("O número de espiras (N) deve ser 2 ou maior.")


    # Parâmetros derivados
    w = s + g  # Passo do enrolamento (winding pitch) 
    kappa = w / s
    gamma = s / h

    # Comprimentos médios dos condutores 
    # 'a' e 'b' são os comprimentos médios dos segmentos do condutor.
    a = A - (N - 1) * w  # Equação (3) modificada
    b = B - (N - 1) * w  # Equação (4)

    # --- Etapa 2: Verificação de Validade da Geometria ---
    if (B - (N - 1) * w) <= 0: return 1e-12 
    rho = ((N - 1) * w + s) / (B - (N - 1) * w)
    if N == 2 and rho > 0.36001: return 1e-12
    if 3 <= N <= 7 and rho > 0.52001: return 1e-12
    if 8 <= N <= 12 and rho > 0.78001: return 1e-12
    if 13 <= N <= 20 and rho > 0.86001: return 1e-12
    if N >= 21 and rho > (N - 1) / (N + 1): return 1e-12

    # --- 4.2. Expressões aproximadas para as autoindutâncias parciais ---

    # Aproximações para GMD (Geometric Mean Distance)
    # log(GMD1) ~ log(s+h) - 3/2 --- Equação (18) 
    log_GMD1 = np.log(s + h) - 1.5

    # log(GMD2) ~ log(s+h) + log(k/2) - ... --- Equação (20) 
    # Usando kappa (κ) no lugar de k para evitar conflito com o índice do somatório
    correction = (-1.46 * gamma + 1.45) / (2.14 * gamma + 1)

    # Aproximações para AMSD (Arithmetic Mean Square Distance)
    # AMSD1^2 = (s^2 + h^2)/6 --- Equação (43) 
    AMSD1_sq = (s**2 + h**2) / 6.0

    # Aproximações para AMD (Arithmetic Mean Distance)
    # AMD1 ~ GMD1 ~ 0.2235(s+h) --- Equações (22) e (17) 
    AMD1 = 0.2235 * (s + h)

    # --- Cálculo das distâncias médias compostas para Autoindutância (L_a, L_b) ---
    k_vals = np.arange(1, N) # k vai de 1 a N-1

    # log(GMD_L) --- Equação (12) 
    sum_log_GMD2 = np.sum((N - k_vals) * np.array([log_GMD2_func(k, w, s, h, correction) for k in k_vals]))
    log_GMD_L = (1 / N**2) * (N * log_GMD1 + 2 * sum_log_GMD2)

    # AMSD_L^2 --- Equação (14) 
    sum_AMSD2_sq = np.sum((N - k_vals) * np.array([AMSD2_sq_func(k, w) for k in k_vals]))
    AMSD_L_sq = (1 / N**2) * (N * AMSD1_sq + 2 * sum_AMSD2_sq)

    # AMD_L --- Equação (16) 
    sum_AMD2 = np.sum((N - k_vals) * np.array([AMD2_func(k, w, s, h, correction) for k in k_vals]))
    AMD_L = (1 / N**2) * (N * AMD1 + 2 * sum_AMD2)

    # --- Cálculo das Autoindutâncias Parciais L_a e L_b ---
    L_a = calculate_L_c(a, AMSD_L_sq, log_GMD_L, AMD_L, mu_0)
    L_b = calculate_L_c(b, AMSD_L_sq, log_GMD_L, AMD_L, mu_0)

    # --- 4.3. Expressão aproximada para as indutâncias mútuas ---
    log_GMD_M_b, AMSD_M_b_sq, AMD_M_b = calculate_mutual_means(a, w, N) # Mútua em relação ao lado 'a' está a uma distância 'b'
    log_GMD_M_a, AMSD_M_a_sq, AMD_M_a = calculate_mutual_means(b, w, N) # Mútua em relação ao lado 'b' está a uma distância 'a'


    # --- Cálculo das Indutâncias Mútuas Parciais M_a e M_b ---
    # M_a é a mútua entre os lados de comprimento 'a', que estão separados por 'b'
    M_a = calculate_M_c(a, log_GMD_M_a, AMSD_M_a_sq, AMD_M_a, mu_0)
    # M_b é a mútua entre os lados de comprimento 'b', que estão separados por 'a'
    M_b = calculate_M_c(b, log_GMD_M_b, AMSD_M_b_sq, AMD_M_b, mu_0)


    # --- 4.4. A Fórmula da Indutância Total ---
    # L = 2 * N^2 * [L_a + L_b - (M_a + M_b)] --- Equação (30) 
    L_total = 2 * (N**2) * (L_a + L_b - (M_a + M_b))

    return L_total

def calculate_ipt_coil_inductance_approx(D_outer, N, g, l1, l2,mu_0):
    """
    Calcula a indutância aproximada de uma bobina circular para IPT,
    mapeando sua geometria para uma bobina quadrada equivalente e usando o modelo do artigo.

    Args:
        D_outer (float): Diâmetro externo da bobina circular (em metros).
        N (int): Número de espiras.
        g (float): Espaçamento entre os fios (gap) (em metros).
        l1, l2 (float): Parâmetros para o cálculo do diâmetro do fio (unidades consistentes).

    Returns:
        float: A indutância aproximada (em Henrys).
    """
    # 1. Calcular o diâmetro do fio 'd' com a fórmula fornecida
    d = 2 * np.sqrt((l1 * l2) / np.pi)

    # 2. Fazer as aproximações de geometria
    # Bobina circular -> quadrada
    A = D_outer
    B = D_outer

    # Fio circular -> quadrado com mesma área
    s = (np.sqrt(np.pi) / 2) * d  # Largura do condutor
    h = s                        # Altura do condutor

    print("--- Parâmetros da Bobina de IPT (Circular) ---")
    print(f"Diâmetro Externo (D_outer): {D_outer:.4f} m")
    print(f"Número de Espiras (N): {N}")
    print(f"Espaçamento (g): {g:.4f} m")
    print(f"Diâmetro do fio calculado (d): {d:.6f} m")
    print("\n--- Parâmetros Mapeados para o Modelo (Quadrado Equivalente) ---")
    print(f"Lado A: {A:.4f} m")
    print(f"Lado B: {B:.4f} m")
    print(f"Largura do condutor (s): {s:.6f} m")
    print(f"Altura do condutor (h): {h:.6f} m")

    # 3. Chamar a função principal com os parâmetros aproximados
    inductance = calculate_rectangular_spiral_inductance(A, B, N, s, g, h,mu_0)

    return inductance

# --- Exemplo de Uso ---
if __name__ == '__main__':
     # --- PARTE 1: Verificação com os exemplos do Artigo (Tabela 3) ---
    print("--- Validação da Implementação com Dados da Tabela 3 do Artigo ---")

    # Parâmetros de entrada comuns a todos os testes da Tabela 3
    A_artigo = 0.1      # m
    B_artigo = 0.05     # m
    w_artigo = 1e-3     # m (pitch)
    s_artigo = 5e-4     # m
    h_artigo = 35e-6    # m

    # A função precisa do gap 'g', não do pitch 'w'. g = w - s
    g_artigo = w_artigo - s_artigo

    print("\nParâmetros de Entrada (Artigo):")
    print(f"  A = {A_artigo*1000} mm, B = {B_artigo*1000} mm, s = {s_artigo*1000} mm, g = {g_artigo*1000} mm, h = {h_artigo*1e6} um")
    print("-" * 50)

    # Dados da Tabela 3 para verificação [N, L_matlab (uH)]
    testes_tabela3 = [
        (2, 1.064),
        (5, 4.785),
        (10, 13.525),
        (15, 22.624)
    ]

    # Loop através dos testes para validação
    for N_teste, L_matlab_uH in testes_tabela3:
        # Chamar a implementação em Python
        L_python_H = calculate_rectangular_spiral_inductance(
            A=A_artigo,
            B=B_artigo,
            N=N_teste,
            s=s_artigo,
            g=g_artigo,
            h=h_artigo,
            mu_0=mu_0
        )
        # Converter para uH
        L_python_uH = L_python_H * 1e6

        # Calcular o erro relativo entre a implementação Python e o resultado MATLAB do artigo
        erro_relativo_percent = abs((L_python_uH - L_matlab_uH) / L_matlab_uH) * 100

        print(f"Para N = {N_teste}:")
        print(f"  > Resultado do Artigo (MATLAB): {L_matlab_uH:.3f} uH")
        print(f"  > Resultado deste Script (Python): {L_python_uH:.3f} uH")
        print(f"  > Erro Relativo (Python vs MATLAB): {erro_relativo_percent:.4f}%")
        print("")

    print("\n" + "="*60 + "\n")
