import numpy as np
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Constantes_fisicas

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

def calculate_L_c(c, AMSD_L_sq, log_GMD_L, AMD_L):
    # Usando a Equação (6) 
    term1 = np.log(np.sqrt(c**2 + AMSD_L_sq) + c)
    term2 = log_GMD_L
    term3 = np.sqrt(1 + (AMSD_L_sq / c**2))
    term4 = AMD_L / c
    return (Constantes_fisicas["Mu_0"] * c / (2 * np.pi)) * (term1 - term2 - term3 + term4)

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

def calculate_M_c(c, log_GMD, AMSD_sq, AMD):
    # Usando a Equação (24) 
    # Nota: No artigo original, a equação para a mútua é nomeada como vartheta_c.
    # A fórmula é a mesma da Equação (6), mas com sinais diferentes nos últimos dois termos.
    term1 = np.log(np.sqrt(c**2 + AMSD_sq) + c)
    term2 = log_GMD
    term3 = np.sqrt(1 + (AMSD_sq / c**2))
    term4 = AMD / c
    return (Constantes_fisicas["Mu_0"] * c / (2 * np.pi)) * (term1 - term2 - term3 + term4)

def calculate_rectangular_spiral_inductance(a, b, n, s, g, h):
    """
    Calcula a indutância de uma bobina espiral planar retangular.

    Args:
        a (float): Comprimento do lado externo mais longo [m].
        b (float): Comprimento do lado externo mais curto [m].
        n (int): Número de voltas ou enrolamentos.
        s (float): Largura do condutor [m].
        g (float): Espaçamento entre os condutores [m].
        h (float): Altura/espessura do condutor [m].

    Returns:
        h (float): A indutância calculada [H].
    """

    # Garantindo que A>B
    if b > a:
        a, b = b, a

    if n < 2:
        # O número de espiras deve ser maior que 2
        print("deu ruim 1")
        return 1e-12

    # Parâmetros derivados
    w = s + g  # Passo do enrolamento (winding pitch) 
    kappa = w / s
    gamma = s / h

    # Comprimentos médios dos condutores 
    # 'a' e 'b' são os comprimentos médios dos segmentos do condutor.
    a = a - (n - 1) * w  # Equação (3) modificada
    b = b - (n - 1) * w  # Equação (4)

    # --- Etapa 2: Verificação de Validade da Geometria ---
    if (b - (n - 1) * w) <= 0:
        print("deu ruim 2")
        return 1e-12
    rho = ((n - 1) * w + s) / (b - (n - 1) * w)
    if n == 2 and rho > 0.36001: 
        print("deu ruim 3")
        return 1e-12
    if 3 <= n <= 7 and rho > 0.52001: 
        print("deu ruim 4")
        return 1e-12
    if 8 <= n <= 12 and rho > 0.78001: 
        print("deu ruim 5")
        return 1e-12
    if 13 <= n <= 20 and rho > 0.86001: 
        print("deu ruim 6")
        return 1e-12
    if n >= 21 and rho > (n - 1) / (n + 1): 
        print("deu ruim 7")
        return 1e-12
    
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
    k_vals = np.arange(1, n) # k vai de 1 a N-1

    # log(GMD_L) --- Equação (12) 
    sum_log_GMD2 = np.sum((n - k_vals) * np.array([log_GMD2_func(k, w, s, h, correction) for k in k_vals]))
    log_GMD_L = (1 / n**2) * (n * log_GMD1 + 2 * sum_log_GMD2)

    # AMSD_L^2 --- Equação (14) 
    sum_AMSD2_sq = np.sum((n - k_vals) * np.array([AMSD2_sq_func(k, w) for k in k_vals]))
    AMSD_L_sq = (1 / n**2) * (n * AMSD1_sq + 2 * sum_AMSD2_sq)

    # AMD_L --- Equação (16) 
    sum_AMD2 = np.sum((n - k_vals) * np.array([AMD2_func(k, w, s, h, correction) for k in k_vals]))
    AMD_L = (1 / n**2) * (n * AMD1 + 2 * sum_AMD2)

    # --- Cálculo das Autoindutâncias Parciais L_a e L_b ---
    L_a = calculate_L_c(a, AMSD_L_sq, log_GMD_L, AMD_L)
    L_b = calculate_L_c(b, AMSD_L_sq, log_GMD_L, AMD_L)

    # --- 4.3. Expressão aproximada para as indutâncias mútuas ---
    log_GMD_M_b, AMSD_M_b_sq, AMD_M_b = calculate_mutual_means(a, w, n) # Mútua em relação ao lado 'a' está a uma distância 'b'
    log_GMD_M_a, AMSD_M_a_sq, AMD_M_a = calculate_mutual_means(b, w, n) # Mútua em relação ao lado 'b' está a uma distância 'a'

    # --- Cálculo das Indutâncias Mútuas Parciais M_a e M_b ---
    # M_a é a mútua entre os lados de comprimento 'a', que estão separados por 'b'
    M_a = calculate_M_c(a, log_GMD_M_a, AMSD_M_a_sq, AMD_M_a)
    # M_b é a mútua entre os lados de comprimento 'b', que estão separados por 'a'
    M_b = calculate_M_c(b, log_GMD_M_b, AMSD_M_b_sq, AMD_M_b)

    # --- 4.4. A Fórmula da Indutância Total ---
    # L = 2 * N^2 * [L_a + L_b - (M_a + M_b)] --- Equação (30) 
    L_total = 2 * (n**2) * (L_a + L_b - (M_a + M_b))

    return L_total

def calculate_ipt_coil_inductance_approx(S,N,A,B,g):
    """
    Calcula a indutância aproximada de uma bobina circular para IPT,
    mapeando sua geometria para uma bobina quadrada equivalente.

    Args:
        S (float): Seção do condutor [mm^2]. (?)
        N (int): Número de espiras. (?)
        A (float): Comprimento do lado externo mais longo [m].
        B (float): Comprimento do lado externo mais curto [m].
        g (float): Espaçamento entre os fios (gap) [m].

    Returns:
        L float: A indutância aproximada [H].
    """
    d = 2 * np.sqrt((A*B)/(np.pi))

    # 2. Fazer as aproximações de geometria
    # Bobina circular -> quadrada
    A = d
    B = d

    # Fio circular -> quadrado com mesma área
    d_condutor = np.sqrt(S)
    s = d_condutor                # Largura do condutor
    h = s                         # Altura do condutor
    
    # 3. Chamar a função principal com os parâmetros aproximados
    L = calculate_rectangular_spiral_inductance(A, B, N, s, g, h)

    return L

# --- Exemplo de Uso ---
if __name__ == '__main__':
    #  # --- PARTE 1: Verificação com os exemplos do Artigo (Tabela 3) ---
    # print("--- Validação da Implementação com Dados da Tabela 3 do Artigo ---")

    # # Parâmetros de entrada comuns a todos os testes da Tabela 3
    # A_artigo = 0.1      # m lado maior
    # B_artigo = 0.05     # m lado menor
    # w_artigo = 1e-3     # m pitch, distancia entre condutores incluindo expessura
    # s_artigo = 5e-4     # m expessura do condutor (S?)
    # h_artigo = 35e-6    # m altura do condutor

    # # A função precisa do gap 'g', não do pitch 'w'. g = w - s
    # g_artigo = w_artigo - s_artigo # espaçamento entre condutores

    # # Dados da Tabela 3 para verificação [N, L_matlab (uH)]
    # testes_tabela3 = [
    #     (2, 1.064),
    #     (5, 4.785),
    #     (10, 13.525),
    #     (15, 22.624)
    # ]

    # # Loop através dos testes para validação
    # for N_teste, L_matlab_uH in testes_tabela3:
    #     # Chamar a implementação em Python
    #     L_python_H = calculate_rectangular_spiral_inductance(
    #         a=A_artigo,
    #         b=B_artigo,
    #         n=N_teste,
    #         s=s_artigo,
    #         g=g_artigo,
    #         h=h_artigo,
    #     )
        
    #     # Converter para uH
    #     L_python_uH = L_python_H * 1e6

    #     # Calcular o erro relativo entre a implementação Python e o resultado MATLAB do artigo
    #     erro_relativo_percent = abs((L_python_uH - L_matlab_uH) / L_matlab_uH) * 100

    #     print(f"Para N = {N_teste}:")
    #     print(f"  > Resultado do Artigo (MATLAB): {L_matlab_uH:.3f} uH")
    #     print(f"  > Resultado deste Script (Python): {L_python_uH:.3f} uH")
    #     print(f"  > Erro Relativo (Python vs MATLAB): {erro_relativo_percent:.4f}%")
    #     print("")

    # print("\n" + "="*60 + "\n")

    # print("--- Validação da Bobina 'Coil 1' (Tabela 4 e 5 do Artigo SAE JA2954) ---")
    # print("Objetivo: Replicar o valor de indutância de 203.68 uH.\n")

    # A = 0.65
    # B = 0.50
    # N = 15
    # T_d = 1.5e-3
    # n_t = 280
    # d_t = 0.2e-3

    # packing_factor = 0.5
    # area_cobre_total = n_t * math.pi * (d_t / 2)**2
    # area_total_cabo = area_cobre_total / packing_factor
    # s = math.sqrt(area_total_cabo)
    # h = s
    # g = T_d

    # A = A - s*7
    # B = B - s*7
    # # PORQUE 7, DESCUBRA NO PROXIMO EPISODIO

    # print("Parâmetros Interpretados para o Cálculo (COM FATOR DE EMPACOTAMENTO):")
    # print(f"  - Área total do cabo (com PF={packing_factor}): {area_total_cabo * 1e6:.4f} mm^2")
    # print(f"  - Largura do condutor equivalente (s): {s * 1e3:.4f} mm")
    # print(f"  - Altura do condutor equivalente (h): {h * 1e3:.4f} mm")
    # print(f"  - Espaçamento entre condutores (g): {g * 1e3:.4f} mm\n")

    # L_calculada_H = calculate_rectangular_spiral_inductance(a=A, b=B, n=N, s=s, g=g, h=h)
    # L_calculada_uH = L_calculada_H * 1e6
    # L_artigo_uH = 203.68

    # erro_relativo = abs((L_calculada_uH - L_artigo_uH) / L_artigo_uH) * 100

    # print("--- Resultados da Validação Final ---")
    # print(f"Indutância alvo do Artigo (Tabela 5): {L_artigo_uH:.2f} uH")
    # print(f"Indutância calculada (script final): {L_calculada_uH:.2f} uH")
    # print(f"Erro relativo final: {erro_relativo:.2f}%")
    
    print("--- Validação da Bobina 'Coil 2' (Tabela 4 e 5 do Artigo SAE JA2954) ---")
    print("Objetivo: Replicar o valor de indutância de 112.67 uH.\n")

    A = 0.38
    B = 0.38
    N = 16
    T_d = 1e-3
    n_t = 280
    d_t = 0.2e-3

    packing_factor = 0.5
    area_cobre_total = n_t * math.pi * (d_t / 2)**2 # s = nt * pi * (dt/2)^2
    area_total_cabo = area_cobre_total / packing_factor
    s = math.sqrt(area_total_cabo)
    h = s
    g = T_d

    A = A - s* (N/2 -1)
    B = B - s* (N/2 -1)

    print("Parâmetros Interpretados para o Cálculo (COM FATOR DE EMPACOTAMENTO):")
    print(f"  - Área total do cabo (com PF={packing_factor}): {area_total_cabo * 1e6:.4f} mm^2")
    print(f"  - Largura do condutor equivalente (s): {s * 1e3:.4f} mm")
    print(f"  - Altura do condutor equivalente (h): {h * 1e3:.4f} mm")
    print(f"  - Espaçamento entre condutores (g): {g * 1e3:.4f} mm\n")

    L_calculada_H = calculate_rectangular_spiral_inductance(a=A, b=B, n=N, s=s, g=g, h=h)
    L_calculada_uH = L_calculada_H * 1e6
    L_artigo_uH = 112.67

    erro_relativo = abs((L_calculada_uH - L_artigo_uH) / L_artigo_uH) * 100

    print("--- Resultados da Validação Final ---")
    print(f"Indutância alvo do Artigo (Tabela 5): {L_artigo_uH:.2f} uH")
    print(f"Indutância calculada (script final): {L_calculada_uH:.2f} uH")
    print(f"Erro relativo final: {erro_relativo:.2f}%")
