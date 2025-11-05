"""
Script Python para calcular a resistência de um fio Litz com base nas equações
apresentadas na dissertação "PROJETO DO ACOPLAMENTO INDUTIVO DE UM
SISTEMA DE TRANSFERÊNCIA DE ENERGIA SEM FIO PARA A RECARGA DE VEÍCULOS ELÉTRICOS"
por Pedro Gelati Pascoal (UFSM, 2023).

O cálculo principal baseia-se na Equação 4.17 e suas equações de apoio
do Capítulo 4.
"""

import math
RHO_COBRE = 1.72e-8
K_BW = 1.03

# --- Funções auxiliares baseadas no documento ---

def calcular_k_nb(n_b: int) -> float:
    """
    Calcula o fator de correção k_nb conforme a Equação 4.18.

    A equação original é: k_nb = 1 + sum_{i=1}^{n_b} * 0.02
    Isso simplifica para: k_nb = 1 + (n_b * 0.02)

    Args:
        n_b (int): O número de "bunches" ou feixes (parâmetro 'nb' da
                   fórmula). Este valor não foi
                   explicitamente definido nas páginas fornecidas do
                   documento e deve ser obtido da norma
                   (DIN; EN; IEC, 2000) [cite_start][cite: 277].

    Returns:
        float: O valor do fator k_nb.
    """
    return 1 + (n_b * 0.02)

def calcular_area_strand(D_str: float) -> float:
    """
    [cite_start]Calcula a área de um único strand (filamento) conforme a Equação 4.16[cite: 275].

    Args:
        [cite_start]D_str (float): Diâmetro de cada strand (em metros)[cite: 276].

    Returns:
        float: A área do strand (em metros quadrados).
    """
    return (math.pi * (D_str ** 2)) / 4

def calcular_diametro_interno(D_o: float, w: float, N: int, s: float) -> float:
    """
    [cite_start]Calcula o diâmetro interno (Di) da bobina conforme a Equação 4.12[cite: 248].

    Args:
        [cite_start]D_o (float): Diâmetro externo da bobina (em metros)[cite: 159].
        [cite_start]w (float): Seção (diâmetro) do fio condutor (em metros)[cite: 160, 175].
        [cite_start]N (int): Número de espiras[cite: 171].
        [cite_start]s (float): Espaçamento entre as espiras (em metros)[cite: 160, 171].

    Returns:
        float: O diâmetro interno (Di) (em metros).
    """
    return D_o - (2 * w * N) - (2 * s * (N - 1))

def calcular_comprimento_fio(N: int, D_o: float, D_i: float) -> float:
    """
    [cite_start]Calcula o comprimento total do fio (l) na bobina conforme a Equação 4.13[cite: 267].

    Args:
        [cite_start]N (int): Número de espiras[cite: 171].
        [cite_start]D_o (float): Diâmetro externo da bobina (em metros)[cite: 159].
        [cite_start]D_i (float): Diâmetro interno da bobina (em metros)[cite: 159].

    Returns:
        float: O comprimento do fio (em metros).
    """
    return (math.pi * N * (D_o + D_i)) / 2

# --- Função Principal ---

def calcular_resistencia_litz(
    l: float,
    A_str: float,
    N_strands: int,
    n_b: int,
    rho: float = RHO_COBRE
) -> float:
    """
    Calcula a resistência máxima do fio Litz conforme a Equação 4.17.

    Fórmula: R_litz = ((rho * l) / (A_str * N_strands)) * k_nb * k_BW

    Args:
        l (float): Comprimento total do fio [m].
        A_str (float): Área de um único strand [m^2].
        N_strands (int): Número total de strands no fio Litz.
        n_b (int): Parâmetro 'nb' para o cálculo do k_nb.
        rho (float, optional): Resistividade do material condutor (Ohm.m).

    Returns:
        R_lits float: O valor da resistência do fio Litz [Ohn].
    """
    k_nb = calcular_k_nb(n_b)
    
    k_BW = K_BW

    resistencia_base = (rho * l) / (A_str * N_strands)

    R_litz = resistencia_base * k_nb * k_BW

    return R_litz

# --- Bloco de Exemplo ---
if __name__ == "__main__":

    print("=" * 60)
    print("Cálculo da Resistência Litz (Bobina Primária L1)")
    print("=" * 60)

    # Parâmetros do projeto para a Bobina Primária (L1)
    D_o1 = 0.5        # Diâmetro externo (m) [cite: 204]
    w1 = 4.4e-3       # Diâmetro do fio Litz (m) [cite: 152, 175]
    N1 = 12           # Número de espiras [cite: 258]
    s1 = 0            # Espaçamento entre espiras (m) [cite: 175, 262]
    D_str_mm = 0.1016 # Diâmetro do strand (mm) (38 AWG) [cite: 138]
    D_str_m = D_str_mm / 1000.0 # Convertido para metros [cite: 275]
    N_strands1 = 900  # Número de strands [cite: 152]

    # --- Passo 1: Calcular Diâmetro Interno (Eq 4.12) ---
    D_i1 = calcular_diametro_interno(D_o1, w1, N1, s1)
    print(f"Parâmetros L1: Do={D_o1} m, w={w1} m, N={N1}, s={s1} m")
    print(f"Diâmetro Interno (D_i1) calculado: {D_i1:.4f} m")
    print(f"    (Valor no documento: 0.3953 m [cite: 262])\n")

    # --- Passo 2: Calcular Comprimento do Fio (Eq 4.13) ---
    l1 = calcular_comprimento_fio(N1, D_o1, D_i1)
    print(f"Comprimento do fio (l1) calculado: {l1:.2f} m")
    print(f"    (Valor no documento: 16.87 m [cite: 268])\n")

    # --- Passo 3: Calcular Área do Strand (Eq 4.16) ---
    A_str1 = calcular_area_strand(D_str_m)
    print(f"Diâmetro do strand: {D_str_mm} mm (ou {D_str_m} m) [cite: 138, 275]")
    print(f"Área do strand (A_str) calculada: {A_str1:.4e} m^2")
    print(f"    (Valor no documento: 8.1073e-9 m^2 [cite: 275])\n")

    # --- Passo 4: Calcular Resistência Litz (Eq 4.17) ---
    # O valor de 'n_b' (Eq 4.18) não é fornecido nas páginas.
    # [cite_start]É um parâmetro da norma (DIN; EN; IEC, 2000)[cite: 277].
    # Vamos assumir um valor de exemplo (ex: n_b = 1).
    n_b_exemplo = 1

    print("-" * 60)
    print(f"Cálculo da Resistência R_litz (Equação 4.17) [cite: 280]")
    print(f"Usando parâmetros: l={l1:.2f} m, A_str={A_str1:.4e} m^2")
    print(f"                   N_strands={N_strands1}, rho={RHO_COBRE} Ohm.m")
    print(f"                   k_BW={K_BW} [cite: 281]")
    print(f"                   n_b={n_b_exemplo} (Valor de exemplo, não consta no doc)")
    print("-" * 60)

    R_litz1 = calcular_resistencia_litz(
        l=l1,
        A_str=A_str1,
        N_strands=N_strands1,
        n_b=n_b_exemplo,
        rho=RHO_COBRE
    )

    k_nb_calc = calcular_k_nb(n_b_exemplo)
    print(f"Fator k_nb calculado (Eq 4.18): {k_nb_calc:.2f}")
    print(f"Resistência Litz (R_litz1) calculada: {R_litz1:.6f} Ohms")
    print("=" * 60)
    
    
    # Verifiquei rapidamente e a unica coisa mais suspeita é knb, de resto ta certo
