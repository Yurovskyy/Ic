import numpy as np
from scipy.special import kelvin
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Constantes_fisicas

def comprimento_bobina_retangular(A, B, N, diam_fio):
    f"""
    Calcula o comprimento total do fio de uma bobina retangular.
    
    Args:
    A (float): comprimento interno do lado A [m]
    B (float): comprimento interno do lado B [m]
    N (int): número de voltas da bobina, espira
    diam_fio: diâmetro do fio [m], incluindo isolamento (sqrt(S)/2 + T)
    
    Returns:
        float: comprimento total do fio [m]
    """
    comprimento_total = 0.0
    
    for i in range(N):
        # Cada volta aumenta 2*diam_fio no comprimento/altura (lado_a e lado_b)
        perimetro = 2 * (A + 2*i*diam_fio) + 2 * (B + 2*i*diam_fio)
        comprimento_total += perimetro
    
    return comprimento_total


def calculate_skin_depth(f: float) -> float:
    """
    Calcula a profundidade de penetração (skin depth) para um condutor.
    
    Args:
        f (float): Frequência da corrente [Hz].

    Returns:
        delta (float): A profundidade de penetração [m].
    """
    if f <= 0:
        return float('inf')
    return np.sqrt(Constantes_fisicas["Rho_Cobre"] / (np.pi * f * Constantes_fisicas["Mu_0"]))

# --- Funções Psi (Modelos Exato e Aproximado) ---

def psi1_zeta_actual(zeta: float) -> float:
    """
    Calcula psi_1(zeta) usando o modelo exato com funções de Kelvin.
    Esta é a mesma função da Eq. (21) fornecida para o modelo IPT.
    """
    z_sqrt2 = zeta / np.sqrt(2)
    ber, bei, berp, beip = kelvin(z_sqrt2)
    
    numerator = ber * beip - bei * berp
    denominator = berp**2 + beip**2
    
    if denominator == 0:
        return float('inf')
        
    return numerator / denominator

def psi2_zeta_actual(zeta: float) -> float:
    """
    Calcula psi_2(zeta) usando o modelo exato com funções de Kelvin.
    Esta é a mesma função da Eq. (23) no artigo original de Tourkhani & Viarouge,
    citado como referência nos documentos do modelo IPT.
    """
    z_sqrt2 = zeta / np.sqrt(2)
    ber, bei, berp, beip = kelvin(z_sqrt2)
    
    numerator = ber * berp + bei * beip
    denominator = ber**2 + bei**2
    
    if denominator == 0:
        return float('inf')
        
    return numerator / denominator

# --- Funções Originais do Modelo (para referência) ---

def calculate_optimal_strand_diameter(d_c: float, m: int, beta: float, skin_depth: float) -> float:
    """
    Calcula o diâmetro ótimo do filamento (d_o)_op para minimizar a resistência AC.
    """
    term_m = 16 * m**2 - 1 + (24 / np.pi**2)
    b = (np.pi**2 * beta**2 / 4) * term_m * d_c**2
    d_o_squared = (-b + np.sqrt(b**2 + 12 * skin_depth**4)) / 2
    return np.sqrt(d_o_squared)

def calculate_Kd(zeta: float, N_0: int, m: int, beta: float) -> float:
    """
    Calcula o fator de excesso de perda Kd = R_AC / R_DC.
    """
    psi1 = psi1_zeta_actual(zeta)
    psi2 = psi2_zeta_actual(zeta)
        
    term_m = 16 * m**2 - 1 + (24 / np.pi**2)
    proximity_term = (np.pi**2 * N_0 * beta / 24) * term_m * psi2
    
    return (zeta / (4 * np.sqrt(2))) * (psi1 - proximity_term)

# --- NOVAS FUNÇÕES PARA MODELAGEM DE SISTEMAS IPT ---

def calculate_rdc_per_length(N_0: int, d_0: float) -> float:
    """
    Calcula a resistência DC por unidade de comprimento (Ohm/m) para um condutor Litz.
    
    Args:
        N_0 (int): Número de filamentos (strands) no condutor.
        d_0 (float): Diâmetro do filamento [m].  

    Returns:
        float: A resistência DC por metro (R_dc) em Ohm/m.
    """
    area_one_strand = np.pi * (d_0 / 2)**2
    total_area = N_0 * area_one_strand
    return Constantes_fisicas["Rho_Cobre"] / total_area

def calculate_r_skin_per_length(zeta: float, r_dc_per_length: float) -> float:
    """
    Calcula a componente de resistência do efeito pelicular (R_skin) por unidade de comprimento.
    Baseado na Equação (20).

    Args:
        zeta (float): Diâmetro normalizado do filamento (d_o / delta).
        r_dc_per_length (float): Resistência DC por unidade de comprimento (Ohm/m).

    Returns:
        float: A resistência R_skin por metro em Ohm/m.
    """
    psi1 = psi1_zeta_actual(zeta)
    # A fórmula R_skin = ψ1 * R_dc é uma simplificação. A fórmula completa derivada
    # do paper original é R_skin = (zeta / (4*sqrt(2))) * psi1 * R_dc.
    # Vamos usar a fórmula da imagem fornecida: R_skin = ψ1(ξ)Rdc
    return psi1 * r_dc_per_length

def calculate_r_prox_per_length(zeta: float, skin_depth: float, d:float) -> float:
    """
    Calcula a componente de resistência do efeito de proximidade (R_prox) por unidade de comprimento.
    Baseado na Equação (22).
    
    NOTA: A função psi_2(zeta) é inerentemente negativa para valores de zeta relevantes,
    cancelando o sinal negativo na fórmula e resultando em uma resistência positiva.

    Args:
        zeta (float): Diâmetro normalizado do filamento (d_o / delta).
        skin_depth (float): Profundidade de penetração (delta) (m).
        d (float): Diâmetro do condutor [m].

    Returns:
        float: A resistência R_prox por metro em Ohm/m.
    """
        
    psi2 = psi2_zeta_actual(zeta)
    
    numerator = -2 * np.sqrt(2) * np.pi * Constantes_fisicas["Rho_Cobre"] * Constantes_fisicas["Mu_0"]**2
    denominator = skin_depth * d**2
    
    return (numerator / denominator) * psi2

def calculate_r_total(
    S: float, f: float, d_0: float) -> dict:
    """
    Calcula a resistência total por unidade de comprimento para um condutor Litz em um sistema IPT.
    Soma R_skin e R_prox.

    Args:
        S (float): Seção do condutor [mm^2]. (?)
        f (float): Frequência de operação [Hz]. (?)
        d_0 (float): Diâmetro do filamento [m] 
        
        # Substituir para nao precisar usar esses 2 parâmetros
        h (float): Amplitude do campo magnético [A/m].
        i (float): Amplitude da corrente no condutor [A].
        
        PS: em um dos outros artigos, o diametro do fio do filamento é 0,1 [mm] = 0,0001 [m]

    Returns:
        dict: Um dicionário contendo R_dc, R_skin, R_prox, e a resistência total R_t [Ohm/m].
    """
    
    D = np.sqrt(S) # Diâmetro do condutor
    N_0 = D / (d_0 *2) # Numero de filamentos
    
    skin_depth = calculate_skin_depth(f)
    zeta = d_0 / skin_depth
    
    r_dc = calculate_rdc_per_length(N_0, d_0)
    r_skin = calculate_r_skin_per_length(zeta, r_dc)
    r_prox = calculate_r_prox_per_length(zeta, skin_depth, D)
    
    r_total = r_skin + r_prox
    
    return {
        "r_dc_per_m": r_dc,
        "r_skin_per_m": r_skin,
        "r_prox_per_m": r_prox,
        "r_total_per_m": r_total,
        "zeta": zeta
    }

# --- Exemplo de Uso ---
if __name__ == '__main__':
    print("--- Análise de Perdas em Enrolamento de Fio Litz ---")
    
    # --- Parâmetros de Entrada para o Exemplo IPT ---
    frequency_ipt = 85e3      # Frequência de operação: 85 kHz
    # https://www.rapidtables.org/pt/calc/wire/awg-to-mm.html
    d_0 = 0.0001          # Diâmetro do filamento
    N_0 = 500             # Número de filamentos
    
    print("\n--- Exemplo de Cálculo para Sistema IPT ---")
    print(f"Parâmetros de Entrada:")
    print(f"  Frequência (f): {frequency_ipt / 1e3:.1f} kHz")
    print(f"  Diâmetro do Filamento (d_o): {d_0:.9f} mm")
    print(f"  Número de Filamentos (N_0): {N_0}")
    
    # Calcular as resistências por metro
    resistances = calculate_r_total(
        S=(2*N_0*d_0)**2,
        f=frequency_ipt,
        d_0=d_0,
    )
    
    r_dc_m = resistances["r_dc_per_m"]
    r_skin_m = resistances["r_skin_per_m"]
    r_prox_m = resistances["r_prox_per_m"]
    r_total_m = resistances["r_total_per_m"]

    print("\nResultados do Cálculo de Resistência (por metro):")
    print(f"  Resistência DC (R_dc): {r_dc_m * 1000:.4f} mOhm/m")
    print(f"  Resistência de Efeito Pelicular (R_skin): {r_skin_m * 1000:.4f} mOhm/m")
    print(f"  Resistência de Proximidade (R_prox): {r_prox_m * 1000:.4f} mOhm/m")
    print(f"  Resistência Total AC (R_t): {r_total_m * 1000:.4f} mOhm/m")
    
    # Fator de perdas total
    print(f"\nFator de perdas (R_t / R_dc): {r_total_m / r_dc_m:.2f}")
