import numpy as np
from scipy.special import kelvin
try:
    from ..config import Constantes_fisicas
except ImportError:
    # Suporte a execução direta do arquivo (sem pacote pai)
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import Constantes_fisicas


def calculate_skin_depth(resistivity: float, frequency: float, mu: float = Constantes_fisicas["Mu_0"]) -> float:
    """
    Calcula a profundidade de penetração (skin depth) para um condutor.
    
    Args:
        resistivity (float): Resistividade do material do condutor (Ohm*m).
        frequency (float): Frequência da corrente (Hz).
        mu (float): Permeabilidade magnética do material (H/m).

    Returns:
        float: A profundidade de penetração (delta) em metros.
    """
    if frequency <= 0:
        return float('inf')
    return np.sqrt(resistivity / (np.pi * frequency * mu))

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

def calculate_Kd(zeta: float, N_0: int, m: int, beta: float, use_approx: bool = False) -> float:
    """
    Calcula o fator de excesso de perda Kd = R_AC / R_DC.
    """
    psi1 = psi1_zeta_actual(zeta)
    psi2 = psi2_zeta_actual(zeta)
        
    term_m = 16 * m**2 - 1 + (24 / np.pi**2)
    proximity_term = (np.pi**2 * N_0 * beta / 24) * term_m * psi2
    
    return (zeta / (4 * np.sqrt(2))) * (psi1 - proximity_term)

# --- NOVAS FUNÇÕES PARA MODELAGEM DE SISTEMAS IPT ---

def calculate_rdc_per_length(N_0: int, d_o: float, resistivity: float = Constantes_fisicas["Rho_Cobre"]) -> float:
    """
    Calcula a resistência DC por unidade de comprimento (Ohm/m) para um condutor Litz.
    
    Args:
        N_0 (int): Número de filamentos (strands) no condutor.
        d_o (float): Diâmetro de um único filamento (m).
        resistivity (float): Resistividade do material do condutor (Ohm*m).

    Returns:
        float: A resistência DC por metro (R_dc) em Ohm/m.
    """
    area_one_strand = np.pi * (d_o / 2)**2
    total_area = N_0 * area_one_strand
    return resistivity / total_area

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

def calculate_r_prox_per_length(zeta: float, skin_depth: float, H: float, I: float, resistivity: float = Constantes_fisicas["Rho_Cobre"]) -> float:
    """
    Calcula a componente de resistência do efeito de proximidade (R_prox) por unidade de comprimento.
    Baseado na Equação (22).
    
    NOTA: A função psi_2(zeta) é inerentemente negativa para valores de zeta relevantes,
    cancelando o sinal negativo na fórmula e resultando em uma resistência positiva.

    Args:
        zeta (float): Diâmetro normalizado do filamento (d_o / delta).
        skin_depth (float): Profundidade de penetração (delta) (m).
        H (float): Amplitude do campo magnético externo (A/m).
        I (float): Amplitude da corrente total no condutor Litz (A).
        resistivity (float): Resistividade do material (Ohm*m).

    Returns:
        float: A resistência R_prox por metro em Ohm/m.
    """
    if I == 0:
        return 0.0
        
    psi2 = psi2_zeta_actual(zeta)
    
    numerator = -2 * np.sqrt(2) * np.pi * resistivity * H**2
    denominator = skin_depth * I**2
    
    return (numerator / denominator) * psi2

def calculate_r_total(
    n: int, d_0: float, f: float, h: float, i: float, rho: float = Constantes_fisicas["Rho_Cobre"]
) -> dict:
    """
    Calcula a resistência total por unidade de comprimento para um condutor Litz em um sistema IPT.
    Soma R_skin e R_prox.

    Args:
        n (int): Número de filamentos no condutor.
        d_o (float): Diâmetro de um filamento [m].
        f (float): Frequência de operação [Hz].
        h (float): Amplitude do campo magnético [A/m].
        i (float): Amplitude da corrente no condutor [A].
        rho (float): Resistividade do material [Ohm*m].

    Returns:
        dict: Um dicionário contendo R_dc, R_skin, R_prox, e a resistência total R_t [Ohm/m].
    """
    skin_depth = calculate_skin_depth(rho, f)
    zeta = d_0 / skin_depth
    
    r_dc = calculate_rdc_per_length(n, d_0, rho)
    r_skin = calculate_r_skin_per_length(zeta, r_dc)
    r_prox = calculate_r_prox_per_length(zeta, skin_depth, H, I, rho)
    
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
    # Valores típicos para um sistema de carregamento sem fio de veículos (SAE J2954 WPT1)
    frequency_ipt = 85e3      # Frequência de operação: 85 kHz
    d_o_ipt = 100e-6          # Diâmetro do filamento: 100 um (aprox. 38 AWG)
    N_0_ipt = 500             # Número de filamentos
    I_ipt = 30                # Corrente RMS na bobina: 30 A
    H_ipt = 600               # Campo magnético estimado na região do enrolamento: 600 A/m
    
    print("\n--- Exemplo de Cálculo para Sistema IPT ---")
    print(f"Parâmetros de Entrada:")
    print(f"  Frequência (f): {frequency_ipt / 1e3:.1f} kHz")
    print(f"  Diâmetro do Filamento (d_o): {d_o_ipt * 1e6:.0f} um")
    print(f"  Número de Filamentos (N_0): {N_0_ipt}")
    print(f"  Corrente (I): {I_ipt} A")
    print(f"  Campo Magnético (H): {H_ipt} A/m")

    # Calcular as resistências por metro
    resistances = calculate_r_total(
        n=N_0_ipt,
        d_0=d_o_ipt,
        f=frequency_ipt,
        H=H_ipt,
        I=I_ipt
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
