# electromagnetic_modeling.py
"""
Módulo para modelagem eletromagnética do sistema WPT.
Referência: Seção 3 do artigo.

NOTA: As funções abaixo são placeholders simplificados. A implementação real
exigiria a resolução das complexas Equações 14-23 do artigo.
"""
import math

def calculate_inductances(individual):
    """
    Calcula as auto-indutâncias (Lp, Ls) e a indutância mútua (M).
    Referência: Seção 3.1 (Eq. 16) e 3.2 (Eq. 18).
    
    Esta é uma aproximação simplificada.
    """
    # A indutância é proporcional ao quadrado do número de espiras
    L_p = individual.variables['N_p']**2 * 1e-6 # Valor em Henry
    L_s = individual.variables['N_s']**2 * 1e-6 # Valor em Henry
    
    # A mútua depende do número de espiras de ambos e de um fator de acoplamento 'k'
    k = 0.2 # Fator de acoplamento típico
    M = k * math.sqrt(L_p * L_s)
    
    return L_p, L_s, M

def calculate_resistances(individual, frequency_hz):
    """
    Calcula as resistências AC (Rp, Rs) considerando efeitos de pele e proximidade.
    Referência: Seção 3.3 (Eq. 19-23).
    
    Esta é uma aproximação que simula o aumento da resistência com a frequência.
    """
    # Resistência DC é inversamente proporcional à área da seção
    R_p_dc = 1 / individual.variables['S_p'] * 0.1
    R_s_dc = 1 / individual.variables['S_s'] * 0.1
    
    # Efeito de pele aumenta com a raiz da frequência
    skin_effect_factor = 1 + 0.1 * math.sqrt(frequency_hz / 1000)
    
    R_p = R_p_dc * skin_effect_factor
    R_s = R_s_dc * skin_effect_factor
    
    return R_p, R_s
