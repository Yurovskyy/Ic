"""
Módulo para cálculos do circuito do sistema WPT.
Referência: Seções 2 e 4 (Etapa 2.2) do artigo.
"""
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Parametros_fixos_projeto, Restricoes, Parametros_numericos
from modelagem_spain import calculate_resistances

def calculate_system_parameters(individual, frequency_hz):
    """
    Calcula os parâmetros elétricos do sistema para uma dada frequência.
    Baseado nas equações da Tabela 1 e Seção 2.
    """
    # Frequência angular
    w = 2 * math.pi * frequency_hz
    
    # Número imaginário
    j = 1j
    
    # Calcula resistências para a frequência atual
    R_p, R_s = calculate_resistances(individual, frequency_hz,Parametros_fixos_projeto["d_0"])
    # (Eq. 1) Calcula a resistência de carga equivalente RL
    R_L = (8 / math.pi**2) * (individual.variables['V_s']**2 / Parametros_fixos_projeto["P_d"])
    
    # (Eqs. 5, 6) Capacitores de ressonância
    C_p = 1 / (individual.L_p * w**2)
    C_s = 1 / (individual.L_s * w**2)
    
    # (Eq. 4 ) Impedância total refletida
    # Não podemos assumir que já temos a impedância em ressonância
    # O método da secante é pra encontrar o ponto de ressonância!
    
    # Impedância total do laço secundário (carga + bobina secundária)
    Z_s_loop = (R_s + R_L) + j * (w * individual.L_s - 1 / (w * C_s))
    
    # Impedância refletida para o primário (será um número complexo)
    if abs(Z_s_loop) < 1e-9: # Evita divisão por zero
        Z_reflected = complex('inf')
    else:
        Z_reflected = (w * individual.M)**2 / Z_s_loop
    
    # Impedância total do laço primário (fonte + bobina primária + impedância refletida)
    Z_total = R_p + j * (w * individual.L_p - 1 / (w * C_p)) + Z_reflected
    
    # Correntes (Ip, Is)
    I_p = individual.variables['V_p'] / Z_total
    I_s = (j*w * individual.M * I_p) / Z_s_loop
    
    # Para cálculos de potência e eficiência, usamos a MAGNITUDE (abs())
    I_p_magnitude = abs(I_p)
    I_s_magnitude = abs(I_s)
    
    # Potência de saída (Eq. 8)
    power_out = R_L * (I_s_magnitude**2)
    # Potência de entrada
    power_in = power_out + (I_p_magnitude**2 * R_p) + (I_s_magnitude**2 * R_s)
    # Eficiência
    efficiency = power_out / power_in
    
    # (Eqs. 9, 10) Tensão nos capacitores
    VC_p = I_p / (j*w * C_p)
    VC_s = I_s / (j*w * C_s)
    
    return {
        'power_out': power_out, 'efficiency': efficiency, 'R_p': R_p, 'R_s': R_s,
        'I_p': I_p, 'I_s': I_s, 'VC_p': VC_p, 'VC_s': VC_s
    }

def secant_method_for_frequency(individual):
    """
    Implementa o método da secante para encontrar a frequência de operação.
    O objetivo é encontrar a frequência 'f' tal que a potência de saída seja 11 kW.
    """
    f_min, f_max = Restricoes['frequency_kHz']
    f_k_minus_1 = f_min * 1000
    f_k = f_max * 1000
    
    params_k_minus_1 = calculate_system_parameters(individual, f_k_minus_1)
    delta_P_k_minus_1 = params_k_minus_1['power_out'] - Parametros_fixos_projeto["P_d"]

    for _ in range(Parametros_numericos["Maximo_iterador_secante"]):
        params_k = calculate_system_parameters(individual, f_k)
        delta_P_k = params_k['power_out'] - Parametros_fixos_projeto["P_d"]

        if abs(delta_P_k) < Parametros_numericos["Tolerancia_secante"]:
            # Convergiu
            return f_k, params_k

        # (Eq. 47) - Fórmula da secante para a próxima frequência
        denominator = delta_P_k - delta_P_k_minus_1
        if abs(denominator) < 1e-9: # Evita divisão por zero
            return None, None 

        f_k_plus_1 = f_k - delta_P_k * (f_k - f_k_minus_1) / denominator
        
        # Atualiza as variáveis para a próxima iteração
        f_k_minus_1, f_k = f_k, f_k_plus_1
        delta_P_k_minus_1 = delta_P_k
        
        # Garante que a frequência permaneça nos limites
        if not (Restricoes['frequency_kHz'][0] * 1000 <= f_k <= Restricoes['frequency_kHz'][1] * 1000):
            return None, None

    return None, None # Não convergiu
