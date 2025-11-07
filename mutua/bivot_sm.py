import numpy as np
import scipy.integrate as integrate
import scipy.constants as const
import sys
import time

def calcular_indutancia_mutua(a, rho_2_max, D, N1, N2):
    """
    Calcula a indutância mútua (M') entre duas bobinas usando a
    integração tripla de (4.27), implementada com scipy.integrate.tplquad.

    A função segue a lógica do documento:
    1. Calcula a integral tripa I.
    2. Calcula M (para uma espira)[cite: 29].
    3. Calcula M' (total) multiplicando por N1 e N2[cite: 31].

    Argumentos:
        a (float): Raio da bobina primária (rho_1) em metros[cite: 6, 19].
        rho_2_max (float): Raio máximo da bobina secundária (rho_2) em metros[cite: 8, 20].
        D (float): Distância entre as bobinas em metros[cite: 11, 23].
        N1 (int): Número de espiras da bobina primária.
        N2 (int): Número de espiras da bobina secundária.

    Retorna:
        tuple: (M_total_uH, M_single_uH)
            M_total_uH (float): Indutância mútua total (M') em micro-Henries (uH).
            M_single_uH (float): Indutância mútua para N1=1, N2=1 (M) em micro-Henries (uH).
    """
    
    # Constante: Permeabilidade do vácuo (mu_0)
    mu_0 = const.mu_0 

    # --- Definição do Integrando (Forma Simplificada) ---
    # O tplquad espera os argumentos na ordem: (interna, meio, externa)
    # Ordem da integral: d(phi_1), d(phi_2), d(rho_2) 
    def integrand_func(phi_1, phi_2, rho_2_var):
        
        # Denominador (r^3)
        # r = sqrt(rho_2^2 + a^2 - 2*rho_2*a*cos(phi_2 - phi_1) + D^2) [cite: 11]
        cos_diff = np.cos(phi_2 - phi_1)
        r_squared = rho_2_var**2 + a**2 - 2 * rho_2_var * a * cos_diff + D**2
        r_cubed = r_squared**1.5 # (r^3)

        # Se r é quase zero (o que não deve acontecer se D > 0), retorne 0
        if r_cubed < 1e-15:
            return 0.0

        # Numerador (Simplificado)
        # rho_2 * (rho_2 * cos(phi_2 - phi_1) - a)
        numerator = rho_2_var * (rho_2_var * cos_diff - a)
        
        return numerator / r_cubed

    # --- Limites da Integração ---
    phi_1_min, phi_1_max = 0, 2 * np.pi  # [cite: 24]
    phi_2_min, phi_2_max = 0, 2 * np.pi  # [cite: 24]
    rho_2_min, rho_2_max_limit = 0, rho_2_max  # [cite: 25]

    # --- Cálculo da Integral Tripla com SciPy ---
    print("\nIniciando cálculo da integral tripla com scipy.integrate.tplquad...")
    print("(Isso pode levar alguns minutos)...")
    start_time = time.time()
    
    # tplquad(func, g, h, q, r, f)
    # g, h = limites de rho_2_var (externa)
    # q, r = limites de phi_2 (meio)
    # f = limites de phi_1 (interna)
    try:
        integral_value, abs_error = integrate.tplquad(
            integrand_func,
            rho_2_min, rho_2_max_limit,  # Limites de rho_2
            phi_2_min, phi_2_max,        # Limites de phi_2
            phi_1_min, phi_1_max         # Limites de phi_1
        )
    except Exception as e:
        print(f"Erro durante a integração: {e}")
        return None, None
    
    end_time = time.time()
    print(f"Cálculo da integral concluído em {end_time - start_time:.2f} segundos.")
    print(f"Valor da integral (I): {integral_value:.5e}")

    # --- Cálculo de M (Uma Espira) ---
    # O documento calcula M (Eq. 4.32) como o resultado de (4.27)
    # para uma espira[cite: 27, 29].
    # Portanto, removemos N1 e N2 da constante de (4.27).
    # M = - (rho_1 * mu_0) / (4 * pi) * integral_value
    # (rho_1 é 'a' [cite: 6, 19])
    
    constant_factor = - (a * mu_0) / (4 * np.pi)
    M_single_loop_H = constant_factor * integral_value
    
    # --- Cálculo de M' (Total) ---
    # M' = M * N1 * N2 [cite: 31]
    M_total_H = M_single_loop_H * N1 * N2
    
    # Converter para micro-Henries (uH) para comparação
    M_single_loop_uH = M_single_loop_H * 1e6
    M_total_uH = M_total_H * 1e6
    
    return M_total_uH, M_single_loop_uH

if __name__ == "__main__":
  
    """
    Função principal para replicar o exemplo do documento
    com N1=12 e N2=13.
    """
    
    # Valores do exemplo extraídos do documento
    a = 0.2238          # Raio primário (rho_1) [cite: 19]
    rho_2_max = 0.2216  # Raio secundário (rho_2) [cite: 20]
    D = 0.16            # Distância [cite: 23]
    
    # Números de espiras (N1, N2) conforme solicitado
    N1 = 12
    N2 = 13
    
    print("--- Cálculo da Indutância Mútua (Exemplo) ---")
    print(f"Parâmetros de entrada:")
    print(f"  Raio primário (a): {a} m [cite: 19]")
    print(f"  Raio secundário (rho_2): {rho_2_max} m [cite: 20]")
    print(f"  Distância (D): {D} m [cite: 23]")
    print(f"  N1 (espiras primárias): {N1}")
    print(f"  N2 (espiras secundárias): {N2}")

    # Chamar a função de cálculo
    M_total_uH, M_single_uH = calcular_indutancia_mutua(a, rho_2_max, D, N1, N2)

    if M_total_uH is None:
        print("\nO cálculo falhou.")

    print("\n--- Resultados Finais ---")
    
    print(f"\nIndutância mútua para uma espira (M):")
    print(f"  Calculado: {M_single_uH:.5f} uH")
    print(f"  Valor do documento (M): 0.17 uH [cite: 27]")

    print(f"\nIndutância mútua total (M'):")
    print(f"  Calculado: {M_total_uH:.5f} uH")
    print(f"  Valor do documento (M'): 26.75 uH [cite: 31]")

    # Verificação de consistência: O valor no documento (26.75) é obtido
    # multiplicando-se o *valor real* de M (não o arredondado 0.17)
    # por N1*N2.
    # (Valor real de M = 26.75 / (12*13) = 0.17147...)
    # Nosso valor calculado (M_single_uH) deve estar próximo de 0.17147.
    print(f"\nVerificação (M_calculado * N1 * N2):")
    print(f"  {M_single_uH:.5f} uH * {N1} * {N2} = {M_single_uH * N1 * N2:.5f} uH")
