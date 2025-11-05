import numpy as np
import math
import os, sys

# --- Constantes e Parâmetros (Definidos para o exemplo) ---
class Constantes_fisicas:
    Mu_0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo (H/m)

class Parametros_numericos:
    Eps_distance = 1e-9 # Pequena distância para evitar divisão por zero

# --- Funções Auxiliares (Modificadas/Novas) ---

def create_rectangular_loop_points(width_a: float, height_b: float, center_z: float, segments_per_side: int) -> np.ndarray:
    """
    Cria os pontos que definem uma única espira retangular no plano xy.
    """
    points = []
    half_a = width_a / 2.0
    half_b = height_b / 2.0
    
    # Coordenadas dos 4 cantos
    corners = [
        np.array([-half_a, -half_b, center_z]), # Canto inferior esquerdo
        np.array([ half_a, -half_b, center_z]), # Canto inferior direito
        np.array([ half_a,  half_b, center_z]), # Canto superior direito
        np.array([-half_a,  half_b, center_z])  # Canto superior esquerdo
    ]

    # Interpola pontos ao longo de cada lado
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        for j in range(segments_per_side):
            point = p1 + (p2 - p1) * (j / segments_per_side)
            points.append(point)
            
    return np.array(points)

def create_coil_geometry(width_a: float, height_b: float, num_turns: int, turn_spacing: float, segments_per_side: int, z_offset: float = 0.0) -> list:
    """
    Cria a geometria completa de uma bobina retangular com múltiplas espiras.
    Cada espira é uma lista de pontos. A função retorna uma lista de espiras.
    """
    coil = []
    # Assume que o enrolamento cresce para dentro a partir das dimensões externas
    for n in range(num_turns):
        # A cada volta, as dimensões diminuem
        current_a = width_a - 2 * n * turn_spacing
        current_b = height_b - 2 * n * turn_spacing
        
        # A posição z é a mesma para todas as espiras de uma bobina
        loop_points = create_rectangular_loop_points(current_a, current_b, z_offset, segments_per_side)
        coil.append(loop_points)
        
    return coil

def get_segment_vectors_and_midpoints(loop_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretiza uma espira (circular ou retangular) em vetores de segmento (dl) 
    e seus pontos médios.
    """
    num_points = len(loop_points)
    segment_vectors, segment_midpoints = [], []
    for i in range(num_points):
        p1 = loop_points[i]
        p2 = loop_points[(i + 1) % num_points]
        dl = p2 - p1
        midpoint = (p1 + p2) / 2.0
        segment_vectors.append(dl)
        segment_midpoints.append(midpoint)
    return np.array(segment_vectors), np.array(segment_midpoints)


def calculate_M_neumann_equation18(coil1_geometry: list, coil2_geometry: list) -> float:
    """
    Calcula a indutância mútua total entre duas bobinas retangulares
    implementando a Equação 18 do artigo.
    """
    total_mutual_inductance = 0.0
    num_loops_1 = len(coil1_geometry)
    num_loops_2 = len(coil2_geometry)

    print(f"Calculando M mútua para {num_loops_1}x{num_loops_2} pares de espiras...")

    # Laço duplo da Equação 18: Soma sobre i=1..N1 e j=1..N2
    for i in range(num_loops_1):
        # Discretiza a espira 'i' da bobina 1
        segments1, midpoints1 = get_segment_vectors_and_midpoints(coil1_geometry[i])
        
        for j in range(num_loops_2):
            # Discretiza a espira 'j' da bobina 2
            segments2, midpoints2 = get_segment_vectors_and_midpoints(coil2_geometry[j])
            
            # Calcula M_ij para este par de espiras usando a soma de Neumann
            M_ij = 0.0
            for k in range(len(segments1)):
                dl1, mid1 = segments1[k], midpoints1[k]
                for l in range(len(segments2)):
                    dl2, mid2 = segments2[l], midpoints2[l]
                    
                    r_vec = mid1 - mid2
                    r = np.linalg.norm(r_vec)
                    
                    if r > Parametros_numericos.Eps_distance:
                        M_ij += np.dot(dl1, dl2) / r
            
            # Adiciona a contribuição M_ij à indutância total
            total_mutual_inductance += M_ij

        # Feedback de progresso
        print(f"  Progresso: Bobina 1, espira {i+1}/{num_loops_1} concluída.")

    return (Constantes_fisicas.Mu_0 / (4 * np.pi)) * total_mutual_inductance

# ==============================================================================
# 3. VALIDAÇÃO COM OS DADOS DO ARTIGO E EQUAÇÃO 18
# ==============================================================================

if __name__ == '__main__':
    # --- Parâmetros extraídos da Tabela 4 do artigo ---
    # Bobina 1 (Primária, maior, z=0)
    Ap = 0.65      # [m]
    Bp = 0.50      # [m]
    Np = 15        # Número de espiras
    T_dp = 1.5e-3  # Espaçamento entre espiras [m]
    n_t = 280
    d_t = 0.2e-3

    packing_factor = 0.5
    area_cobre_total = n_t * math.pi * (d_t / 2)**2
    area_total_cabo = area_cobre_total / packing_factor
    s = math.sqrt(area_total_cabo)

    Ap = Ap - s*7
    Bp = Bp - s*7

    # Bobina 2 (Secundária, menor, em z=distancia)
    As = 0.38      # [m]
    Bs = 0.38      # [m]
    Ns = 16        # Número de espiras
    T_ds = 1.0e-3  # Espaçamento entre espiras [m]

    As = As - s* (Ns/2 - 1)
    Bs = Bs - s* (Ns/2 - 1)
    
    # Distância axial entre as bobinas
    distancia = 0.25 # [m]

    # Parâmetro de precisão para a discretização numérica
    segmentos_por_lado = 20

    print("--- Implementação da Equação 18 (Neumann para Múltiplas Espiras) ---")

    # 1. Gerar a geometria 3D completa de cada bobina
    print("Gerando geometria das bobinas...")
    coil_p_geom = create_coil_geometry(Ap, Bp, Np, T_dp, segmentos_por_lado, z_offset=0.0)
    coil_s_geom = create_coil_geometry(As, Bs, Ns, T_ds, segmentos_por_lado, z_offset=distancia)
    
    # 2. Calcular a indutância mútua total
    M_total_eq18 = calculate_M_neumann_equation18(coil_p_geom, coil_s_geom)
    
    # Valor de referência do artigo (experimental)
    M_paper_experimental = 19.5e-6 # Convertido de µH para H
    M_paper_proposto = 19.67e-6 # Valor do método proposto no artigo

    print("\n--- Resultados Finais ---")
    print(f"Mútua total (Eq. 18, Numérico):       {M_total_eq18 * 1e6:.2f} µH")
    print(f"Mútua do Artigo (Método Proposto):    {M_paper_proposto * 1e6:.2f} µH")
    print(f"Mútua do Artigo (Experimental):       {M_paper_experimental * 1e6:.2f} µH")
    print("-" * 50)

    # Calcula e exibe o erro percentual
    erro_percentual = abs((M_total_eq18 - M_paper_experimental) / M_paper_experimental) * 100
    print(f"Erro percentual vs. Experimental: {erro_percentual:.2f}%")
