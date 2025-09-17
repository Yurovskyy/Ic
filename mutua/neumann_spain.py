import numpy as np
from scipy.special import ellipk, ellipe # Para as integrais elípticas
import math
from ..config import mu_0

# ==============================================================================
# CONSTANTES DE MÓDULO
# ==============================================================================
mu_0 = mu_0  # Permeabilidade do vácuo (H/m)
EPS_DISTANCE = 1e-9       # Tolerância para evitar divisão por zero em distâncias

# ==============================================================================
# 1. IMPLEMENTAÇÃO DE REFERÊNCIA (FÓRMULA ANALÍTICA EXATA)
# ==============================================================================

def calculate_M_analytical_circular(radius1: float, radius2: float, distance: float) -> float:
    """
    Calcula a indutância mútua entre duas espiras circulares coaxiais
    usando a fórmula analítica exata com integrais elípticas.

    Args:
        radius1 (float): Raio da primeira espira (a), em metros.
        radius2 (float): Raio da segunda espira (b), em metros.
        distance (float): Distância axial entre as espiras (d), em metros.

    Returns:
        float: A indutância mútua (M) em Henries.
    """
    if distance == 0 and radius1 == radius2:
        # Caso especial: autoindutância de uma espira fina (tende ao infinito), não Mútua.
        # Mas para M, se d=0, o campo é no plano.
        # A fórmula diverge se os fios se tocam. Retorna um valor simbólico.
        return float('inf')
        
    # Parâmetro k^2 da fórmula
    k_squared = (4 * radius1 * radius2) / ((radius1 + radius2)**2 + distance**2)
    
    # k é a raiz quadrada do módulo para as funções elípticas
    k = np.sqrt(k_squared)

    # K(k) e E(k) são as integrais elípticas completas
    K_k = ellipk(k_squared)
    E_k = ellipe(k_squared)
    
    # Aplica a fórmula completa
    term1 = (2 / k - k) * K_k
    term2 = (2 / k) * E_k
    
    M = mu_0 * np.sqrt(radius1 * radius2) * (term1 - term2)
    
    return M

# ==============================================================================
# 2. IMPLEMENTAÇÃO NUMÉRICA (MÉTODO USADO NO SEU CÓDIGO, ADAPTADO)
# ==============================================================================

# Funções auxiliares (as mesmas da primeira resposta, mas para espiras circulares)
def create_circular_loop_points(radius: float, num_segments: int, center_z: float = 0.0) -> np.ndarray:
    points = []
    delta_angle = 2 * np.pi / num_segments
    for i in range(num_segments):
        angle = i * delta_angle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append(np.array([x, y, center_z]))
    return np.array(points)

def get_segment_vectors_and_midpoints(loop_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def calculate_M_numerical_circular(radius1: float, radius2: float, distance: float, num_segments: int) -> float:
    """
    Calcula a indutância mútua entre duas espiras circulares coaxiais
    usando a aproximação numérica da fórmula de Neumann.
    """
    # Cria as geometrias das duas espiras
    points1 = create_circular_loop_points(radius1, num_segments, center_z=0.0)
    points2 = create_circular_loop_points(radius2, num_segments, center_z=distance)

    # Discretiza as espiras em segmentos dl e pontos médios
    segments1, midpoints1 = get_segment_vectors_and_midpoints(points1)
    segments2, midpoints2 = get_segment_vectors_and_midpoints(points2)
    
    # Aplica a soma de Neumann
    M_numerical = 0.0
    for i in range(num_segments):
        dl1, mid1 = segments1[i], midpoints1[i]
        for j in range(num_segments):
            dl2, mid2 = segments2[j], midpoints2[j]
            r_vec = mid1 - mid2
            r = np.linalg.norm(r_vec)
            if r > EPS_DISTANCE:
                M_numerical += np.dot(dl1, dl2) / r
    
    return (mu_0 / (4 * np.pi)) * M_numerical

# ==============================================================================
# 3. COMPARAÇÃO E VALIDAÇÃO
# ==============================================================================

if __name__ == '__main__':
    # --- Parâmetros do Teste ---
    r1 = 0.1   # Raio da espira 1 (10 cm)
    r2 = 0.15  # Raio da espira 2 (15 cm)
    d = 0.05   # Distância entre as espiras (5 cm)
    
    # Parâmetro de precisão para o método numérico
    # Quanto maior, mais preciso (e mais lento) será o cálculo.
    n_segments = 400

    print("--- Validação da Implementação da Indutância Mútua ---")
    print(f"Parâmetros: Raio 1 = {r1} m, Raio 2 = {r2} m, Distância = {d} m")
    print("-" * 55)

    # Calcula usando a fórmula analítica (gabarito)
    M_analytical = calculate_M_analytical_circular(r1, r2, d)
    print(f"Resultado (Fórmula Analítica Exata): M = {M_analytical:.6e} H")

    # Calcula usando a implementação numérica
    M_numerical = calculate_M_numerical_circular(r1, r2, d, n_segments)
    print(f"Resultado (Implementação Numérica):   M = {M_numerical:.6e} H (com {n_segments} segmentos)")
    
    print("-" * 55)

    # Calcula e exibe o erro percentual
    error_percent = abs((M_numerical - M_analytical) / M_analytical) * 100
    print(f"Erro Percentual: {error_percent:.4f}%")
    
    if error_percent < 0.1:
        print("\nVALIDAÇÃO BEM-SUCEDIDA: O resultado numérico é extremamente próximo do analítico.")
    else:
        print("\nAVISO: O erro é maior que o esperado. Verifique os parâmetros.")
