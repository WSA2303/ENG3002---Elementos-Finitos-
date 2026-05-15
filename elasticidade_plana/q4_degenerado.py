import numpy as np


# ============================================================
# VERIFICAÇÃO: Q4 DEGENERADO COM DOIS NÓS COINCIDENTES VIRA CST?
# ============================================================
# Ideia:
#   1) Define-se um triângulo CST com nós 1, 2 e 3.
#   2) Define-se um Q4 degenerado com nós 3 e 4 coincidentes.
#   3) Calcula-se a matriz de rigidez do Q4.
#   4) Impõe-se que os nós coincidentes tenham os mesmos deslocamentos:
#          u4 = u3, v4 = v3
#      por meio de uma matriz de transformação T.
#   5) Compara-se K_Q4_degenerado com K_CST.
# ============================================================


def D_plane_stress(E, nu):
    return E / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=float)


def q4_shape_functions(xi, eta):
    N = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ], dtype=float)

    dN_dxi = 0.25 * np.array([
        -(1.0 - eta),
         (1.0 - eta),
         (1.0 + eta),
        -(1.0 + eta),
    ], dtype=float)

    dN_deta = 0.25 * np.array([
        -(1.0 - xi),
        -(1.0 + xi),
         (1.0 + xi),
         (1.0 - xi),
    ], dtype=float)

    return N, dN_dxi, dN_deta


def q4_B_matrix(xy, xi, eta):
    _, dN_dxi, dN_deta = q4_shape_functions(xi, eta)

    J = np.array([
        [np.dot(dN_dxi, xy[:, 0]),  np.dot(dN_deta, xy[:, 0])],
        [np.dot(dN_dxi, xy[:, 1]),  np.dot(dN_deta, xy[:, 1])],
    ], dtype=float)

    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError("detJ <= 0. Verifique a ordem dos nós.")

    invJ = np.linalg.inv(J)
    B = np.zeros((3, 8), dtype=float)

    for i in range(4):
        grad_nat = np.array([dN_dxi[i], dN_deta[i]])
        dN_dx, dN_dy = invJ.T @ grad_nat

        B[0, 2*i + 0] = dN_dx
        B[1, 2*i + 1] = dN_dy
        B[2, 2*i + 0] = dN_dy
        B[2, 2*i + 1] = dN_dx

    return B, detJ


def q4_element_stiffness(xy, D, thickness):
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    ke = np.zeros((8, 8), dtype=float)
    detJ_values = []

    for xi, eta in gauss_points:
        B, detJ = q4_B_matrix(xy, xi, eta)
        ke += thickness * (B.T @ D @ B) * detJ
        detJ_values.append(detJ)

    return ke, detJ_values


def cst_B_matrix(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    if A <= 0.0:
        raise ValueError("Área <= 0. Verifique a ordem dos nós.")

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = 1.0 / (2.0 * A) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3],
    ], dtype=float)

    return B, A


def cst_element_stiffness(xy, D, thickness):
    B, A = cst_B_matrix(xy)
    ke = thickness * A * (B.T @ D @ B)
    return ke, A


def main():
    E = 210e9
    nu = 0.30
    thickness = 0.01
    D = D_plane_stress(E, nu)

    # Triângulo físico usado no CST.
    # Nós locais: 1, 2, 3
    xy_cst = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
    ], dtype=float)

    # Q4 degenerado: nó 4 coincide com o nó 3.
    # Nós locais: 1, 2, 3, 4
    xy_q4_degenerado = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)

    K_cst, area_cst = cst_element_stiffness(xy_cst, D, thickness)
    K_q4_8x8, detJ_values = q4_element_stiffness(xy_q4_degenerado, D, thickness)

    # Matriz de transformação para fundir os nós 3 e 4 do Q4:
    # q_Q4 = T q_TRI
    #
    # q_TRI = [u1, v1, u2, v2, u3, v3]^T
    # q_Q4  = [u1, v1, u2, v2, u3, v3, u4, v4]^T
    # com u4 = u3 e v4 = v3.
    T = np.zeros((8, 6), dtype=float)
    T[0, 0] = 1.0
    T[1, 1] = 1.0
    T[2, 2] = 1.0
    T[3, 3] = 1.0
    T[4, 4] = 1.0
    T[5, 5] = 1.0
    T[6, 4] = 1.0
    T[7, 5] = 1.0

    K_q4_degenerado_6x6 = T.T @ K_q4_8x8 @ T

    norma_cst = np.linalg.norm(K_cst)
    norma_dif = np.linalg.norm(K_q4_degenerado_6x6 - K_cst)
    erro_relativo = norma_dif / norma_cst

    print("\n========= Q4 DEGENERADO x CST =========\n")
    print(f"Área do CST = {area_cst:.6f}")
    print(f"Soma dos detJ nos 4 pontos de Gauss = {sum(detJ_values):.6f}")
    print("detJ nos pontos de Gauss do Q4 degenerado:")
    for i, detJ in enumerate(detJ_values, start=1):
        print(f"  GP{i}: detJ = {detJ:.6e}")

    print()
    print(f"||K_CST|| = {norma_cst:.6e}")
    print(f"||K_Q4_degenerado - K_CST|| = {norma_dif:.6e}")
    print(f"Erro relativo matricial = {erro_relativo:.6e}")

    print("\nConclusão numérica:")
    if erro_relativo < 1e-10:
        print("O Q4 degenerado, com os nós coincidentes fundidos, reproduz a matriz do CST.")
    else:
        print("O Q4 degenerado não reproduziu exatamente a matriz do CST neste teste.")

    print("\nObservação:")
    print("Se os dois nós coincidentes forem mantidos como graus de liberdade independentes,")
    print("o elemento fica com 8 GDL e não é equivalente ao CST, que possui 6 GDL.")


if __name__ == "__main__":
    main()
