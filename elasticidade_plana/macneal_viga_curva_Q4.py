import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TESTE DE MACNEAL - VIGA CURVA 90°
# Elementos: Q4 e CST, usando 6 elementos no arco.
# ============================================================
# Dados do enunciado:
#   raio interno  = 4.12
#   raio externo  = 4.32
#   arco          = 90°
#   espessura     = 0.1
#   E             = 1e7
#   nu            = 0.25
#
# Observação importante:
#   O enunciado mostrado não explicita o valor da força aplicada.
#   Como o problema é linear elástico, os deslocamentos escalam
#   linearmente com a força. O valor P pode ser ajustado se necessário.
# ============================================================


# ------------------------------------------------------------
# Dados do problema
# ------------------------------------------------------------
E = 1.0e7
nu = 0.25
rin = 4.12
rout = 4.32
thickness = 0.1
arc_angle = np.pi / 2.0
nel_arc = 6
P = 1.0

# Valor analítico fornecido na tabela do enunciado
u_analitico = 0.08734

# Pasta de saída
outputs_dir = Path("outputs_macneal")
outputs_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Matriz constitutiva - Estado Plano de Tensões
# ------------------------------------------------------------
def D_plane_stress(E, nu):
    return E / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=float)


# ------------------------------------------------------------
# Malha da viga curva
# ------------------------------------------------------------
def make_curved_beam_mesh(rin, rout, nel_arc, arc_angle):
    """
    Gera a malha de uma viga curva com 1 elemento na espessura radial
    e nel_arc elementos ao longo do arco.

    Ordem dos nós por seção angular:
        nó interno, nó externo

    Ordem local do Q4:
        n1 = interno, seção i
        n2 = externo, seção i
        n3 = externo, seção i+1
        n4 = interno, seção i+1
    """
    theta = np.linspace(0.0, arc_angle, nel_arc + 1)

    coords = []
    for th in theta:
        coords.append([rin * np.cos(th), rin * np.sin(th)])
        coords.append([rout * np.cos(th), rout * np.sin(th)])
    coords = np.array(coords, dtype=float)

    quads = []
    for i in range(nel_arc):
        n1 = 2*i
        n2 = 2*i + 1
        n3 = 2*(i + 1) + 1
        n4 = 2*(i + 1)
        quads.append([n1, n2, n3, n4])

    return coords, np.array(quads, dtype=int), theta


# ------------------------------------------------------------
# Elemento Q4
# ------------------------------------------------------------
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

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = np.dot(dN_dxi,  xy[:, 0])
    J[0, 1] = np.dot(dN_deta, xy[:, 0])
    J[1, 0] = np.dot(dN_dxi,  xy[:, 1])
    J[1, 1] = np.dot(dN_deta, xy[:, 1])

    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError("Elemento Q4 com detJ <= 0. Verifique a ordem dos nós.")

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
    for xi, eta in gauss_points:
        B, detJ = q4_B_matrix(xy, xi, eta)
        ke += thickness * (B.T @ D @ B) * detJ

    return ke


# ------------------------------------------------------------
# Elemento CST
# ------------------------------------------------------------
def cst_B_matrix(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

    if A <= 0.0:
        raise ValueError("Elemento CST com área <= 0. Verifique a ordem dos nós.")

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
    return thickness * A * (B.T @ D @ B)


def split_q4_to_cst(quads, diagonal="13"):
    """
    Divide cada elemento Q4 em dois CST.

    diagonal='13': usa a diagonal entre os nós locais 1 e 3
    diagonal='24': usa a diagonal entre os nós locais 2 e 4
    """
    tris = []
    for q in quads:
        n1, n2, n3, n4 = q

        if diagonal == "13":
            tris.append([n1, n2, n3])
            tris.append([n1, n3, n4])
        elif diagonal == "24":
            tris.append([n1, n2, n4])
            tris.append([n2, n3, n4])
        else:
            raise ValueError("diagonal deve ser '13' ou '24'.")

    return np.array(tris, dtype=int)


# ------------------------------------------------------------
# Montagem e solução
# ------------------------------------------------------------
def assemble_and_solve(model="Q4", diagonal="13"):
    coords, quads, theta = make_curved_beam_mesh(rin, rout, nel_arc, arc_angle)
    D = D_plane_stress(E, nu)

    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    if model.upper() == "Q4":
        elements = quads
        for q in elements:
            ke = q4_element_stiffness(coords[q], D, thickness)
            dofs = np.array([[2*n, 2*n + 1] for n in q]).ravel()
            K[np.ix_(dofs, dofs)] += ke

    elif model.upper() == "CST":
        elements = split_q4_to_cst(quads, diagonal=diagonal)
        for tri in elements:
            ke = cst_element_stiffness(coords[tri], D, thickness)
            dofs = np.array([[2*n, 2*n + 1] for n in tri]).ravel()
            K[np.ix_(dofs, dofs)] += ke

    else:
        raise ValueError("model deve ser 'Q4' ou 'CST'.")

    # Engaste na seção theta = 0: nós interno e externo da primeira seção.
    fixed_nodes = [0, 1]
    constrained = []
    for n in fixed_nodes:
        constrained.extend([2*n, 2*n + 1])

    # Carga vertical descendente na extremidade livre theta = 90°.
    # Como há dois nós na seção livre, aplica-se metade da força em cada nó.
    free_inner = 2 * nel_arc
    free_outer = 2 * nel_arc + 1
    F[2*free_inner + 1] += -P / 2.0
    F[2*free_outer + 1] += -P / 2.0

    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, constrained)

    U = np.zeros(ndof, dtype=float)
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])

    reactions = K @ U - F

    # Deslocamento médio da seção livre na direção da carga.
    uy_inner = U[2*free_inner + 1]
    uy_outer = U[2*free_outer + 1]
    deslocamento_carga = -0.5 * (uy_inner + uy_outer)

    return coords, quads, U, reactions, deslocamento_carga


# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------
def plot_mesh_and_deformed(coords, quads, U, title, output_path, scale=None):
    Uxy = U.reshape((-1, 2))

    if scale is None:
        max_u = np.max(np.linalg.norm(Uxy, axis=1))
        size = np.max(coords[:, 0]) - np.min(coords[:, 0])
        scale = 0.15 * size / max_u if max_u > 0 else 1.0

    coords_def = coords + scale * Uxy

    plt.figure(figsize=(6, 6))

    for q in quads:
        xy = coords[np.r_[q, q[0]]]
        plt.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.8)

    for q in quads:
        xy = coords_def[np.r_[q, q[0]]]
        plt.plot(xy[:, 0], xy[:, 1], "r-", linewidth=0.9)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} - deformada x {scale:.2e}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_cst_diagonal_mesh(coords, quads, diagonal, output_path):
    tris = split_q4_to_cst(quads, diagonal=diagonal)

    plt.figure(figsize=(6, 6))
    for tri in tris:
        xy = coords[np.r_[tri, tri[0]]]
        plt.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.8)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Malha CST com diagonal {diagonal}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def plot_cst_deformed(coords, quads, U, diagonal, title, output_path, scale=None):
    tris = split_q4_to_cst(quads, diagonal=diagonal)

    Uxy = U.reshape((-1, 2))

    if scale is None:
        max_u = np.max(np.linalg.norm(Uxy, axis=1))
        size = np.max(coords[:, 0]) - np.min(coords[:, 0])
        scale = 0.15 * size / max_u if max_u > 0 else 1.0

    coords_def = coords + scale * Uxy

    plt.figure(figsize=(6, 6))

    # Malha original com triângulos
    for tri in tris:
        xy = coords[np.r_[tri, tri[0]]]
        plt.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.8)

    # Malha deformada com triângulos
    for tri in tris:
        xy = coords_def[np.r_[tri, tri[0]]]
        plt.plot(xy[:, 0], xy[:, 1], "r-", linewidth=0.9)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} - deformada x {scale:.2e}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
def main():
    coords_q4, quads_q4, U_q4, R_q4, u_q4 = assemble_and_solve(model="Q4")
    coords_cst, quads_cst, U_cst, R_cst, u_cst = assemble_and_solve(model="CST", diagonal="13")

    erro_q4 = abs(u_q4 - u_analitico) / abs(u_analitico) * 100.0
    erro_cst = abs(u_cst - u_analitico) / abs(u_analitico) * 100.0

    print("\n================ TESTE DE MACNEAL - VIGA CURVA ================\n")
    print(f"raio interno  = {rin}")
    print(f"raio externo  = {rout}")
    print(f"espessura     = {thickness}")
    print(f"E             = {E:.4e}")
    print(f"nu            = {nu}")
    print(f"arco          = 90 graus")
    print(f"força P       = {P}")
    print(f"nº elementos no arco = {nel_arc}")
    print()
    print("Deslocamento na direção da carga:")
    print(f"Analítico = {u_analitico:.6e}")
    print(f"Q4        = {u_q4:.6e}   erro = {erro_q4:.3f} %")
    print(f"CST       = {u_cst:.6e}   erro = {erro_cst:.3f} %")
    print()
    print("Referência da tabela do enunciado:")
    print("Q2 = 0,002184; Q4 = 0,07275; Q8 = 0,08716")

    plot_mesh_and_deformed(
        coords_q4, quads_q4, U_q4,
        "MacNeal - Q4",
        outputs_dir / "macneal_q4_deformada.png"
    )

    plot_mesh_and_deformed(
        coords_cst, quads_cst, U_cst,
        "MacNeal - CST",
        outputs_dir / "macneal_cst_deformada.png"
    )

    plot_cst_diagonal_mesh(
        coords_cst, quads_cst,
        diagonal="13",
        output_path=outputs_dir / "macneal_cst_malha_diagonal.png"
    )

    plot_cst_deformed(
    coords_cst,
    quads_cst,
    U_cst,
    diagonal="13",
    title="MacNeal - CST",
    output_path=outputs_dir / "macneal_cst_deformada_com_diagonal.png"
    )

    print(f"\nFiguras salvas em: {outputs_dir.resolve()}\n")

    plt.show()


if __name__ == "__main__":
    main()
