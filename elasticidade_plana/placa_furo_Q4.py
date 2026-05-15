import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# PLACA INFINITA COM FURO CENTRAL - EPT - ELEMENTO Q4
# ============================================================
# Modelo numérico:
#   - Usa 1/4 da placa por simetria.
#   - O domínio infinito é truncado em r = R.
#   - A borda do furo r = a fica livre de tração.
#   - Na borda externa r = R, aplica-se a tração analítica de Kirsch.
#   - Elemento usado: quadrilátero bilinear Q4 com integração 2x2.
#
# Resultado analítico esperado:
#   Kt = sigma_theta_theta(r=a, theta=90°) / sigma0 = 3.
#
# Observação:
#   A tensão na borda do furo é estimada por recuperação nodal
#   a partir das tensões calculadas nos pontos de Gauss.
# ============================================================


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
# Funções de forma do Q4
# ------------------------------------------------------------
def q4_shape_functions(xi, eta):
    """
    Elemento Q4 mestre.

    Ordem local dos nós:
        4 ---- 3
        |      |
        1 ---- 2

    Coordenadas naturais:
        nó 1: (-1, -1)
        nó 2: ( 1, -1)
        nó 3: ( 1,  1)
        nó 4: (-1,  1)
    """
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
    """
    Calcula a matriz B do elemento Q4 em um ponto natural (xi, eta).

    xy: matriz 4x2 com coordenadas dos nós na ordem [1, 2, 3, 4].
    """
    _, dN_dxi, dN_deta = q4_shape_functions(xi, eta)

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = np.dot(dN_dxi,  xy[:, 0])  # dx/dxi
    J[0, 1] = np.dot(dN_deta, xy[:, 0])  # dx/deta
    J[1, 0] = np.dot(dN_dxi,  xy[:, 1])  # dy/dxi
    J[1, 1] = np.dot(dN_deta, xy[:, 1])  # dy/deta

    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError("Elemento Q4 com detJ <= 0. Verifique a conectividade.")

    invJ = np.linalg.inv(J)

    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        grad_nat = np.array([dN_dxi[i], dN_deta[i]])
        grad_xy = invJ.T @ grad_nat
        dN_dx, dN_dy = grad_xy

        B[0, 2*i + 0] = dN_dx
        B[1, 2*i + 1] = dN_dy
        B[2, 2*i + 0] = dN_dy
        B[2, 2*i + 1] = dN_dx

    return B, detJ


# ------------------------------------------------------------
# Solução analítica de Kirsch
# ------------------------------------------------------------
def kirsch_stress_polar(r, theta, a, sigma0):
    """
    Tensões analíticas em coordenadas polares para placa infinita
    com furo circular sob tração remota sigma0 na direção x.

    Convenção:
        e_r     = [cos(theta), sin(theta)]
        e_theta = [-sin(theta), cos(theta)]
    """
    ar2 = (a / r) ** 2
    ar4 = ar2 ** 2

    sigma_rr = (sigma0 / 2.0) * (1.0 - ar2) \
             + (sigma0 / 2.0) * (1.0 - 4.0 * ar2 + 3.0 * ar4) * np.cos(2.0 * theta)

    sigma_tt = (sigma0 / 2.0) * (1.0 + ar2) \
             - (sigma0 / 2.0) * (1.0 + 3.0 * ar4) * np.cos(2.0 * theta)

    tau_rt = -(sigma0 / 2.0) * (1.0 + 2.0 * ar2 - 3.0 * ar4) * np.sin(2.0 * theta)

    return sigma_rr, sigma_tt, tau_rt


def sigma_theta_from_cartesian(stress_xy, theta):
    """
    Calcula sigma_theta_theta a partir de [sigma_xx, sigma_yy, tau_xy].
    """
    sigma_xx, sigma_yy, tau_xy = stress_xy
    s = np.sin(theta)
    c = np.cos(theta)
    return sigma_xx * s**2 + sigma_yy * c**2 - 2.0 * tau_xy * s * c


# ------------------------------------------------------------
# Malha Q4 de 1/4 de anel
# ------------------------------------------------------------
def make_quarter_annulus_q4_mesh(a=1.0, R=6.0, nr=24, nt=96, radial_power=2.0):
    """
    Gera uma malha estruturada Q4 para 1/4 de anel.

    radial_power controla o refinamento radial perto do furo:
        radial_power = 1.0 -> espaçamento radial uniforme
        radial_power = 2.0 -> mais elementos próximos ao furo

    A concentração de tensões ocorre na borda do furo, então é útil
    refinar mais essa região.
    """
    s = np.linspace(0.0, 1.0, nr + 1)
    r_values = a + (R - a) * s**radial_power
    theta_values = np.linspace(0.0, np.pi / 2.0, nt + 1)

    coords = []
    for r in r_values:
        for th in theta_values:
            coords.append([r * np.cos(th), r * np.sin(th)])
    coords = np.array(coords, dtype=float)

    def node(i, j):
        return i * (nt + 1) + j

    quads = []
    for i in range(nr):
        for j in range(nt):
            n1 = node(i,     j)
            n2 = node(i + 1, j)
            n3 = node(i + 1, j + 1)
            n4 = node(i,     j + 1)
            quads.append([n1, n2, n3, n4])

    return coords, np.array(quads, dtype=int), r_values, theta_values


# ------------------------------------------------------------
# Montagem global
# ------------------------------------------------------------
def assemble_q4_system(coords, quads, E, nu, thickness):
    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    D = D_plane_stress(E, nu)

    if SCIPY_AVAILABLE:
        K = lil_matrix((ndof, ndof), dtype=float)
    else:
        print("Aviso: scipy não encontrado. Usando matriz densa.")
        K = np.zeros((ndof, ndof), dtype=float)

    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    for quad in quads:
        xy = coords[quad]
        ke = np.zeros((8, 8), dtype=float)

        for xi, eta in gauss_points:
            B, detJ = q4_B_matrix(xy, xi, eta)
            ke += thickness * (B.T @ D @ B) * detJ

        dofs = np.array([
            2*quad[0] + 0, 2*quad[0] + 1,
            2*quad[1] + 0, 2*quad[1] + 1,
            2*quad[2] + 0, 2*quad[2] + 1,
            2*quad[3] + 0, 2*quad[3] + 1,
        ], dtype=int)

        for ii in range(8):
            for jj in range(8):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]

    if SCIPY_AVAILABLE:
        K = K.tocsr()

    F = np.zeros(ndof, dtype=float)
    return K, F, D


# ------------------------------------------------------------
# Tração analítica na borda externa r = R
# ------------------------------------------------------------
def apply_outer_kirsch_traction_q4(F, coords, R, a, sigma0, thickness, nr, nt):
    """
    Aplica tração na borda externa r = R.
    A tração é obtida pela solução de Kirsch no contorno truncado.
    """
    def node(i, j):
        return i * (nt + 1) + j

    for j in range(nt):
        n1 = node(nr, j)
        n2 = node(nr, j + 1)

        p1 = coords[n1]
        p2 = coords[n2]
        edge_length = np.linalg.norm(p2 - p1)
        midpoint = 0.5 * (p1 + p2)
        theta_mid = np.arctan2(midpoint[1], midpoint[0])

        sigma_rr, _, tau_rt = kirsch_stress_polar(R, theta_mid, a, sigma0)

        er = np.array([np.cos(theta_mid), np.sin(theta_mid)])
        et = np.array([-np.sin(theta_mid), np.cos(theta_mid)])
        traction = sigma_rr * er + tau_rt * et

        # Vetor de força consistente em uma aresta linear:
        # f1 = f2 = t * L_aresta / 2
        for n in [n1, n2]:
            F[2*n + 0] += traction[0] * thickness * edge_length / 2.0
            F[2*n + 1] += traction[1] * thickness * edge_length / 2.0


# ------------------------------------------------------------
# Condições de simetria
# ------------------------------------------------------------
def get_symmetry_constraints(nr, nt):
    """
    Para 1/4 do domínio:
      - theta = 0°: v = 0
      - theta = 90°: u = 0
    """
    constrained = []

    def node(i, j):
        return i * (nt + 1) + j

    for i in range(nr + 1):
        constrained.append(2 * node(i, 0) + 1)   # v = 0 em theta = 0
        constrained.append(2 * node(i, nt) + 0)  # u = 0 em theta = 90°

    return np.array(sorted(set(constrained)), dtype=int)


# ------------------------------------------------------------
# Solução do sistema linear
# ------------------------------------------------------------
def solve_system(K, F, constrained):
    ndof = len(F)
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, constrained)

    U = np.zeros(ndof, dtype=float)

    if SCIPY_AVAILABLE:
        U[free] = spsolve(K[free][:, free], F[free])
        reactions = K @ U - F
    else:
        U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        reactions = K @ U - F

    return U, reactions


# ------------------------------------------------------------
# Pós-processamento de tensões
# ------------------------------------------------------------
def compute_q4_stresses(coords, quads, U, D):
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    gp_stresses = []
    centroids = np.zeros((len(quads), 2), dtype=float)

    for e, quad in enumerate(quads):
        xy = coords[quad]
        dofs = np.array([
            2*quad[0] + 0, 2*quad[0] + 1,
            2*quad[1] + 0, 2*quad[1] + 1,
            2*quad[2] + 0, 2*quad[2] + 1,
            2*quad[3] + 0, 2*quad[3] + 1,
        ], dtype=int)
        ue = U[dofs]

        elem_gp_stress = []
        for xi, eta in gauss_points:
            B, _ = q4_B_matrix(xy, xi, eta)
            strain = B @ ue
            stress = D @ strain
            elem_gp_stress.append(stress)

        gp_stresses.append(np.array(elem_gp_stress))
        centroids[e] = xy.mean(axis=0)

    return gp_stresses, centroids


def recover_nodal_stresses_from_gauss(coords, quads, gp_stresses):
    """
    Recupera tensões nodais por extrapolação dos 4 pontos de Gauss
    para os 4 nós do elemento e posterior média entre elementos vizinhos.
    """
    n_nodes = coords.shape[0]
    nodal_stress = np.zeros((n_nodes, 3), dtype=float)
    count = np.zeros(n_nodes, dtype=float)

    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    # Matriz das funções de forma avaliadas nos pontos de Gauss.
    # nodal_values = inv(N_gp) @ values_gp
    N_gp = np.array([q4_shape_functions(xi, eta)[0] for xi, eta in gauss_points])
    extrap = np.linalg.inv(N_gp)

    for e, quad in enumerate(quads):
        stress_nodes_local = extrap @ gp_stresses[e]

        for local_i, node_i in enumerate(quad):
            nodal_stress[node_i] += stress_nodes_local[local_i]
            count[node_i] += 1.0

    count[count == 0.0] = 1.0
    nodal_stress /= count[:, None]
    return nodal_stress


def estimate_Kt_q4(nodal_stress, theta_values, sigma0, nt):
    """
    Estima Kt na borda do furo a partir das tensões nodais recuperadas.
    A borda do furo corresponde ao anel i = 0.
    """
    sigma_tt_hole = []

    def node(i, j):
        return i * (nt + 1) + j

    for j, theta in enumerate(theta_values):
        n = node(0, j)
        sigma_tt = sigma_theta_from_cartesian(nodal_stress[n], theta)
        sigma_tt_hole.append(sigma_tt / sigma0)

    sigma_tt_hole = np.array(sigma_tt_hole)
    idx_max = int(np.argmax(sigma_tt_hole))

    Kt_theta_90 = sigma_tt_hole[-1]
    Kt_max = sigma_tt_hole[idx_max]
    theta_max = theta_values[idx_max]

    return Kt_theta_90, Kt_max, theta_max, sigma_tt_hole


# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------
def plot_q4_mesh(coords, quads, output_path):
    plt.figure(figsize=(6, 6))
    for quad in quads:
        xy = coords[np.r_[quad, quad[0]]]
        plt.plot(xy[:, 0], xy[:, 1], linewidth=0.35)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Malha Q4 - 1/4 de placa com furo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_deformed_q4(coords, quads, U, output_path, scale=None):
    Uxy = U.reshape((-1, 2))

    if scale is None:
        max_u = np.max(np.linalg.norm(Uxy, axis=1))
        domain_size = np.max(coords[:, 0]) - np.min(coords[:, 0])
        scale = 0.08 * domain_size / max_u if max_u > 0.0 else 1.0

    coords_def = coords + scale * Uxy

    plt.figure(figsize=(6, 6))
    for quad in quads:
        xy = coords[np.r_[quad, quad[0]]]
        plt.plot(xy[:, 0], xy[:, 1], linewidth=0.25)

    for quad in quads:
        xy = coords_def[np.r_[quad, quad[0]]]
        plt.plot(xy[:, 0], xy[:, 1], linewidth=0.35)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Malha inicial e deformada - Q4 x {scale:.2e}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_sigma_theta_field(coords, quads, nodal_stress, output_path):
    """
    Para visualizar campo em tripcolor, divide cada Q4 em dois triângulos.
    O valor nodal plotado é sigma_theta_theta/sigma0 calculado em cada nó.
    """
    tris = []
    for q in quads:
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
    tris = np.array(tris, dtype=int)

    # Este sigma0 é 1.0 no script principal. Como a análise é linear,
    # a variável já está normalizada se sigma0=1.
    values = np.zeros(coords.shape[0], dtype=float)
    for i, p in enumerate(coords):
        theta = np.arctan2(p[1], p[0])
        values[i] = sigma_theta_from_cartesian(nodal_stress[i], theta)

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)

    plt.figure(figsize=(7, 6))
    plt.tripcolor(triang, values, shading="gouraud")
    plt.colorbar(label=r"$\sigma_{\theta\theta}/\sigma_0$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Distribuição de $\sigma_{\theta\theta}/\sigma_0$ - Q4")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_hole_stress_q4(theta_values, sigma_tt_hole, a, sigma0, output_path):
    analytical = []
    for theta in theta_values:
        _, sigma_tt_an, _ = kirsch_stress_polar(a, theta, a, sigma0)
        analytical.append(sigma_tt_an / sigma0)
    analytical = np.array(analytical)

    theta_deg = np.degrees(theta_values)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, analytical, label="analítico")
    plt.plot(theta_deg, sigma_tt_hole, "o-", markersize=3, label="Q4 - tensão nodal recuperada")
    plt.axhline(3.0, linestyle="--", linewidth=1.0, label=r"$K_t=3$")
    plt.xlabel(r"$\theta$ [graus]")
    plt.ylabel(r"$\sigma_{\theta\theta}/\sigma_0$ na borda do furo")
    plt.title("Tensão circunferencial na borda do furo - Q4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------
def main():
    outputs_dir = Path("outputs_furo_q4")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Dados do problema
    E = 210e9
    nu = 0.30
    thickness = 1.0
    sigma0 = 1.0

    # Geometria do domínio truncado
    a = 1.0
    R = 6.0 * a

    # Malha Q4
    nr = 24
    nt = 96
    radial_power = 2.0  # refinamento radial próximo ao furo

    coords, quads, r_values, theta_values = make_quarter_annulus_q4_mesh(
        a=a, R=R, nr=nr, nt=nt, radial_power=radial_power
    )

    K, F, D = assemble_q4_system(coords, quads, E, nu, thickness)
    apply_outer_kirsch_traction_q4(F, coords, R, a, sigma0, thickness, nr, nt)

    constrained = get_symmetry_constraints(nr, nt)
    U, reactions = solve_system(K, F, constrained)

    gp_stresses, centroids = compute_q4_stresses(coords, quads, U, D)
    nodal_stress = recover_nodal_stresses_from_gauss(coords, quads, gp_stresses)

    Kt_theta_90, Kt_max, theta_max, sigma_tt_hole = estimate_Kt_q4(
        nodal_stress, theta_values, sigma0, nt
    )

    Kt_analytical = 3.0
    error_theta_90 = abs(Kt_theta_90 - Kt_analytical) / Kt_analytical * 100.0
    error_max = abs(Kt_max - Kt_analytical) / Kt_analytical * 100.0

    # Valores obtidos anteriormente com CST para comparação.
    # Atualize estes valores caso rode uma malha CST diferente.
    Kt_cst = 2.982744
    error_cst = abs(Kt_cst - Kt_analytical) / Kt_analytical * 100.0

    print("\n================ PLACA INFINITA COM FURO - Q4 ================\n")
    print("Estado plano de tensões (EPT)")
    print(f"a = {a:.4g}")
    print(f"R/a = {R/a:.4g}")
    print(f"nr = {nr}, nt = {nt}, radial_power = {radial_power}")
    print(f"nós = {coords.shape[0]}, elementos Q4 = {len(quads)}")
    print()
    print("Fator de concentração de tensões:")
    print(f"Kt analítico                         = {Kt_analytical:.6f}")
    print(f"Kt numérico Q4, theta=90°            = {Kt_theta_90:.6f}")
    print(f"Erro Q4 no ponto theta=90°           = {error_theta_90:.3f} %")
    print(f"Kt numérico Q4, máximo na borda      = {Kt_max:.6f}")
    print(f"Erro Q4 no máximo da borda           = {error_max:.3f} %")
    print(f"Theta do máximo Q4                   = {np.degrees(theta_max):.3f} graus")
    print()
    print("Comparação com CST da lista 6:")
    print(f"Kt CST                               = {Kt_cst:.6f}")
    print(f"Erro CST                             = {error_cst:.3f} %")

    plot_q4_mesh(coords, quads, outputs_dir / "malha_q4_quarto_anel.png")
    plot_deformed_q4(coords, quads, U, outputs_dir / "malha_q4_deformada.png")
    plot_sigma_theta_field(coords, quads, nodal_stress, outputs_dir / "sigma_theta_q4_normalizada.png")
    plot_hole_stress_q4(theta_values, sigma_tt_hole, a, sigma0, outputs_dir / "sigma_theta_borda_furo_q4.png")

    print(f"\nFiguras salvas em: {outputs_dir.resolve()}\n")

    plt.show()


if __name__ == "__main__":
    main()
