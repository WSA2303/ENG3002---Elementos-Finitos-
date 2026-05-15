import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# PLACA INFINITA COM FURO CENTRAL - EPT - ELEMENTO CST
# ============================================================
# Modelo numérico:
#   - Usa 1/4 de uma placa infinita, por simetria.
#   - O domínio infinito é truncado em um raio externo R.
#   - Na borda externa r = R, aplica-se a tração analítica de Kirsch.
#   - Na borda do furo r = a, a superfície é livre de tração.
#   - Elemento utilizado: CST, triangular linear de deformação constante.
#
# Resultado esperado:
#   Kt analítico = sigma_theta_theta(r=a, theta=90°)/sigma0 = 3.
#
# Observação:
#   A variável theta é medida a partir do eixo x positivo.
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
# Matriz B do CST
# ------------------------------------------------------------
def cst_B_matrix(xy):
    """
    xy: matriz 3x2 com coordenadas dos 3 nós do triângulo.
    Retorna B e área do triângulo.
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * twoA

    if A <= 0:
        raise ValueError("Elemento com área não positiva. Verifique a conectividade.")

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = 1.0 / (2.0 * A) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3]
    ], dtype=float)

    return B, A


# ------------------------------------------------------------
# Solução analítica de Kirsch
# ------------------------------------------------------------
def kirsch_stress_polar(r, theta, a, sigma0):
    """
    Tensões analíticas em coordenadas polares para placa infinita
    com furo circular sob tração remota sigma0 na direção x.

    Convenção usada:
        e_r     = [cos(theta), sin(theta)]
        e_theta = [-sin(theta), cos(theta)]

    Com essa convenção, tau_rtheta possui sinal negativo na solução clássica.
    Algumas apostilas escrevem tau_rtheta com sinal oposto por adotarem
    convenção diferente para a direção theta.
    """
    ar2 = (a / r) ** 2
    ar4 = ar2 ** 2

    sigma_rr = (sigma0 / 2.0) * (1.0 - ar2) \
             + (sigma0 / 2.0) * (1.0 - 4.0 * ar2 + 3.0 * ar4) * np.cos(2.0 * theta)

    sigma_tt = (sigma0 / 2.0) * (1.0 + ar2) \
             - (sigma0 / 2.0) * (1.0 + 3.0 * ar4) * np.cos(2.0 * theta)

    tau_rt = -(sigma0 / 2.0) * (1.0 + 2.0 * ar2 - 3.0 * ar4) * np.sin(2.0 * theta)

    return sigma_rr, sigma_tt, tau_rt


def polar_to_cartesian_stress(sigma_rr, sigma_tt, tau_rt, theta):
    """
    Converte tensões polares para cartesianas.
    Retorna [sigma_xx, sigma_yy, tau_xy].
    """
    c = np.cos(theta)
    s = np.sin(theta)

    sigma_xx = sigma_rr * c**2 + sigma_tt * s**2 - 2.0 * tau_rt * s * c
    sigma_yy = sigma_rr * s**2 + sigma_tt * c**2 + 2.0 * tau_rt * s * c
    tau_xy = (sigma_rr - sigma_tt) * s * c + tau_rt * (c**2 - s**2)

    return np.array([sigma_xx, sigma_yy, tau_xy], dtype=float)


def sigma_theta_from_cartesian(stress_xy, theta):
    """
    Calcula sigma_theta_theta a partir de [sigma_xx, sigma_yy, tau_xy].
    """
    sigma_xx, sigma_yy, tau_xy = stress_xy
    s = np.sin(theta)
    c = np.cos(theta)

    return sigma_xx * s**2 + sigma_yy * c**2 - 2.0 * tau_xy * s * c


# ------------------------------------------------------------
# Malha de 1/4 de anel: a <= r <= R, 0 <= theta <= pi/2
# ------------------------------------------------------------
def make_quarter_annulus_mesh(a=1.0, R=6.0, nr=24, nt=96):
    """
    Gera uma malha estruturada de elementos triangulares CST
    para 1/4 de anel.

    nr: divisões radiais
    nt: divisões angulares
    """
    r_values = np.linspace(a, R, nr + 1)
    theta_values = np.linspace(0.0, np.pi / 2.0, nt + 1)

    coords = []
    for i, r in enumerate(r_values):
        for j, th in enumerate(theta_values):
            coords.append([r * np.cos(th), r * np.sin(th)])
    coords = np.array(coords, dtype=float)

    def node(i, j):
        return i * (nt + 1) + j

    tris = []
    for i in range(nr):
        for j in range(nt):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)

            # Divisão de cada quadrilátero curvo em dois triângulos.
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])

    return coords, np.array(tris, dtype=int), r_values, theta_values


# ------------------------------------------------------------
# Montagem global
# ------------------------------------------------------------
def assemble_cst_system(coords, tris, E, nu, thickness):
    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    D = D_plane_stress(E, nu)

    if SCIPY_AVAILABLE:
        K = lil_matrix((ndof, ndof), dtype=float)
    else:
        print("Aviso: scipy não encontrado. Usando matriz densa.")
        K = np.zeros((ndof, ndof), dtype=float)

    for tri in tris:
        xy = coords[tri]
        B, A = cst_B_matrix(xy)
        ke = thickness * A * (B.T @ D @ B)

        dofs = np.array([
            2 * tri[0] + 0, 2 * tri[0] + 1,
            2 * tri[1] + 0, 2 * tri[1] + 1,
            2 * tri[2] + 0, 2 * tri[2] + 1
        ], dtype=int)

        for ii in range(6):
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]

    if SCIPY_AVAILABLE:
        K = K.tocsr()

    F = np.zeros(ndof, dtype=float)
    return K, F, D


# ------------------------------------------------------------
# Carregamento externo analítico na borda r = R
# ------------------------------------------------------------
def apply_outer_kirsch_traction(F, coords, R, a, sigma0, thickness, nr, nt):
    """
    Aplica o vetor de forças nodais consistente na borda externa r = R.
    A tração é dada por t = sigma_rr e_r + tau_rtheta e_theta.
    """
    def node(i, j):
        return i * (nt + 1) + j

    for j in range(nt):
        n1 = node(nr, j)
        n2 = node(nr, j + 1)

        p1 = coords[n1]
        p2 = coords[n2]
        edge_length = np.linalg.norm(p2 - p1)

        theta_mid = np.arctan2(0.5 * (p1[1] + p2[1]), 0.5 * (p1[0] + p2[0]))

        sigma_rr, _, tau_rt = kirsch_stress_polar(R, theta_mid, a, sigma0)

        er = np.array([np.cos(theta_mid), np.sin(theta_mid)])
        et = np.array([-np.sin(theta_mid), np.cos(theta_mid)])

        traction = sigma_rr * er + tau_rt * et

        # Força nodal consistente em aresta linear:
        # f1 = f2 = t * espessura * L_aresta / 2
        for n in [n1, n2]:
            F[2 * n + 0] += traction[0] * thickness * edge_length / 2.0
            F[2 * n + 1] += traction[1] * thickness * edge_length / 2.0


# ------------------------------------------------------------
# Condições de simetria
# ------------------------------------------------------------
def get_symmetry_constraints(nr, nt):
    """
    1/4 do domínio:
      - eixo x, theta=0: v = 0
      - eixo y, theta=90°: u = 0
    """
    constrained = []

    def node(i, j):
        return i * (nt + 1) + j

    # theta = 0 -> j = 0 -> restringe v
    for i in range(nr + 1):
        n = node(i, 0)
        constrained.append(2 * n + 1)

    # theta = pi/2 -> j = nt -> restringe u
    for i in range(nr + 1):
        n = node(i, nt)
        constrained.append(2 * n + 0)

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
def compute_element_stresses(coords, tris, U, D):
    n_elem = len(tris)
    strains = np.zeros((n_elem, 3), dtype=float)
    stresses = np.zeros((n_elem, 3), dtype=float)
    centroids = np.zeros((n_elem, 2), dtype=float)

    for e, tri in enumerate(tris):
        xy = coords[tri]
        B, _ = cst_B_matrix(xy)

        dofs = np.array([
            2 * tri[0] + 0, 2 * tri[0] + 1,
            2 * tri[1] + 0, 2 * tri[1] + 1,
            2 * tri[2] + 0, 2 * tri[2] + 1
        ], dtype=int)

        ue = U[dofs]
        strains[e] = B @ ue
        stresses[e] = D @ strains[e]
        centroids[e] = xy.mean(axis=0)

    return strains, stresses, centroids


def recover_nodal_stresses(n_nodes, tris, elem_stresses):
    nodal_stress = np.zeros((n_nodes, 3), dtype=float)
    count = np.zeros(n_nodes, dtype=float)

    for e, tri in enumerate(tris):
        for n in tri:
            nodal_stress[n] += elem_stresses[e]
            count[n] += 1.0

    count[count == 0.0] = 1.0
    nodal_stress /= count[:, None]
    return nodal_stress


def estimate_Kt(coords, tris, stresses, centroids, nodal_stress, a, R, sigma0, nr, nt):
    dr = (R - a) / nr

    # 1) Estimativa no nó da borda do furo em theta = 90°.
    top_hole_node = 0 * (nt + 1) + nt
    theta_top = np.pi / 2.0
    sigma_tt_top_node = sigma_theta_from_cartesian(nodal_stress[top_hole_node], theta_top)
    Kt_top_node = sigma_tt_top_node / sigma0

    # 2) Estimativa por máximo sigma_theta_theta nos elementos do primeiro anel.
    Kt_first_ring = -np.inf
    theta_at_max = None

    for e, c in enumerate(centroids):
        r_c = np.linalg.norm(c)
        if r_c <= a + 1.5 * dr:
            theta_c = np.arctan2(c[1], c[0])
            sigma_tt = sigma_theta_from_cartesian(stresses[e], theta_c)
            val = sigma_tt / sigma0
            if val > Kt_first_ring:
                Kt_first_ring = val
                theta_at_max = theta_c

    return Kt_top_node, Kt_first_ring, theta_at_max


# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------
def plot_mesh(coords, tris, output_path):
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)

    plt.figure(figsize=(6, 6))
    plt.triplot(triang, linewidth=0.35)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Malha CST - 1/4 de placa com furo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_sigma_theta(coords, tris, stresses, centroids, sigma0, output_path):
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)

    sigma_tt = np.zeros(len(tris), dtype=float)
    for e, c in enumerate(centroids):
        theta = np.arctan2(c[1], c[0])
        sigma_tt[e] = sigma_theta_from_cartesian(stresses[e], theta) / sigma0

    plt.figure(figsize=(7, 6))
    plt.tripcolor(triang, facecolors=sigma_tt, shading="flat")
    plt.colorbar(label=r"$\sigma_{\theta\theta}/\sigma_0$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Distribuição de $\sigma_{\theta\theta}/\sigma_0$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_hole_stress(theta_values, nodal_stress, a, sigma0, nt, output_path):
    numerical = []
    analytical = []

    for j, theta in enumerate(theta_values):
        n = j  # nó na borda interna: i=0, j=j
        sigma_tt_num = sigma_theta_from_cartesian(nodal_stress[n], theta) / sigma0
        _, sigma_tt_an, _ = kirsch_stress_polar(a, theta, a, sigma0)

        numerical.append(sigma_tt_num)
        analytical.append(sigma_tt_an / sigma0)

    theta_deg = np.degrees(theta_values)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, analytical, label="analítico")
    plt.plot(theta_deg, numerical, "o-", markersize=3, label="CST - tensão nodal recuperada")
    plt.axhline(3.0, linestyle="--", linewidth=1.0, label=r"$K_t=3$")
    plt.xlabel(r"$\theta$ [graus]")
    plt.ylabel(r"$\sigma_{\theta\theta}/\sigma_0$ na borda do furo")
    plt.title("Tensão circunferencial na borda do furo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_deformed(coords, tris, U, output_path, scale=None):
    Uxy = U.reshape((-1, 2))

    if scale is None:
        max_u = np.max(np.linalg.norm(Uxy, axis=1))
        domain_size = np.max(coords[:, 0]) - np.min(coords[:, 0])
        scale = 0.08 * domain_size / max_u if max_u > 0 else 1.0

    coords_def = coords + scale * Uxy

    triang0 = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
    triang1 = mtri.Triangulation(coords_def[:, 0], coords_def[:, 1], tris)

    plt.figure(figsize=(6, 6))
    plt.triplot(triang0, linewidth=0.25, label="inicial")
    plt.triplot(triang1, linewidth=0.35, label=f"deformada x {scale:.2e}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Malha inicial e deformada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------
def main():
    outputs_dir = Path("outputs_furo_cst")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Dados do problema
    E = 210e9          # Pa
    nu = 0.30          # adimensional
    thickness = 1.0    # espessura, arbitrária para EPT
    sigma0 = 1.0       # tensão remota de referência

    # Geometria truncada do domínio infinito
    a = 1.0            # raio do furo
    R = 6.0 * a        # raio externo do modelo truncado

    # Refinamento da malha
    nr = 24            # divisões radiais
    nt = 96            # divisões angulares no quarto de domínio

    coords, tris, r_values, theta_values = make_quarter_annulus_mesh(a=a, R=R, nr=nr, nt=nt)

    K, F, D = assemble_cst_system(coords, tris, E, nu, thickness)
    apply_outer_kirsch_traction(F, coords, R, a, sigma0, thickness, nr, nt)

    constrained = get_symmetry_constraints(nr, nt)
    U, reactions = solve_system(K, F, constrained)

    strains, stresses, centroids = compute_element_stresses(coords, tris, U, D)
    nodal_stress = recover_nodal_stresses(coords.shape[0], tris, stresses)

    Kt_top_node, Kt_first_ring, theta_at_max = estimate_Kt(
        coords, tris, stresses, centroids, nodal_stress,
        a, R, sigma0, nr, nt
    )

    Kt_analytical = 3.0
    error_top_node = abs(Kt_top_node - Kt_analytical) / Kt_analytical * 100.0
    error_first_ring = abs(Kt_first_ring - Kt_analytical) / Kt_analytical * 100.0

    print("\n================ PLACA INFINITA COM FURO - CST ================\n")
    print(f"Estado plano de tensões (EPT)")
    print(f"a = {a:.4g}")
    print(f"R/a = {R/a:.4g}")
    print(f"nr = {nr}, nt = {nt}")
    print(f"nós = {coords.shape[0]}, elementos CST = {len(tris)}")
    print()
    print("Fator de concentração de tensões:")
    print(f"Kt analítico                         = {Kt_analytical:.6f}")
    print(f"Kt numérico, nó theta=90°             = {Kt_top_node:.6f}")
    print(f"Erro no nó theta=90°                  = {error_top_node:.3f} %")
    print(f"Kt numérico, máximo no primeiro anel  = {Kt_first_ring:.6f}")
    print(f"Erro no máximo do primeiro anel       = {error_first_ring:.3f} %")
    print(f"Theta do máximo no primeiro anel      = {np.degrees(theta_at_max):.3f} graus")

    plot_mesh(coords, tris, outputs_dir / "malha_cst_quarto_anel.png")
    plot_sigma_theta(coords, tris, stresses, centroids, sigma0, outputs_dir / "sigma_theta_normalizada.png")
    plot_hole_stress(theta_values, nodal_stress, a, sigma0, nt, outputs_dir / "sigma_theta_borda_furo.png")
    plot_deformed(coords, tris, U, outputs_dir / "malha_deformada.png")

    print(f"\nFiguras salvas em: {outputs_dir.resolve()}\n")

    plt.show()


if __name__ == "__main__":
    main()
