import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# MODELOS DE MEF PARA VIGA EM BALANÇO:
# (a) Elemento de viga/pórtico plano Euler-Bernoulli
# (b), (c), (d) Elementos quadrangulares Q4
# ============================================================


@dataclass
class BeamResult:
    x: np.ndarray
    U: np.ndarray
    K: np.ndarray
    F: np.ndarray
    reactions: np.ndarray
    constrained: np.ndarray
    v_tip: float
    v_tip_analytical: float


@dataclass
class Q4Result:
    coords: np.ndarray
    quads: np.ndarray
    U: np.ndarray
    K: np.ndarray
    F: np.ndarray
    reactions: np.ndarray
    constrained: np.ndarray
    strain_gp: list
    stress_gp: list
    v_right_mid: float
    v_right_avg: float
    mesh_name: str
    load_case: str


# ============================================================
# SOLUÇÃO DE SISTEMA LINEAR COM CONDIÇÕES DE CONTORNO ESSENCIAIS
# ============================================================

def solve_linear_system(K, F, constrained_dofs):
    ndof = K.shape[0]
    constrained = np.array(sorted(set(constrained_dofs)), dtype=int)
    free = np.setdiff1d(np.arange(ndof), constrained)

    U = np.zeros(ndof)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    reactions = K @ U - F
    return U, reactions, constrained


# ============================================================
# MODELO (a): VIGA / PÓRTICO PLANO
# ============================================================

def frame_element_stiffness(E, A, I, Le):
    """
    Matriz de rigidez de elemento de viga/pórtico plano horizontal.

    GDL locais:
    [u1, v1, theta1, u2, v2, theta2]
    """
    EA = E * A
    EI = E * I

    k = np.array([
        [ EA/Le,           0,            0, -EA/Le,           0,            0],
        [     0,  12*EI/Le**3,  6*EI/Le**2,      0, -12*EI/Le**3,  6*EI/Le**2],
        [     0,   6*EI/Le**2,    4*EI/Le,      0,  -6*EI/Le**2,    2*EI/Le],
        [-EA/Le,           0,            0,  EA/Le,           0,            0],
        [     0, -12*EI/Le**3, -6*EI/Le**2,      0,  12*EI/Le**3, -6*EI/Le**2],
        [     0,   6*EI/Le**2,    2*EI/Le,      0,  -6*EI/Le**2,    4*EI/Le],
    ], dtype=float)

    return k


def solve_beam_model(E=210e9, L=1.0, h=0.2, t=0.01, P=1000.0, n_elem=20):
    """
    Modelo (a): viga/pórtico plano Euler-Bernoulli.
    P > 0 significa força vertical para baixo na extremidade livre.
    """
    A = t * h
    I = t * h**3 / 12.0

    n_nodes = n_elem + 1
    x = np.linspace(0.0, L, n_nodes)

    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    Le = L / n_elem
    ke = frame_element_stiffness(E, A, I, Le)

    for e in range(n_elem):
        n1, n2 = e, e + 1
        dofs = np.array([
            3*n1 + 0, 3*n1 + 1, 3*n1 + 2,
            3*n2 + 0, 3*n2 + 1, 3*n2 + 2
        ])
        K[np.ix_(dofs, dofs)] += ke

    # Carga concentrada na ponta livre, direção y negativa.
    F[3*(n_nodes - 1) + 1] = -P

    # Engaste à esquerda: u = v = theta = 0.
    constrained = [0, 1, 2]

    U, reactions, constrained = solve_linear_system(K, F, constrained)

    v_tip = U[3*(n_nodes - 1) + 1]
    v_tip_analytical = -P * L**3 / (3.0 * E * I)

    return BeamResult(
        x=x,
        U=U,
        K=K,
        F=F,
        reactions=reactions,
        constrained=constrained,
        v_tip=v_tip,
        v_tip_analytical=v_tip_analytical
    )


# ============================================================
# ELEMENTO Q4
# ============================================================

def D_matrix(E, nu, plane="stress"):
    """
    Matriz constitutiva para elasticidade plana.

    plane='stress' -> Estado Plano de Tensões, EPT
    plane='strain' -> Estado Plano de Deformações, EPD
    """
    if plane.lower() in ("stress", "ept", "plane_stress"):
        return E / (1.0 - nu**2) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu)/2.0]
        ], dtype=float)

    if plane.lower() in ("strain", "epd", "plane_strain"):
        return E / ((1.0 + nu) * (1.0 - 2.0*nu)) * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0*nu)/2.0]
        ], dtype=float)

    raise ValueError("Use plane='stress' para EPT ou plane='strain' para EPD.")


def q4_shape_functions(xi, eta):
    """
    Funções de forma do quadrilátero bilinear Q4 no elemento mestre.

    Ordem dos nós locais:
        4 ---- 3
        |      |
        1 ---- 2

    Coordenadas naturais:
        nó 1: (-1,-1)
        nó 2: ( 1,-1)
        nó 3: ( 1, 1)
        nó 4: (-1, 1)
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
    Calcula a matriz B do elemento Q4 em um ponto de Gauss.

    xy: matriz 4x2 com coordenadas dos nós na ordem [1,2,3,4].
    """
    _, dN_dxi, dN_deta = q4_shape_functions(xi, eta)

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = np.dot(dN_dxi,  xy[:, 0])  # dx/dxi
    J[0, 1] = np.dot(dN_deta, xy[:, 0])  # dx/deta
    J[1, 0] = np.dot(dN_dxi,  xy[:, 1])  # dy/dxi
    J[1, 1] = np.dot(dN_deta, xy[:, 1])  # dy/deta

    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError("Elemento Q4 com detJ <= 0. Verifique a ordem dos nós.")

    invJ = np.linalg.inv(J)

    dN_dx = np.zeros(4, dtype=float)
    dN_dy = np.zeros(4, dtype=float)

    for i in range(4):
        # [dN/dx, dN/dy]^T = J^{-T} [dN/dxi, dN/deta]^T
        grad_nat = np.array([dN_dxi[i], dN_deta[i]])
        grad_xy = invJ.T @ grad_nat
        dN_dx[i] = grad_xy[0]
        dN_dy[i] = grad_xy[1]

    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        B[0, 2*i + 0] = dN_dx[i]
        B[1, 2*i + 1] = dN_dy[i]
        B[2, 2*i + 0] = dN_dy[i]
        B[2, 2*i + 1] = dN_dx[i]

    return B, detJ


def q4_element_stiffness(xy, D, t, integration="2x2"):
    """
    Matriz de rigidez do elemento Q4.

    integration='2x2' -> integração completa, padrão para Q4.
    integration='1x1' -> integração reduzida, apenas para teste.
    """
    if integration == "2x2":
        g = 1.0 / np.sqrt(3.0)
        gauss_points = [(-g, -g, 1.0), (g, -g, 1.0), (g, g, 1.0), (-g, g, 1.0)]
    elif integration == "1x1":
        gauss_points = [(0.0, 0.0, 4.0)]
    else:
        raise ValueError("integration deve ser '2x2' ou '1x1'.")

    ke = np.zeros((8, 8), dtype=float)

    for xi, eta, w in gauss_points:
        B, detJ = q4_B_matrix(xy, xi, eta)
        ke += t * (B.T @ D @ B) * detJ * w

    return ke


# ============================================================
# MALHA RETANGULAR Q4
# ============================================================

def make_rectangular_q4_mesh(L=1.0, h=0.2, nx=5, ny=1):
    """
    Gera uma malha retangular L x h com elementos Q4.

    Coordenadas:
        0 <= x <= L
        -h/2 <= y <= h/2

    Ordem local de cada elemento:
        n4 ---- n3
        |       |
        n1 ---- n2
    """
    xvals = np.linspace(0.0, L, nx + 1)
    yvals = np.linspace(-h/2.0, h/2.0, ny + 1)

    coords = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords.append([xvals[i], yvals[j]])
    coords = np.array(coords, dtype=float)

    def node(i, j):
        return j * (nx + 1) + i

    quads = []
    for j in range(ny):
        for i in range(nx):
            n1 = node(i,     j)
            n2 = node(i + 1, j)
            n3 = node(i + 1, j + 1)
            n4 = node(i,     j + 1)
            quads.append([n1, n2, n3, n4])

    return coords, np.array(quads, dtype=int)


# ============================================================
# CARREGAMENTO
# ============================================================

def apply_q4_load(F, coords, L, h, t, P, load_case="edge"):
    """
    Aplica carregamento na extremidade direita do modelo Q4.

    P > 0 significa carga vertical total para baixo.

    load_case:
        'edge'      -> força total P distribuída na borda direita.
                       Esta opção é mais adequada para comparar com a viga.
        'point_mid' -> força P concentrada no nó do meio da borda direita.
        'point_top' -> força P concentrada no canto superior direito.
                       Aproxima a seta desenhada no canto da figura.
    """
    tol = 1e-12
    right_nodes = np.where(np.isclose(coords[:, 0], L, atol=tol))[0]
    right_nodes = right_nodes[np.argsort(coords[right_nodes, 1])]

    if load_case == "edge":
        # tração vertical uniforme ty tal que ty*h*t = -P
        ty = -P / (h * t)

        for a_node, b_node in zip(right_nodes[:-1], right_nodes[1:]):
            ya, yb = coords[a_node, 1], coords[b_node, 1]
            edge_length = abs(yb - ya)

            # Vetor nodal consistente de uma aresta linear:
            # f_a = f_b = ty*t*Le/2, na direção y.
            F[2*a_node + 1] += ty * t * edge_length / 2.0
            F[2*b_node + 1] += ty * t * edge_length / 2.0

    elif load_case == "point_mid":
        mid_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1]))]
        F[2*mid_node + 1] += -P

    elif load_case == "point_top":
        top_node = right_nodes[np.argmax(coords[right_nodes, 1])]
        F[2*top_node + 1] += -P

    else:
        raise ValueError("load_case deve ser 'edge', 'point_mid' ou 'point_top'.")


# ============================================================
# SOLUÇÃO Q4
# ============================================================

def solve_q4_model(
    E=210e9, nu=0.3, L=1.0, h=0.2, t=0.01, P=1000.0,
    nx=5, ny=1, plane="stress", load_case="edge", mesh_name="Q4",
    integration="2x2"
):
    coords, quads = make_rectangular_q4_mesh(L=L, h=h, nx=nx, ny=ny)
    D = D_matrix(E, nu, plane)

    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    for quad in quads:
        xy = coords[quad, :]
        ke = q4_element_stiffness(xy, D, t, integration=integration)

        dofs = np.array([
            2*quad[0] + 0, 2*quad[0] + 1,
            2*quad[1] + 0, 2*quad[1] + 1,
            2*quad[2] + 0, 2*quad[2] + 1,
            2*quad[3] + 0, 2*quad[3] + 1,
        ], dtype=int)

        K[np.ix_(dofs, dofs)] += ke

    apply_q4_load(F, coords, L, h, t, P, load_case=load_case)

    # Engaste na borda esquerda: u = v = 0.
    left_nodes = np.where(np.isclose(coords[:, 0], 0.0, atol=1e-12))[0]
    constrained = []
    for n in left_nodes:
        constrained += [2*n + 0, 2*n + 1]

    U, reactions, constrained = solve_linear_system(K, F, constrained)

    # Deformações e tensões nos pontos de Gauss.
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]
    strain_gp = []
    stress_gp = []

    for quad in quads:
        xy = coords[quad, :]
        dofs = np.array([
            2*quad[0] + 0, 2*quad[0] + 1,
            2*quad[1] + 0, 2*quad[1] + 1,
            2*quad[2] + 0, 2*quad[2] + 1,
            2*quad[3] + 0, 2*quad[3] + 1,
        ], dtype=int)
        ue = U[dofs]

        elem_strain = []
        elem_stress = []
        for xi, eta in gauss_points:
            B, _ = q4_B_matrix(xy, xi, eta)
            eps = B @ ue
            sig = D @ eps
            elem_strain.append(eps)
            elem_stress.append(sig)

        strain_gp.append(np.array(elem_strain))
        stress_gp.append(np.array(elem_stress))

    right_nodes = np.where(np.isclose(coords[:, 0], L, atol=1e-12))[0]
    mid_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1]))]

    v_right_mid = U[2*mid_node + 1]
    v_right_avg = np.mean(U[2*right_nodes + 1])

    return Q4Result(
        coords=coords,
        quads=quads,
        U=U,
        K=K,
        F=F,
        reactions=reactions,
        constrained=constrained,
        strain_gp=strain_gp,
        stress_gp=stress_gp,
        v_right_mid=v_right_mid,
        v_right_avg=v_right_avg,
        mesh_name=mesh_name,
        load_case=load_case
    )


# ============================================================
# PÓS-PROCESSAMENTO
# ============================================================

def print_comparison(beam, q4_results):
    print("\n================ COMPARAÇÃO DE DESLOCAMENTOS ================\n")
    print(f"Viga - deslocamento vertical na ponta pelo MEF : {beam.v_tip:.6e} m")
    print(f"Viga - solução Euler-Bernoulli exata           : {beam.v_tip_analytical:.6e} m")
    print()

    print("Q4 - deslocamento vertical na borda direita")
    print("Malha      carga        v(meio da borda) [m]     v(médio da borda) [m]     razão v_Q4/v_viga")
    for r in q4_results:
        print(f"{r.mesh_name:10s} {r.load_case:10s} {r.v_right_mid: .6e}          {r.v_right_avg: .6e}          {r.v_right_mid/beam.v_tip: .6f}")


def plot_beam_deformed(beam, output_path, scale=1.0):
    x = beam.x
    v = beam.U[1::3]

    plt.figure(figsize=(9, 3))
    plt.plot(x, np.zeros_like(x), "k--", linewidth=1.0, label="configuração inicial")
    plt.plot(x, scale*v, "o-", label=f"deformada x {scale:g}")
    plt.xlabel("x [m]")
    plt.ylabel("v [m] escalado")
    plt.title("Modelo de viga")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_q4_mesh_and_deformed(result, output_path, scale=1.0):
    coords = result.coords
    U = result.U
    def_coords = coords + scale * U.reshape((-1, 2))

    plt.figure(figsize=(9, 3))

    for quad in result.quads:
        xy = coords[np.r_[quad, quad[0]], :]
        plt.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.7)

    for quad in result.quads:
        xy = def_coords[np.r_[quad, quad[0]], :]
        plt.plot(xy[:, 0], xy[:, 1], "r-", linewidth=0.8)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{result.mesh_name} - Q4 - carga {result.load_case} - deformada x {scale:g}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_q4_sigma_x(result, output_path):
    """
    Plota sigma_x média dos pontos de Gauss de cada elemento no centróide.
    """
    centroids = result.coords[result.quads].mean(axis=1)
    sigma_x = np.array([np.mean(elem[:, 0]) for elem in result.stress_gp])

    plt.figure(figsize=(9, 3))
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=sigma_x, s=40)
    plt.colorbar(sc, label=r"$\sigma_x$ [Pa]")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Tensão média $\\sigma_x$ por elemento - {result.mesh_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_q4_convergence(beam, q4_results, output_path_ratio, output_path_error):
    """
    Constrói curvas de convergência para o elemento Q4.

    Eixo x: número de elementos Q4.
    Eixo y 1: razão entre o deslocamento Q4 e o deslocamento de referência.
    Eixo y 2: erro relativo percentual em relação ao modelo de viga.
    """
    n_elems = np.array([len(r.quads) for r in q4_results], dtype=float)
    v_q4 = np.array([r.v_right_mid for r in q4_results], dtype=float)
    v_ref = beam.v_tip

    ratio = np.abs(v_q4) / abs(v_ref)
    error_percent = np.abs(v_q4 - v_ref) / abs(v_ref) * 100.0
    labels = [r.mesh_name for r in q4_results]

    # Curva 1: deslocamento normalizado
    plt.figure(figsize=(8, 5))
    plt.plot(n_elems, ratio, "o-", linewidth=2.0, label="|v_Q4| / |v_ref|")
    plt.axhline(1.0, linestyle="--", linewidth=1.2, label="referência")

    for x, y, label in zip(n_elems, ratio, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.xlabel("Número de elementos Q4")
    plt.ylabel("|v_Q4| / |v_ref|")
    plt.title("Curva de convergência do deslocamento - Q4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path_ratio, dpi=300, bbox_inches="tight")

    # Curva 2: erro relativo
    plt.figure(figsize=(8, 5))
    plt.plot(n_elems, error_percent, "o-", linewidth=2.0, label="erro relativo")

    for x, y, label in zip(n_elems, error_percent, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.xlabel("Número de elementos Q4")
    plt.ylabel("Erro relativo [%]")
    plt.title("Erro relativo do deslocamento - Q4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path_error, dpi=300, bbox_inches="tight")

    print("")
    print("Dados da curva de convergência:")
    print("Malha      N_elem     v_Q4 [m]           |v_Q4|/|v_ref|     erro [%]")
    for label, ne, v, r, err in zip(labels, n_elems, v_q4, ratio, error_percent):
        print(f"{label:10s} {int(ne):6d}   {v: .6e}       {r: .6f}        {err: .3f}")


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    outputs_dir = Path("outputs_q4")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Dados do problema
    # ------------------------------------------------------------
    E = 210e9      # Pa
    nu = 0.30
    h = 0.20       # m
    L = 5*h        # m, relação comprimento/altura = 5:1
    t = 0.01       # m, espessura fora do plano
    P = 1000.0     # N, carga total para baixo

    # Opções:
    #   'edge'      -> carga total P distribuída na borda direita
    #   'point_mid' -> carga concentrada no meio da borda direita
    #   'point_top' -> carga concentrada no canto superior direito
    load_case = "edge"

    # Integração completa padrão do Q4.
    integration = "2x2"

    # ------------------------------------------------------------
    # Modelo (a): viga
    # ------------------------------------------------------------
    beam = solve_beam_model(E=E, L=L, h=h, t=t, P=P, n_elem=20)

    # ------------------------------------------------------------
    # Modelos Q4 das figuras (b), (c) e (d)
    # ------------------------------------------------------------
    q4_b = solve_q4_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                          nx=5, ny=1, plane="stress", load_case=load_case,
                          mesh_name="malha_b", integration=integration)

    q4_c = solve_q4_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                          nx=10, ny=2, plane="stress", load_case=load_case,
                          mesh_name="malha_c", integration=integration)

    q4_d = solve_q4_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                          nx=20, ny=4, plane="stress", load_case=load_case,
                          mesh_name="malha_d", integration=integration)

    q4_results = [q4_b, q4_c, q4_d]
    print_comparison(beam, q4_results)

    plot_q4_convergence(
        beam,
        q4_results,
        outputs_dir / "curva_convergencia_q4_deslocamento.png",
        outputs_dir / "curva_convergencia_q4_erro.png"
    )

    # Escala automática apenas para visualização.
    max_abs_v = max(abs(beam.v_tip), *(abs(r.v_right_mid) for r in q4_results))
    scale = 0.15*h / max_abs_v

    plot_beam_deformed(beam, outputs_dir / "modelo_a_viga_deformada.png", scale=scale)

    for r in q4_results:
        plot_q4_mesh_and_deformed(r, outputs_dir / f"{r.mesh_name}_q4_deformada.png", scale=scale)
        plot_q4_sigma_x(r, outputs_dir / f"{r.mesh_name}_q4_sigma_x.png")

    print(f"\nFiguras salvas em: {outputs_dir.resolve()}\n")

    plt.show()


if __name__ == "__main__":
    main()
