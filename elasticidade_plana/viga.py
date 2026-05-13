import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# MODELOS DE MEF PARA VIGA EM BALANÇO:
# (a) Elemento de viga/pórtico plano Euler-Bernoulli
# (b), (c), (d) Elementos triangulares CST
#
# Autor: gerado para estudo de Elementos Finitos
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
class CSTResult:
    coords: np.ndarray
    tris: np.ndarray
    U: np.ndarray
    K: np.ndarray
    F: np.ndarray
    reactions: np.ndarray
    constrained: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    v_right_mid: float
    v_right_avg: float
    pattern: str
    load_case: str


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

    Graus de liberdade locais:
    [u1, v1, theta1, u2, v2, theta2]

    Como o elemento é horizontal, a matriz local já coincide com a global.
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

    # Carga concentrada na ponta livre: direção y negativa.
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
# MODELOS CST: (b), (c), (d)
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


def cst_B_matrix(xy):
    """
    Matriz B do CST para um triângulo de 3 nós.

    xy: array 3x2 com coordenadas:
        [[x1,y1],
         [x2,y2],
         [x3,y3]]

    Retorna:
        B    -> matriz deformação-deslocamento, 3x6
        Atri -> área positiva do triângulo
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    twoA = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    Atri = 0.5 * twoA

    if Atri <= 0:
        raise ValueError("Triângulo com área não positiva. Verifique a ordem dos nós.")

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = 1.0/(2.0*Atri) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3]
    ], dtype=float)

    return B, Atri


def make_rectangular_cst_mesh(L=1.0, h=0.2, nx=20, ny=4, pattern="alternating"):
    """
    Gera uma malha retangular L x h com elementos CST.

    Coordenadas:
        0 <= x <= L
        -h/2 <= y <= h/2

    pattern:
        'symmetric'   -> malha tipo (b): metade inferior com diagonais '/'
                         e metade superior com diagonais '\'
        'alternating' -> alterna diagonais em tabuleiro, mantido apenas para teste
        'backslash'  -> todas as diagonais no sentido '\', representando a malha (c)
        'slash'      -> todas as diagonais no sentido '/', representando a malha (d)
    """
    xvals = np.linspace(0.0, L, nx + 1)
    yvals = np.linspace(-h/2.0, h/2.0, ny + 1)

    coords = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords.append([xvals[i], yvals[j]])
    coords = np.array(coords, dtype=float)

    def node(i, j):
        return j*(nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00 = node(i,   j)     # inferior esquerdo
            n10 = node(i+1, j)     # inferior direito
            n01 = node(i,   j+1)   # superior esquerdo
            n11 = node(i+1, j+1)   # superior direito

            if pattern == "symmetric":
                # Malha tipo (b) da figura do enunciado:
                # abaixo da linha média, usa diagonal '/'; acima, usa '\'.
                y_center = 0.25 * (coords[n00, 1] + coords[n10, 1] + coords[n01, 1] + coords[n11, 1])
                diag = "slash" if y_center < 0.0 else "backslash"
            elif pattern == "alternating":
                # Malha em tabuleiro, mantida apenas como opção de teste.
                diag = "slash" if (i + j) % 2 == 0 else "backslash"
            else:
                diag = pattern

            if diag == "slash":
                # diagonal n00 -> n11
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])

            elif diag == "backslash":
                # diagonal n01 -> n10
                tris.append([n00, n10, n01])
                tris.append([n10, n11, n01])

            else:
                raise ValueError("pattern deve ser 'symmetric', 'alternating', 'slash' ou 'backslash'.")

    return coords, np.array(tris, dtype=int)


def apply_cst_load(F, coords, L, h, t, P, load_case="edge"):
    """
    Aplica carregamento no modelo CST.

    P > 0 significa carga vertical total para baixo.

    load_case:
        'edge'      -> força total P distribuída na borda direita.
                       É a forma mais adequada para comparar com a viga,
                       pois a resultante passa pelo centro da seção.
        'point_mid' -> força P concentrada no nó do meio da borda direita.
        'point_top' -> força P concentrada no canto superior direito.
                       Útil quando o enunciado quer uma força no canto.
    """
    tol = 1e-12
    right_nodes = np.where(np.isclose(coords[:, 0], L, atol=tol))[0]
    right_nodes = right_nodes[np.argsort(coords[right_nodes, 1])]

    if load_case == "edge":
        # tração vertical uniforme ty tal que ty*h*t = -P
        ty = -P / (h * t)

        for a, b in zip(right_nodes[:-1], right_nodes[1:]):
            ya, yb = coords[a, 1], coords[b, 1]
            edge_length = abs(yb - ya)

            # Vetor nodal consistente de um elemento de linha linear:
            # f_a = f_b = ty*t*Le/2, na direção y.
            F[2*a + 1] += ty * t * edge_length / 2.0
            F[2*b + 1] += ty * t * edge_length / 2.0

    elif load_case == "point_mid":
        mid_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1]))]
        F[2*mid_node + 1] += -P

    elif load_case == "point_top":
        top_node = right_nodes[np.argmax(coords[right_nodes, 1])]
        F[2*top_node + 1] += -P

    else:
        raise ValueError("load_case deve ser 'edge', 'point_mid' ou 'point_top'.")


def solve_cst_model(
    E=210e9, nu=0.3, L=1.0, h=0.2, t=0.01, P=1000.0,
    nx=20, ny=4, pattern="alternating", plane="stress", load_case="edge"
):
    """
    Modelos (b), (c) e (d): domínio 2D discretizado por CST.
    """
    coords, tris = make_rectangular_cst_mesh(L, h, nx, ny, pattern)
    D = D_matrix(E, nu, plane)

    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # Montagem da matriz global.
    for tri in tris:
        xy = coords[tri, :]
        B, Atri = cst_B_matrix(xy)
        ke = t * Atri * (B.T @ D @ B)

        dofs = np.array([
            2*tri[0] + 0, 2*tri[0] + 1,
            2*tri[1] + 0, 2*tri[1] + 1,
            2*tri[2] + 0, 2*tri[2] + 1
        ])

        K[np.ix_(dofs, dofs)] += ke

    # Carga na extremidade direita.
    apply_cst_load(F, coords, L, h, t, P, load_case=load_case)

    # Engaste na borda esquerda: u = v = 0 para todos os nós com x=0.
    left_nodes = np.where(np.isclose(coords[:, 0], 0.0, atol=1e-12))[0]
    constrained = []
    for n in left_nodes:
        constrained += [2*n + 0, 2*n + 1]

    U, reactions, constrained = solve_linear_system(K, F, constrained)

    # Deformações e tensões constantes por elemento.
    strain = np.zeros((len(tris), 3))
    stress = np.zeros((len(tris), 3))
    for e, tri in enumerate(tris):
        B, _ = cst_B_matrix(coords[tri, :])
        dofs = np.array([
            2*tri[0] + 0, 2*tri[0] + 1,
            2*tri[1] + 0, 2*tri[1] + 1,
            2*tri[2] + 0, 2*tri[2] + 1
        ])
        strain[e, :] = B @ U[dofs]
        stress[e, :] = D @ strain[e, :]

    # Deslocamento na extremidade direita:
    # 1) nó do meio da borda direita;
    # 2) média dos deslocamentos verticais da borda direita.
    right_nodes = np.where(np.isclose(coords[:, 0], L, atol=1e-12))[0]
    mid_node = right_nodes[np.argmin(np.abs(coords[right_nodes, 1]))]

    v_right_mid = U[2*mid_node + 1]
    v_right_avg = np.mean(U[2*right_nodes + 1])

    return CSTResult(
        coords=coords,
        tris=tris,
        U=U,
        K=K,
        F=F,
        reactions=reactions,
        constrained=constrained,
        strain=strain,
        stress=stress,
        v_right_mid=v_right_mid,
        v_right_avg=v_right_avg,
        pattern=pattern,
        load_case=load_case
    )


# ============================================================
# PÓS-PROCESSAMENTO
# ============================================================

def print_comparison(beam, cst_results):
    print("\n================ COMPARAÇÃO DE DESLOCAMENTOS ================\n")
    print(f"Viga - flecha na ponta pelo MEF      : {beam.v_tip:.6e} m")
    print(f"Viga - solução Euler-Bernoulli exata : {beam.v_tip_analytical:.6e} m")
    print()

    print("CST - deslocamento vertical na borda direita")
    print("Malha/padrão     carga        v(meio da borda) [m]     v(médio da borda) [m]")
    for r in cst_results:
        print(f"{r.pattern:14s} {r.load_case:10s} {r.v_right_mid: .6e}          {r.v_right_avg: .6e}")

    print("\nRazão em relação à flecha da viga:")
    for r in cst_results:
        print(f"{r.pattern:14s} {r.v_right_mid/beam.v_tip: .6f}")


def plot_beam_deformed(beam, scale=1.0):
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


def plot_cst_mesh_and_deformed(result, scale=1.0):
    coords = result.coords
    U = result.U
    def_coords = coords + scale * U.reshape((-1, 2))

    plt.figure(figsize=(9, 3))

    # Malha original
    for tri in result.tris:
        xy = coords[np.r_[tri, tri[0]], :]
        plt.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.5)

    # Malha deformada
    for tri in result.tris:
        xy = def_coords[np.r_[tri, tri[0]], :]
        plt.plot(xy[:, 0], xy[:, 1], "r-", linewidth=0.6)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"CST - {result.pattern} - carga {result.load_case} - deformada x {scale:g}")
    plt.grid(True)
    plt.tight_layout()


def plot_cst_sigma_x(result):
    """
    Plota sigma_x constante por elemento no centróide.
    É um pós-processamento simples, sem suavização nodal.
    """
    coords = result.coords
    centroids = coords[result.tris].mean(axis=1)
    sigma_x = result.stress[:, 0]

    plt.figure(figsize=(9, 3))
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=sigma_x, s=30)
    plt.colorbar(sc, label=r"$\sigma_x$ [Pa]")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Tensão $\\sigma_x$ por elemento - CST {result.pattern}")
    plt.tight_layout()


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    # ------------------------------------------------------------
    # Pasta de saída das figuras
    # ------------------------------------------------------------
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Dados do problema
    # ------------------------------------------------------------
    E = 210e9      # Pa
    nu = 0.30
    h = 0.20       # m
    L = 5*h        # m, razão comprimento/altura = 5:1
    t = 0.01       # m, espessura fora do plano
    P = 1000.0     # N, carga total para baixo

    # Refinamento da malha.
    # Altere nx e ny para estudar convergência.
    nx = 20        # divisões no comprimento
    ny = 4         # divisões na altura

    # Use:
    # 'edge'      -> mais adequado para comparar com a viga
    # 'point_mid' -> força concentrada no meio da borda direita
    # 'point_top' -> força concentrada no canto superior direito
    load_case = "edge"

    # ------------------------------------------------------------
    # Modelo (a): viga
    # ------------------------------------------------------------
    beam = solve_beam_model(E=E, L=L, h=h, t=t, P=P, n_elem=nx)

    # ------------------------------------------------------------
    # Modelos CST: (b), (c), (d)
    # ------------------------------------------------------------
    cst_b = solve_cst_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                            nx=nx, ny=ny, pattern="symmetric",
                            plane="stress", load_case=load_case)

    cst_c = solve_cst_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                            nx=nx, ny=ny, pattern="backslash",
                            plane="stress", load_case=load_case)

    cst_d = solve_cst_model(E=E, nu=nu, L=L, h=h, t=t, P=P,
                            nx=nx, ny=ny, pattern="slash",
                            plane="stress", load_case=load_case)

    print_comparison(beam, [cst_b, cst_c, cst_d])

    # Escala automática apenas para visualização.
    max_abs_v = max(
        abs(beam.v_tip),
        abs(cst_b.v_right_mid),
        abs(cst_c.v_right_mid),
        abs(cst_d.v_right_mid)
    )
    scale = 0.15*h / max_abs_v

    plot_beam_deformed(beam, scale=scale)
    plt.savefig(outputs_dir / "modelo_a_viga_deformada.png", dpi=300, bbox_inches="tight")

    plot_cst_mesh_and_deformed(cst_b, scale=scale)
    plt.savefig(outputs_dir / "modelo_b_cst_symmetric_deformada.png", dpi=300, bbox_inches="tight")

    plot_cst_mesh_and_deformed(cst_c, scale=scale)
    plt.savefig(outputs_dir / "modelo_c_cst_backslash_deformada.png", dpi=300, bbox_inches="tight")

    plot_cst_mesh_and_deformed(cst_d, scale=scale)
    plt.savefig(outputs_dir / "modelo_d_cst_slash_deformada.png", dpi=300, bbox_inches="tight")

    # Exemplo de tensão sigma_x por elemento para a malha (b).
    plot_cst_sigma_x(cst_b)
    plt.savefig(outputs_dir / "modelo_b_cst_sigma_x.png", dpi=300, bbox_inches="tight")

    print(f"Figuras salvas em: {outputs_dir.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
