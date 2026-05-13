from pathlib import Path
import shutil
import zipfile

base_dir = Path("/mnt/data/convergencia_viga2D_ticks")
base_dir.mkdir(exist_ok=True)

script_path = base_dir / "convergencia_viga2D_ticks_corrigido.py"

script_code = r'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# ============================================================
# CURVAS DE CONVERGÊNCIA - ELEMENTO DE VIGA 2D EULER-BERNOULLI
# Sistema SI: N, m, Pa
# ============================================================

OUT = Path("outputs_convergencia_viga2D")
OUT.mkdir(exist_ok=True)

# ============================================================
# DADOS DO PROBLEMA
# ============================================================

E = 205e9        # Pa
I = 8.0e-6       # m^4
A = 0.00146373   # m^2
L_total = 1.0    # m

P = 10000.0      # N
q0 = 10000.0     # N/m

# Malhas utilizadas nas curvas de convergência
malhas = np.array([1, 2, 4, 8, 16, 32], dtype=int)

# ============================================================
# GRAUS DE LIBERDADE
# ============================================================

DOF_U = 0
DOF_V = 1
DOF_T = 2
NGLN = 3


# ============================================================
# FUNÇÕES DO ELEMENTO DE VIGA 2D
# ============================================================

def k_local_frame(E, A, I, L):
    """Matriz de rigidez local do elemento de viga/pórtico 2D Euler-Bernoulli.

    Ordem dos GDL locais:
    [u1, v1, theta1, u2, v2, theta2]
    """
    EA_L = E * A / L
    EI_L3 = E * I / L**3

    return np.array([
        [ EA_L,          0.0,          0.0, -EA_L,          0.0,          0.0],
        [ 0.0,     12*EI_L3,  6*L*EI_L3,  0.0,    -12*EI_L3,  6*L*EI_L3],
        [ 0.0,   6*L*EI_L3, 4*L**2*EI_L3, 0.0, -6*L*EI_L3, 2*L**2*EI_L3],
        [-EA_L,          0.0,          0.0,  EA_L,          0.0,          0.0],
        [ 0.0,    -12*EI_L3, -6*L*EI_L3,  0.0,     12*EI_L3, -6*L*EI_L3],
        [ 0.0,   6*L*EI_L3, 2*L**2*EI_L3, 0.0, -6*L*EI_L3, 4*L**2*EI_L3]
    ], dtype=float)


def T_matrix(c, s):
    """Matriz de transformação: u_local = T u_global."""
    return np.array([
        [ c,  s, 0.0, 0.0, 0.0, 0.0],
        [-s,  c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  s, 0.0],
        [0.0, 0.0, 0.0, -s,  c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=float)


def dofs_element(n1, n2):
    """Retorna os GDL globais de um elemento de dois nós."""
    return np.array([
        n1*NGLN + DOF_U,
        n1*NGLN + DOF_V,
        n1*NGLN + DOF_T,
        n2*NGLN + DOF_U,
        n2*NGLN + DOF_V,
        n2*NGLN + DOF_T
    ], dtype=int)


def hermite_N(x, L):
    """Funções de forma de Hermite para deslocamento transversal."""
    xi = x / L

    N1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    N2 = L * (xi - 2.0*xi**2 + xi**3)
    N3 = 3.0*xi**2 - 2.0*xi**3
    N4 = L * (-xi**2 + xi**3)

    return N1, N2, N3, N4


# ============================================================
# CARGAS NODAIS EQUIVALENTES CONSISTENTES
# ============================================================

def equiv_point_load_local(P_local_v, a, Le):
    """Carga nodal equivalente para força concentrada transversal dentro do elemento.

    P_local_v: força no eixo local v.
    a: posição da força medida a partir do nó inicial.
    Le: comprimento do elemento.
    """
    N1, N2, N3, N4 = hermite_N(a, Le)

    f = np.zeros(6)
    f[1] = P_local_v * N1
    f[2] = P_local_v * N2
    f[4] = P_local_v * N3
    f[5] = P_local_v * N4

    return f


def equiv_udl_local(q, Le):
    """Carga nodal equivalente para carga distribuída uniforme no eixo local v.

    q negativo representa carga para baixo no sistema local da viga horizontal.
    """
    f = np.zeros(6)
    f[1] = q * Le / 2.0
    f[2] = q * Le**2 / 12.0
    f[4] = q * Le / 2.0
    f[5] = -q * Le**2 / 12.0

    return f


def equiv_linear_load_local(qi, qj, Le):
    """Carga nodal equivalente para carga distribuída linear no eixo local v.

    qi: intensidade no nó inicial do elemento.
    qj: intensidade no nó final do elemento.
    """
    f = np.zeros(6)

    f[1] = Le * (7.0*qi + 3.0*qj) / 20.0
    f[2] = Le**2 * (qi/20.0 + qj/30.0)
    f[4] = Le * (3.0*qi + 7.0*qj) / 20.0
    f[5] = -Le**2 * (qi/30.0 + qj/20.0)

    return f


# ============================================================
# MONTAGEM E SOLUÇÃO
# ============================================================

def assemble_beam(nel, case):
    """Monta e resolve uma viga horizontal com nel elementos."""
    nnos = nel + 1
    xcoords = np.linspace(0.0, L_total, nnos)
    coords = np.column_stack((xcoords, np.zeros(nnos)))

    K = np.zeros((nnos*NGLN, nnos*NGLN))
    F = np.zeros(nnos*NGLN)

    Le = L_total / nel

    for e in range(nel):
        n1 = e
        n2 = e + 1

        k_local = k_local_frame(E, A, I, Le)

        # A viga está horizontal, então c=1 e s=0.
        T = T_matrix(1.0, 0.0)

        k_global = T.T @ k_local @ T
        dofs = dofs_element(n1, n2)

        K[np.ix_(dofs, dofs)] += k_global

        # Cargas distribuídas consistentes
        if case == "balanco_uniforme":
            f_local = equiv_udl_local(-q0, Le)
            F[dofs] += T.T @ f_local

        elif case == "balanco_linear":
            # Carga triangular: q(x) = -q0*x/L_total
            # Zero no engaste e q0 na extremidade livre.
            x_i = xcoords[n1]
            x_j = xcoords[n2]

            qi = -q0 * x_i / L_total
            qj = -q0 * x_j / L_total

            f_local = equiv_linear_load_local(qi, qj, Le)
            F[dofs] += T.T @ f_local

    # Carga concentrada no meio da viga simplesmente apoiada
    if case == "simples_pontual":
        xmid = L_total / 2.0
        idx_mid = np.where(np.isclose(xcoords, xmid))[0]

        if idx_mid.size > 0:
            # O ponto médio é um nó da malha
            n = int(idx_mid[0])
            F[n*NGLN + DOF_V] += -P
        else:
            # O ponto médio está dentro de um elemento
            e = int(np.searchsorted(xcoords, xmid) - 1)
            e = max(0, min(e, nel - 1))

            n1 = e
            n2 = e + 1
            a = xmid - xcoords[n1]

            f_local = equiv_point_load_local(-P, a, Le)
            dofs = dofs_element(n1, n2)

            F[dofs] += T_matrix(1.0, 0.0).T @ f_local

    K_original = K.copy()
    F_original = F.copy()

    # Condições de contorno
    if case == "simples_pontual":
        # Viga simplesmente apoiada:
        # v(0)=0, v(L)=0.
        # u(0)=0 é imposto apenas para remover movimento de corpo rígido axial.
        restraints = [
            (0, DOF_U, 0.0),
            (0, DOF_V, 0.0),
            (nnos - 1, DOF_V, 0.0)
        ]
    else:
        # Viga em balanço:
        # u(0)=0, v(0)=0, theta(0)=0.
        restraints = [
            (0, DOF_U, 0.0),
            (0, DOF_V, 0.0),
            (0, DOF_T, 0.0)
        ]

    for node, dof, value in restraints:
        gdlf = node*NGLN + dof

        F -= K[:, gdlf] * value

        K[gdlf, :] = 0.0
        K[:, gdlf] = 0.0
        K[gdlf, gdlf] = 1.0
        F[gdlf] = value

    U = np.linalg.solve(K, F)
    R = K_original @ U - F_original

    return coords, U, R


def interp_v_at_x(coords, U, x_eval):
    """Interpola o deslocamento transversal v(x) usando Hermite."""
    xcoords = coords[:, 0]
    nel = len(xcoords) - 1

    if np.isclose(x_eval, xcoords[-1]):
        e = nel - 1
    else:
        e = int(np.searchsorted(xcoords, x_eval) - 1)
        e = max(0, min(e, nel - 1))

    x1 = xcoords[e]
    Le = xcoords[e + 1] - x1
    a = x_eval - x1

    v1 = U[e*NGLN + DOF_V]
    t1 = U[e*NGLN + DOF_T]
    v2 = U[(e + 1)*NGLN + DOF_V]
    t2 = U[(e + 1)*NGLN + DOF_T]

    N1, N2, N3, N4 = hermite_N(a, Le)

    return N1*v1 + N2*t1 + N3*v2 + N4*t2


# ============================================================
# EXECUÇÃO DOS CASOS
# ============================================================

def run_case(case):
    fe_vals = []
    exact_vals = []
    errors = []

    for nel in malhas:
        coords, U, R = assemble_beam(nel, case)

        if case == "simples_pontual":
            # Deflexão máxima no meio do vão
            val = interp_v_at_x(coords, U, L_total / 2.0)
            exact = -P * L_total**3 / (48.0 * E * I)

        elif case == "balanco_uniforme":
            # Deflexão na extremidade livre
            val = U[-NGLN + DOF_V]
            exact = -q0 * L_total**4 / (8.0 * E * I)

        elif case == "balanco_linear":
            # Carga triangular crescente:
            # q(x) = q0*x/L
            # Deflexão na extremidade livre:
            val = U[-NGLN + DOF_V]
            exact = -11.0 * q0 * L_total**4 / (120.0 * E * I)

        fe_vals.append(val)
        exact_vals.append(exact)

        erro = abs((val - exact) / exact) * 100.0
        errors.append(erro)

    return np.array(fe_vals), np.array(exact_vals), np.array(errors)


cases = {
    "simples_pontual": "Viga simplesmente apoiada - carga concentrada no meio",
    "balanco_uniforme": "Viga em balanço - carga distribuída uniforme",
    "balanco_linear": "Viga em balanço - carga distribuída linear"
}

all_results = {}

for key in cases:
    all_results[key] = run_case(key)


# ============================================================
# SAÍDA EM CSV E RELATÓRIO
# ============================================================

csv_path = OUT / "resultados_convergencia.csv"

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("caso,nel,valor_mef_m,valor_analitico_m,valor_mef_mm,valor_analitico_mm,erro_percentual\n")

    for key, (fe, ex, err) in all_results.items():
        for nel, fei, exi, eri in zip(malhas, fe, ex, err):
            f.write(
                f"{key},{nel},"
                f"{fei:.12e},{exi:.12e},"
                f"{fei*1e3:.12e},{exi*1e3:.12e},"
                f"{eri:.12e}\n"
            )


report_path = OUT / "relatorio_convergencia_viga2D.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - CURVAS DE CONVERGÊNCIA DO ELEMENTO DE VIGA 2D\n")
    f.write("="*72 + "\n\n")

    f.write("Sistema de unidades: SI, isto é, N, m e Pa.\n")
    f.write("Os deslocamentos também são apresentados em mm.\n\n")

    f.write("DADOS ADOTADOS\n")
    f.write(f"E  = {E:.6e} Pa\n")
    f.write(f"I  = {I:.6e} m^4\n")
    f.write(f"A  = {A:.6e} m^2\n")
    f.write(f"L  = {L_total:.6e} m\n")
    f.write(f"P  = {P:.6e} N\n")
    f.write(f"q0 = {q0:.6e} N/m\n\n")

    f.write("Malhas usadas: " + ", ".join(str(n) for n in malhas) + " elementos.\n\n")

    f.write("SOLUÇÕES ANALÍTICAS USADAS\n")
    f.write("a) Viga simplesmente apoiada com carga no meio:\n")
    f.write("   v_max = -P L^3/(48 E I)\n\n")

    f.write("b) Viga em balanço com carga distribuída uniforme:\n")
    f.write("   v_L = -q0 L^4/(8 E I)\n\n")

    f.write("c) Viga em balanço com carga distribuída linear q(x)=q0*x/L:\n")
    f.write("   v_L = -11 q0 L^4/(120 E I)\n\n")

    for key, title in cases.items():
        fe, ex, err = all_results[key]

        f.write(title + "\n")
        f.write("-"*72 + "\n")
        f.write("NEL       MEF [mm]        Analítico [mm]       Erro [%]\n")

        for nel, fei, exi, eri in zip(malhas, fe, ex, err):
            f.write(f"{nel:3d}  {fei*1e3:16.8f}  {exi*1e3:16.8f}  {eri:14.8e}\n")

        f.write("\n")


# ============================================================
# CONFIGURAÇÃO DOS TICKS DO EIXO X
# ============================================================

def configurar_ticks_malha(ax, malhas):
    """Mostra todos os números de elementos no eixo x dos gráficos."""
    ax.set_xscale("log")
    ax.set_xticks(malhas)
    ax.set_xticklabels([str(n) for n in malhas])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")


# ============================================================
# GRÁFICOS INDIVIDUAIS
# ============================================================

for key, title in cases.items():
    fe, ex, err = all_results[key]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Evita problema visual no log quando o erro é numericamente zero.
    err_plot = np.maximum(err, 1e-14)

    ax.loglog(malhas, err_plot, marker="o")
    configurar_ticks_malha(ax, malhas)

    ax.set_xlabel("Número de elementos")
    ax.set_ylabel("Erro relativo [%]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT / f"curva_erro_{key}.png", dpi=200)
    plt.close(fig)


# ============================================================
# GRÁFICO COMPARATIVO
# ============================================================

fig, ax = plt.subplots(figsize=(8.5, 5.2))

for key, title in cases.items():
    fe, ex, err = all_results[key]

    err_plot = np.maximum(err, 1e-14)

    ax.loglog(malhas, err_plot, marker="o", label=title)

configurar_ticks_malha(ax, malhas)

ax.set_xlabel("Número de elementos")
ax.set_ylabel("Erro relativo [%]")
ax.set_title("Curvas de convergência - elemento de viga 2D")
ax.grid(True, which="both", alpha=0.4)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "curvas_convergencia_comparativo.png", dpi=200)
plt.close(fig)


# ============================================================
# SAÍDA NO TERMINAL
# ============================================================

print("Resultados gerados em:", OUT.resolve())
print("\nResumo:")

for key, title in cases.items():
    fe, ex, err = all_results[key]

    print("\n" + title)
    print("NEL       MEF [mm]       Analítico [mm]      Erro [%]")

    for nel, fei, exi, eri in zip(malhas, fe, ex, err):
        print(f"{nel:3d}  {fei*1e3:14.8f}  {exi*1e3:14.8f}  {eri:12.6e}")
'''

script_path.write_text(script_code.strip() + "\n", encoding="utf-8")

# Run the generated script in its folder to create outputs
import subprocess, sys
subprocess.run([sys.executable, str(script_path)], cwd=base_dir, check=True)

zip_path = Path("/mnt/data/convergencia_viga2D_ticks_corrigido.zip")
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in base_dir.rglob("*"):
        z.write(p, p.relative_to(base_dir.parent))

print("Arquivos gerados:")
print(script_path)
print(base_dir / "outputs_convergencia_viga2D")
print(zip_path)
