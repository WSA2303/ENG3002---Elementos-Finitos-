from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROGRAMA DE ELEMENTOS FINITOS PARA VIGAS/PÓRTICOS 2D
# Conversão do script MATLAB viga2D.m para Python
# ============================================================
# Graus de liberdade por nó:
# 0 -> deslocamento em x
# 1 -> deslocamento em y
# 2 -> rotação theta_z
# ============================================================


def read_numeric_rows(filename: str | Path) -> list[list[float]]:
    """Lê linhas numéricas de um arquivo de entrada com cabeçalhos de texto.

    O arquivo original possui linhas como:
        2, 3, 7, 8, 2, 3, 1000
        1,4, 210e9, 5.5E-3, ...

    Esta função ignora linhas sem números e retorna apenas as linhas numéricas.
    """
    path = Path(filename)
    rows: list[list[float]] = []

    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    with path.open("r", encoding="latin-1") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped[0] not in "+-.0123456789":
                continue
            values = number_pattern.findall(line)
            if values:
                rows.append([float(v) for v in values])

    return rows


def read_input(filename: str | Path):
    """Lê os dados do problema no formato do arquivo Viga2D.txt."""
    rows = read_numeric_rows(filename)

    # Primeira linha numérica: NNOE, NGLN, NNOS, NEL, NFORC, NRESTR, ESC
    prop = rows[0]
    NNOE = int(prop[0])
    NGLN = int(prop[1])
    NNOS = int(prop[2])
    NEL = int(prop[3])
    NFORC = int(prop[4])
    NRESTR = int(prop[5])
    ESC = float(prop[6])

    idx = 1

    # Coordenadas: NNOS linhas com x, y
    coordenadas = np.array(rows[idx:idx + NNOS], dtype=float)
    idx += NNOS

    # Conectividade e propriedades: NEL linhas com no1, no2, E, A, I, h
    conect_prop = np.array(rows[idx:idx + NEL], dtype=float)
    idx += NEL

    conect = conect_prop[:, 0:2].astype(int) - 1  # converte de 1-based para 0-based
    E = conect_prop[:, 2]
    A = conect_prop[:, 3]
    I = conect_prop[:, 4]
    h = conect_prop[:, 5]

    # Forças: nó, grau de liberdade, valor
    forcas = np.array(rows[idx:idx + NFORC], dtype=float)
    idx += NFORC
    forcas[:, 0] -= 1  # nó: 1-based -> 0-based
    # O grau de liberdade já está no padrão 0, 1, 2

    # Restrições: nó, grau de liberdade
    restricoes = np.array(rows[idx:idx + NRESTR], dtype=float)
    restricoes[:, 0] -= 1  # nó: 1-based -> 0-based

    return NNOE, NGLN, NNOS, NEL, NFORC, NRESTR, ESC, coordenadas, conect, E, A, I, h, forcas, restricoes


def frame_element_stiffness_global(E: float, A: float, I: float, L: float, c: float, s: float) -> np.ndarray:
    """Matriz de rigidez 6x6 de elemento de pórtico/viga 2D em coordenadas globais."""
    EA_L = E * A / L
    EI = E * I

    k = np.zeros((6, 6), dtype=float)

    k[0, 0] = EA_L * c**2 + 12 * EI / L**3 * s**2
    k[0, 1] = (EA_L - 12 * EI / L**3) * s * c
    k[0, 2] = 6 * EI / L**2 * s
    k[0, 3] = -EA_L * c**2 - 12 * EI / L**3 * s**2
    k[0, 4] = (-EA_L + 12 * EI / L**3) * s * c
    k[0, 5] = -6 * EI / L**2 * s

    k[1, 1] = EA_L * s**2 + 12 * EI / L**3 * c**2
    k[1, 2] = 6 * EI / L**2 * c
    k[1, 3] = -k[0, 1]
    k[1, 4] = -k[1, 1]
    k[1, 5] = k[1, 2]

    k[2, 2] = 4 * EI / L
    k[2, 3] = k[0, 2]
    k[2, 4] = -k[1, 2]
    k[2, 5] = 2 * EI / L

    k[3, 3] = k[0, 0]
    k[3, 4] = k[0, 1]
    k[3, 5] = k[0, 2]

    k[4, 4] = k[1, 1]
    k[4, 5] = -k[1, 2]

    k[5, 5] = k[2, 2]

    # Completa por simetria
    i_lower = np.tril_indices(6, -1)
    k[i_lower] = k.T[i_lower]

    return k


def frame_element_stiffness_local(E: float, A: float, I: float, L: float) -> np.ndarray:
    """Matriz de rigidez 6x6 do elemento em coordenadas locais."""
    EA_L = E * A / L
    EI = E * I

    k = np.array([
        [ EA_L,          0,           0, -EA_L,          0,           0],
        [    0,  12*EI/L**3,  6*EI/L**2,     0, -12*EI/L**3,  6*EI/L**2],
        [    0,   6*EI/L**2,    4*EI/L,     0,  -6*EI/L**2,    2*EI/L],
        [-EA_L,          0,           0,  EA_L,          0,           0],
        [    0, -12*EI/L**3, -6*EI/L**2,     0,  12*EI/L**3, -6*EI/L**2],
        [    0,   6*EI/L**2,    2*EI/L,     0,  -6*EI/L**2,    4*EI/L],
    ], dtype=float)

    return k


def element_dof_indices(conect_element: np.ndarray, ngln: int = 3) -> list[int]:
    """Retorna os índices globais dos 6 GDL de um elemento de 2 nós."""
    dofs: list[int] = []
    for node in conect_element:
        base = int(node) * ngln
        dofs.extend([base, base + 1, base + 2])
    return dofs


def calcular_escala_automatica(
    coordenadas: np.ndarray,
    desloc: np.ndarray,
    ngln: int = 3,
    fracao_tamanho: float = 0.15,
    escala_min: float = 1.0,
    escala_max: float | None = 1.0e6,
) -> float:
    """Calcula automaticamente o fator de ampliação da deformada.

    A ideia é fazer o maior deslocamento translacional plotado ficar igual a
    uma fração do tamanho característico da estrutura.

    Parâmetros
    ----------
    coordenadas:
        Coordenadas nodais indeformadas.
    desloc:
        Vetor global de deslocamentos. Para pórtico 2D: [ux, uy, theta] por nó.
    ngln:
        Número de graus de liberdade por nó.
    fracao_tamanho:
        Fração do tamanho da estrutura que será usada para representar o maior
        deslocamento amplificado. Exemplo: 0.15 significa 15%.
    escala_min:
        Menor fator permitido. Use 1.0 para não reduzir deformações grandes.
    escala_max:
        Maior fator permitido. Use None para não impor limite superior.
    """

    # Deslocamentos translacionais dos nós: ux e uy.
    desloc_xy = np.column_stack((desloc[0::ngln], desloc[1::ngln]))

    # Maior deslocamento translacional nodal em módulo.
    desloc_max = np.max(np.linalg.norm(desloc_xy, axis=1))

    # Tamanho característico da estrutura: diagonal da caixa envolvente.
    dx_modelo = np.ptp(coordenadas[:, 0])
    dy_modelo = np.ptp(coordenadas[:, 1])
    tamanho_modelo = np.hypot(dx_modelo, dy_modelo)

    if desloc_max <= 0.0 or tamanho_modelo <= 0.0:
        return 1.0

    escala = fracao_tamanho * tamanho_modelo / desloc_max

    if escala_min is not None:
        escala = max(escala, escala_min)

    if escala_max is not None:
        escala = min(escala, escala_max)

    return float(escala)


def solve_frame(
    input_file: str | Path = "Viga2D.txt",
    show_plots: bool = True,
    output_dir: str | Path | None = None,
    usar_escala_automatica: bool = True,
    fracao_deformada: float = 0.15,
):
    (
        NNOE, NGLN, NNOS, NEL, NFORC, NRESTR, ESC,
        coordenadas, conect, E, A, I, h, forcas, restricoes,
    ) = read_input(input_file)

    total_dofs = NNOS * NGLN
    Kglobal = np.zeros((total_dofs, total_dofs), dtype=float)
    Fglobal = np.zeros(total_dofs, dtype=float)

    # Ângulo e comprimento dos elementos
    angles = np.zeros(NEL)
    lengths = np.zeros(NEL)

    for e in range(NEL):
        n1, n2 = conect[e]
        dx = coordenadas[n2, 0] - coordenadas[n1, 0]
        dy = coordenadas[n2, 1] - coordenadas[n1, 1]
        angles[e] = np.arctan2(dy, dx)
        lengths[e] = np.hypot(dx, dy)

    # Montagem da matriz de rigidez global
    for e in range(NEL):
        L = lengths[e]
        c = np.cos(angles[e])
        s = np.sin(angles[e])
        ke = frame_element_stiffness_global(E[e], A[e], I[e], L, c, s)
        dofs = element_dof_indices(conect[e], NGLN)

        for a, A_global in enumerate(dofs):
            for b, B_global in enumerate(dofs):
                Kglobal[A_global, B_global] += ke[a, b]

    # Vetor de forças global
    for node, dof, value in forcas:
        idx = int(node) * NGLN + int(dof)
        Fglobal[idx] += value

    # Guarda matriz e vetor originais para calcular reações depois
    Koriginal = Kglobal.copy()
    Foriginal = Fglobal.copy()

    # Aplicação das restrições pelo método direto
    for node, dof in restricoes:
        idx = int(node) * NGLN + int(dof)
        Kglobal[idx, :] = 0.0
        Kglobal[:, idx] = 0.0
        Kglobal[idx, idx] = 1.0
        Fglobal[idx] = 0.0

    # Solução do sistema K u = F
    desloc = np.linalg.solve(Kglobal, Fglobal)
    desloc_mm = desloc * 1e3

    # Reações nodais
    reacoes = Koriginal @ desloc - Foriginal

    # Fator de ampliação da deformada
    # O valor ESC lido do arquivo continua disponível como escala manual.
    # Se usar_escala_automatica=True, ele é substituído por um valor calculado
    # a partir do tamanho da estrutura e do maior deslocamento nodal.
    ESC_manual = ESC

    if usar_escala_automatica:
        ESC = calcular_escala_automatica(
            coordenadas,
            desloc,
            ngln=NGLN,
            fracao_tamanho=fracao_deformada,
        )

    # Coordenadas deformadas
    coordeform = coordenadas.copy()
    coordeform[:, 0] += ESC * desloc[0::3]
    coordeform[:, 1] += ESC * desloc[1::3]

    # Esforços locais nos elementos
    FF = np.zeros((NEL, NNOE * NGLN), dtype=float)
    sigma_N = np.zeros(NEL)
    sigma_M = np.zeros(NEL)
    sigma_xx = np.zeros(NEL)

    for e in range(NEL):
        L = lengths[e]
        c = np.cos(angles[e])
        s = np.sin(angles[e])
        k_local = frame_element_stiffness_local(E[e], A[e], I[e], L)

        n1, n2 = conect[e]
        u_global_e = desloc[element_dof_indices(conect[e], NGLN)]

        # Transformação dos deslocamentos globais para locais
        # [u_local, v_local, theta] por nó
        u_local = np.array([
            u_global_e[0] * c + u_global_e[1] * s,
           -u_global_e[0] * s + u_global_e[1] * c,
            u_global_e[2],
            u_global_e[3] * c + u_global_e[4] * s,
           -u_global_e[3] * s + u_global_e[4] * c,
            u_global_e[5],
        ])

        FF[e, :] = k_local @ u_local

        sigma_N[e] = FF[e, 0] / A[e]
        momento_critico = FF[e, 2] if abs(FF[e, 2]) >= abs(FF[e, 5]) else FF[e, 5]
        sigma_M[e] = momento_critico * h[e] / (2 * I[e])
        sigma_xx[e] = abs(sigma_N[e]) + abs(sigma_M[e])

    if show_plots:
        plot_results(coordenadas, coordeform, conect, sigma_xx, ESC, output_dir=output_dir)

    return {
        "Kglobal": Kglobal,
        "Fglobal": Fglobal,
        "desloc": desloc,
        "desloc_mm": desloc_mm,
        "reacoes": reacoes,
        "forcas_elementares_locais": FF,
        "sigma_N": sigma_N,
        "sigma_M": sigma_M,
        "sigma_xx": sigma_xx,
        "coordenadas_deformadas": coordeform,
        "escala_deformada": ESC,
        "escala_manual_arquivo": ESC_manual,
        "comprimentos": lengths,
        "angulos_rad": angles,
    }


def plot_results(coordenadas: np.ndarray, coordeform: np.ndarray, conect: np.ndarray,
                 sigma_xx: np.ndarray, escala: float, output_dir: str | Path | None = None):
    """Plota geometria indeformada, deformada e tensões por elemento.

    Se output_dir for informado, salva as figuras nessa pasta.
    """
    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Figura 1: estrutura deformada
    fig1 = plt.figure(figsize=(8, 6))
    for e, (n1, n2) in enumerate(conect):
        x = coordenadas[[n1, n2], 0]
        y = coordenadas[[n1, n2], 1]
        plt.plot(x, y, "k--", linewidth=1, marker="o", label="Indeformada" if e == 0 else None)

        xd = coordeform[[n1, n2], 0]
        yd = coordeform[[n1, n2], 1]
        plt.plot(xd, yd, linewidth=3, marker="o", label="Deformada" if e == 0 else None)

    plt.axis("equal")
    plt.title(f"Estrutura deformada - escala = {escala:g}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        fig1.savefig(output_path / "estrutura_deformada.png", dpi=300, bbox_inches="tight")

    # Figura 2: tensões normais
    fig2 = plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=np.min(sigma_xx), vmax=np.max(sigma_xx))

    for e, (n1, n2) in enumerate(conect):
        xd = coordeform[[n1, n2], 0]
        yd = coordeform[[n1, n2], 1]
        color = cmap(norm(sigma_xx[e]))
        plt.plot(xd, yd, color=color, linewidth=3, marker="o")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Tensão normal σxx [Pa]")

    plt.axis("equal")
    plt.title("Tensões normais")
    plt.xlabel("x deformado")
    plt.ylabel("y deformado")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        fig2.savefig(output_path / "tensoes_normais.png", dpi=300, bbox_inches="tight")

    plt.show()


def print_results(results: dict):
    """Imprime resultados principais em formato tabular simples."""
    desloc_mm = results["desloc_mm"]
    reacoes = results["reacoes"]
    FF = results["forcas_elementares_locais"]
    sigma_xx = results["sigma_xx"]

    print("\nFATOR DE AMPLIAÇÃO DA DEFORMADA")
    print(f"Escala usada no desenho = {results['escala_deformada']:.6g}")
    print(f"Escala manual lida do arquivo = {results['escala_manual_arquivo']:.6g}")

    print("\nDESLOCAMENTOS NODAIS")
    print("Nó        ux [mm]        uy [mm]      theta [rad]")
    for i in range(len(desloc_mm) // 3):
        print(f"{i + 1:2d}  {desloc_mm[3*i]:14.6e} {desloc_mm[3*i+1]:14.6e} {desloc_mm[3*i+2]/1e3:14.6e}")

    print("\nREAÇÕES NODAIS")
    print("Nó          Rx [N]          Ry [N]        Mz [N.m]")
    for i in range(len(reacoes) // 3):
        print(f"{i + 1:2d}  {reacoes[3*i]:14.6e} {reacoes[3*i+1]:14.6e} {reacoes[3*i+2]:14.6e}")

    print("\nESFORÇOS LOCAIS POR ELEMENTO")
    print("Elem        N1          V1          M1          N2          V2          M2")
    for e in range(FF.shape[0]):
        print(f"{e + 1:2d}  " + " ".join(f"{v:11.4e}" for v in FF[e]))

    print("\nTENSÃO NORMAL MÁXIMA POR ELEMENTO")
    print("Elem     sigma_xx [MPa]")
    for e, sig in enumerate(sigma_xx, start=1):
        print(f"{e:2d}      {sig / 1e6:12.6f}")


if __name__ == "__main__":
    # Estrutura esperada do projeto:
    # projeto/
    # ├── data/
    # │   └── Viga2D.txt
    # ├── output/
    # └── viga2D.py
    #
    # Caso este script esteja dentro da própria pasta data, o código também funciona.

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name.lower() == "data" else script_dir

    input_path = project_dir / "data" / "Viga2D.txt"
    output_path = project_dir / "output"

    results = solve_frame(input_path, show_plots=True, output_dir=output_path)
    print_results(results)

    print(f"\nImagens salvas em: {output_path.resolve()}")
