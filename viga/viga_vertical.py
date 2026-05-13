from pathlib import Path
from viga2D import solve_frame, print_results

# ============================================================
# CASO 2 - VIGA VERTICAL EM BALANCO
# Engaste no no 1 e carga horizontal na ponta livre no no 2
# ============================================================

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
output_dir = project_dir / "output" / "viga_vertical"
data_dir.mkdir(exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / "Viga2D_vertical.txt"

input_file.write_text("""NNOE NGLN NNOS NEL NFORC NRESTR ESC
2, 3, 2, 1, 1, 3, 1000
COORDENADAS GEOMETRICAS
0, 0
0, 1
CONECTIVIDADE
NO 1, NO 2, E, A, I, h
1, 2, 210e9, 5.5E-3, 1.6638E-5, 0.1357
FORCAS
2, 0, 5000
RESTRICOES
1, 0
1, 1
1, 2
""", encoding="latin-1")

results = solve_frame(input_file, show_plots=True, output_dir=output_dir)
print_results(results)

# Solucao analitica para viga em balanco com carga transversal na ponta
P = 5000.0
L = 1.0
E = 210e9
I = 1.6638e-5
v_max = P * L**3 / (3 * E * I)
theta_max = P * L**2 / (2 * E * I)

print("\nSOLUCAO ANALITICA - VIGA EM BALANCO")
print(f"Deslocamento transversal maximo = {v_max:.6e} m = {v_max*1000:.6f} mm")
print(f"Rotacao na ponta livre          = {theta_max:.6e} rad")
print(f"\nImagens salvas em: {output_dir.resolve()}")
