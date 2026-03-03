import re
from pathlib import Path
from ortools.sat.python import cp_model

def solve_blocked_queens(n, blocks):
    model = cp_model.CpModel()
    queens = [model.new_int_var(0, n-1, f'queen_{i}') for i in range(n)]

    model.add_all_different(queens)
    model.add_all_different([queens[i] + i for i in range(n)])
    model.add_all_different([queens[i] - i for i in range(n)])

    for r, c in blocks:
        if 0 <= r < n and 0 <= c < n:
            model.add(queens[c] != r)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for row in range(n):
            line = ""
            for col in range(n):
                if solver.Value(queens[col]) == row:
                    line += " Q "
                else:
                    line += " - "
            print(line)

        print("\nPozitii exacte:")
        for i in range(n):
            row_val = solver.Value(queens[i])
            print(f"Coloana {i}: Randul {row_val}")
    else:
        print("Nu s-a gasit nicio solutie.")

    return status == cp_model.OPTIMAL or status == cp_model.FEASIBLE

def parse_param_file(file_path):
    content = Path(file_path).read_text()

    n_match = re.search(r'letting n\s*=\s*(\d+)', content)
    n = int(n_match.group(1)) if n_match else 0

    blocks_match = re.search(r'letting blocks\s*=\s*\[(.*?)]', content, re.DOTALL)
    blocks = []
    if blocks_match:
        pairs = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*]', blocks_match.group(1))
        blocks = [(int(r), int(c)) for r, c in pairs]

    return n, blocks

def process_all_instances(root_folder):
    for path in Path(root_folder).rglob('*.param'):
        n, blocks = parse_param_file(path)
        if n > 0:
            result = solve_blocked_queens(n, blocks)
            status_str = "SOLVED" if result else "NO SOLUTION"
            print(f"\n {path.name} {status_str}")

process_all_instances('block-10')