from ortools.sat.python import cp_model

def solve_n_queens(n):
    model = cp_model.CpModel()

    # queens[i] = j  <=> regina de pe coloana i este pe linia j
    queens = [model.NewIntVar(1, n, f'queen_{i}') for i in range(n)]

    model.AddAllDifferent(queens)

    # Diagonale diferite
    # Diagonala 1: rand + coloana: unice
    # Diagonala 2: rand - coloana: unice
    model.AddAllDifferent([queens[i] + i for i in range(n)])
    model.AddAllDifferent([queens[i] - i for i in range(n)])

    # Pozitii blocate
    model.Add(queens[0] != 3)
    model.Add(queens[1] != 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Afisare
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for row in range(1, n + 1):
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

solve_n_queens(4)

# o solutie
# -  -  Q  - 
#  Q  -  -  - 
#  -  -  -  Q 
#  -  Q  -  - 

# Pozitii exacte:
# Coloana 0: Randul 2
# Coloana 1: Randul 4
# Coloana 2: Randul 1
# Coloana 3: Randul 3