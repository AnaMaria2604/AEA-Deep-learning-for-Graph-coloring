import networkx as nx
import time
import pandas as pd
import os
import argparse
# import gnn2
import gnn_4
import gnn_pytorch
from gnn import gnn_coloring as gnn_numpy_coloring
from tabucol import tabucol
from tabucol_basic import tabucol as tabucol_basic
import numpy as np

VERIFIED_RESULTS = {
    "queen5_5": "5",  # 1
    "queen6_6": "7",    # 2
    "queen7_7": "7",    # 3
   "le450_25c": "25",  # 4
    "david": "11",    # 5
    "anna": "11",     # 6
    "jean": "10",     # 7
    "games120": "9",  # 8
    "homer": "13",    # 9
    "huck": "11",     # 10
    "miles250": "8",  # 11
    "miles500": "20", # 12
    "miles750": "31", # 13
    "miles1000": "42",# 14
    "miles1500": "73",# 15
    "myciel3" : "4", # 16
    "myciel4" : "5", # 17
    "myciel5" : "6", # 18
    "myciel6" : "7", # 19
    "myciel7" : "8", # 20
    "dsjc250.5": "28", # 21
    "flat300_28_0": "28", # 22
    "dsjc500.9": "126", # 23
    "dsjc1000.5": "85", # 24
}

   
def read_col_file(filepath):
    G = nx.Graph()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('c'):
                continue

            if line.startswith('p'):
                parts = line.split()
                # p edge num_nodes num_edges
                num_nodes = int(parts[2])
                G.add_nodes_from(range(1, num_nodes + 1))

            elif line.startswith('e'):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                G.add_edge(u, v)
    return G

def _select_gnn_backend(name):
    if name == "gnn_pytorch":
        return gnn_pytorch.gnn_coloring
    if name == "gnn2":
        return gnn2.gnn_coloring
    if name == "auto":
        return gnn_4.gnn_coloring
    
    return gnn_numpy_coloring #gnn


def run_benchmark(instance_name, G, nr_runs=1, gnn_backend="auto"):
    print(f"\n--- Testing Instance: {instance_name} ---")

    target_chi_str = VERIFIED_RESULTS.get(instance_name, "?")
    try:
        target_chi = int(target_chi_str.split('/')[0]) if target_chi_str != "?" else 0
    except Exception:
        target_chi = 0

    # ------------------------------------------------------------------
    # 1. GNN coloring
    # ------------------------------------------------------------------
    # gnn_fn = _select_gnn_backend(gnn_backend, instance_name)
    # best_k_gnn   = np.zeros(nr_runs, dtype=int)
    # t_gnn_total  = np.zeros(nr_runs)

    # for i in range(nr_runs):
    #     start = time.time()
    #     k_gnn_i, _ = gnn_fn(G, seed=i)
    #     t_gnn_total[i] = time.time() - start
    #     best_k_gnn[i] = k_gnn_i
    #     print(f"  GNN run {i+1:2d} [{gnn_backend}]: {k_gnn_i} colors  ({t_gnn_total[i]:.2f}s)")

    # mean_k_gnn   = np.mean(best_k_gnn)
    # min_k_gnn    = int(np.min(best_k_gnn))
    # max_k_gnn    = int(np.max(best_k_gnn))
    # std_k_gnn    = np.std(best_k_gnn)
    # mean_t_gnn   = np.mean(t_gnn_total)
    # print(f"GNN summary: mean={mean_k_gnn:.2f}  min={min_k_gnn}  max={max_k_gnn}  "
    #       f"stdev={std_k_gnn:.2f}  avg_time={mean_t_gnn:.2f}s")


    # 2. Tabucol (improved)
    nx_coloring   = nx.greedy_color(G, strategy="largest_first")
    k_tabu_init   = len(set(nx_coloring.values()))
    print(f"  Tabucol (improved) starting upper bound (nx greedy): {k_tabu_init} colors")

    best_k_tabu  = np.full(nr_runs, k_tabu_init, dtype=float)
    t_tabu_total = np.zeros(nr_runs)

    TABU_RESTARTS = 3          # independent restarts per k before declaring infeasible
    TABU_ITERS    = 50000      # iterations per single tabucol call

    for i in range(nr_runs):
        if k_tabu_init > 1:
            for k in range(k_tabu_init - 1, 1, -1):
                found = False
                t_start = time.time()
                for _ in range(TABU_RESTARTS):
                    success, _ = tabucol(G, k, iterations=TABU_ITERS)
                    if success:
                        found = True
                        break
                t_tabu_total[i] += time.time() - t_start

                if found:
                    best_k_tabu[i] = k
                    print(f"  Tabucol run {i+1}: found {k} colors!")
                    if target_chi > 0 and k <= target_chi:
                        break
                else:
                    print(f"  Tabucol run {i+1}: failed at {k} colors "
                          f"(after {TABU_RESTARTS} restarts).")
                    break

    mean_time_tabu = np.mean(t_tabu_total)
    mean_tabu      = np.mean(best_k_tabu)
    min_tabu       = int(np.min(best_k_tabu))
    max_tabu       = int(np.max(best_k_tabu))
    st_dev_tabu    = np.std(best_k_tabu)

    # 3. Tabucol (basic, non-optimized)
    best_k_tabu_basic  = np.full(nr_runs, k_tabu_init, dtype=float)
    t_tabu_basic_total = np.zeros(nr_runs)

    for i in range(nr_runs):
        if k_tabu_init > 1:
            for k in range(k_tabu_init - 1, 1, -1):
                found = False
                t_start = time.time()
                for _ in range(TABU_RESTARTS):
                    success, _ = tabucol_basic(G, k, iterations=TABU_ITERS)
                    if success:
                        found = True
                        break
                t_tabu_basic_total[i] += time.time() - t_start

                if found:
                    best_k_tabu_basic[i] = k
                    print(f"  TabucolBasic run {i+1}: found {k} colors!")
                    if target_chi > 0 and k <= target_chi:
                        break
                else:
                    print(f"  TabucolBasic run {i+1}: failed at {k} colors "
                          f"(after {TABU_RESTARTS} restarts).")
                    break

    mean_time_tabu_basic = np.mean(t_tabu_basic_total)
    mean_tabu_basic      = np.mean(best_k_tabu_basic)
    min_tabu_basic       = int(np.min(best_k_tabu_basic))
    max_tabu_basic       = int(np.max(best_k_tabu_basic))
    st_dev_tabu_basic    = np.std(best_k_tabu_basic)

    return {
        "Instance":            instance_name,
        "Nodes":               G.number_of_nodes(),
        "Edges":               G.number_of_edges(),
        
        # "GNN (mean)":          f"{mean_k_gnn:.2f}",
        # "GNN (min)":           min_k_gnn,
        # "GNN (max)":           max_k_gnn,
        # "GNN (stdev)":         f"{std_k_gnn:.2f}",
        # "GNN Time (mean s)":   f"{mean_t_gnn:.4f}",
        
        "Tabucol (mean)":      f"{mean_tabu:.2f}",
        "Tabucol (min)":       min_tabu,
        "Tabucol (max)":       max_tabu,
        "Tabucol (stdev)":     f"{st_dev_tabu:.2f}",
        "Tabucol Time (mean s)": f"{mean_time_tabu:.4f}",

        "TabucolBasic (mean)":      f"{mean_tabu_basic:.2f}",
        "TabucolBasic (min)":       min_tabu_basic,
        "TabucolBasic (max)":       max_tabu_basic,
        "TabucolBasic (stdev)":     f"{st_dev_tabu_basic:.2f}",
        "TabucolBasic Time (mean s)": f"{mean_time_tabu_basic:.4f}",
        
        "Verified Best k":     target_chi_str,
    }

def generate_markdown_table(results, output_path):
    """
    Write a grouped markdown report.
    `results` is a list of (category, record_dict) tuples.
    Columns written: Instance | Nodes | Edges |
                     GNN mean | GNN min | GNN max | GNN stdev | GNN Time (s) |
                     Tabu mean | Tabu min | Tabu max | Tabu stdev | Tabu Time (s) |
                     Best k
    """
    COLS = [
        ("Instance",              "Instance"),
        ("Nodes",                 "Nodes"),
        ("Edges",                 "Edges"),
        # ("GNN (mean)",            "GNN mean k"),
        # ("GNN (min)",             "GNN min k"),
        # ("GNN (max)",             "GNN max k"),
        # ("GNN (stdev)",           "GNN stdev"),
        # ("GNN Time (mean s)",     "GNN time (s)"),
        ("Tabucol (mean)",        "Tabu mean k"),
        ("Tabucol (min)",         "Tabu min k"),
        ("Tabucol (max)",         "Tabu max k"),
        ("Tabucol (stdev)",       "Tabu stdev"),
        ("Tabucol Time (mean s)", "Tabu time (s)"),
        ("TabucolBasic (mean)",        "TabuBasic mean k"),
        ("TabucolBasic (min)",         "TabuBasic min k"),
        ("TabucolBasic (max)",         "TabuBasic max k"),
        ("TabucolBasic (stdev)",       "TabuBasic stdev"),
        ("TabucolBasic Time (mean s)", "TabuBasic time (s)"),
        ("Verified Best k",       "Best k"),
    ]
    keys   = [c[0] for c in COLS]
    labels = [c[1] for c in COLS]

    header = "| " + " | ".join(labels) + " |"
    sep    = "| " + " | ".join(["---:"] * 3 + ["---:"] * 10 + ["---:"]) + " |"

    # group by category preserving insertion order
    from collections import OrderedDict
    groups = OrderedDict()
    for cat, rec in results:
        groups.setdefault(cat, []).append(rec)

    with open(output_path, "w") as f:
        f.write("# Graph Coloring Benchmark Results\n\n")
        f.write("> **GNN** — 2-layer GCN (numpy, self-supervised coloring loss )  \n")
        f.write("> **Tabu** — TabuCol local search, independent upper bound from nx largest-first greedy  \n")
        f.write("> All values averaged over 15 independent runs.\n\n")

        for cat, records in groups.items():
            f.write(f"## {cat.capitalize()}\n\n")
            f.write(header + "\n")
            f.write(sep + "\n")
            for rec in records:
                row = "| " + " | ".join(str(rec.get(k, "-")) for k in keys) + " |"
                f.write(row + "\n")
            f.write("\n")

 
def main():
    parser = argparse.ArgumentParser(description="Graph Coloring Benchmark")
    # parser.add_argument("--instances-dir", default="inst", help="Directory containing .col instances")
    # parser.add_argument("--output", default="results2.md", help="Output markdown file for results table")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--instances-dir", default=os.path.join(script_dir, "inst"), help="Directory containing .col instances")
    parser.add_argument("--output", default=os.path.join(script_dir, "comparison-tabucol.md"), help="Output markdown file for results table")
    # parser.add_argument(
    #     "--gnn-backend",
    #     choices=["auto", "gnn", "gnn2"],
    #     default="auto",
    #     help="GNN backend to use: NumPy, PyTorch, or auto-detect.",
    # )
    args = parser.parse_args()
    
    if not os.path.exists(args.instances_dir):
        print(f"Error: Instances directory '{args.instances_dir}' does not exist.")
        print("Please create it and add .col files to it.")
        return

    
    col_files = [f for f in os.listdir(args.instances_dir) if f.endswith('.col')]
    if not col_files:
        print(f"No .col files found in {args.instances_dir}")
        return
        
    ordered_files = []
    categories = [
        ("set1-myciel", [
            "myciel3", "myciel4", "myciel5", "myciel6", "myciel7",
        ]),
        ("set2-queen", [
            "queen5_5", "queen6_6", "queen7_7",
        ]),
        ("set3", [
            "anna", "david", "jean", "huck", "homer", "games120",
        ]),
        # ("set4-medium", [
        #     "flat300_28_0", "dsjc250.5", "le450_25c",
        # ]),
        # ("set5-miles", [
        #     "miles250", "miles500", "miles750", "miles1000", "miles1500",
        # ]),
        # ("set6-large", [
        #     "dsjc500.9", "dsjc1000.5",
        # ]),
    ]

    for cat_name, instances in categories:
        for inst in instances:
            filename = f"{inst}.col"
            if filename in col_files:
                ordered_files.append((filename, cat_name))


    categorised_results = []   # list of (category, record_dict)

    for filename, category in ordered_files:
        filepath = os.path.join(args.instances_dir, filename)
        instance_name = os.path.splitext(filename)[0]

        print(f"\n[{category.upper()}] Reading {filename}...")
        G = read_col_file(filepath)
        res = run_benchmark(instance_name, G) # gnn_backend=args.gnn_backend)
        categorised_results.append((category, res))

    if not categorised_results:
        print("No results to display.")
        return

    df = pd.DataFrame([r for _, r in categorised_results])
    print("\nFINAL COMPARISON TABLE")
    print(df.to_string(index=False))

    try:
        generate_markdown_table(categorised_results, args.output)
        print(f"\nResults successfully written to {args.output}")
    except Exception as e:
        print(f"Could not write markdown table to {args.output}: {e}")

if __name__ == "__main__":
    main()
