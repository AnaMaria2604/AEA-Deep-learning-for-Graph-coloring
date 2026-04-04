import networkx as nx
import time
import pandas as pd
import os
import argparse
from greedy import greedy_sdl_coloring
from tabucol import tabucol
import numpy as np

VERIFIED_RESULTS = {
    "queen5_5": "5",
    "queen6_6": "7",
   # "le450_25c": "25",
    "david": "11",
    "anna": "11",
    "jean": "10",
    "games120": "9",
    "homer": "13",
    "huck": "11",
    "miles250": "8",
    "miles500": "20",
    "miles750": "31",
    "miles1000": "42",
    "miles1500": "73",
    "myciel3" : "4",
    "myciel4" : "5",
    "myciel5" : "6",
    "myciel6" : "7",
    "myciel7" : "8",
    # "dsjc250.5": "28",
    # "flat300_28_0": "28",
    # "dsjc500.9": "126",
    # "dsjc1000.5": "85",
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

def run_benchmark(instance_name, G, nr_runs=15):
    print(f"\n--- Testing Instance: {instance_name} ---")
    
    # 1. Greedy (Baseline)
    start = time.time()
    k_greedy, _ = greedy_sdl_coloring(G)
    end = time.time()
    t_greedy = end - start
    print(f"Greedy found {k_greedy} colors!")
    
    # 2. Tabucol (Attempt to find the best k)
    #best_k_tabu = k_greedy
    best_k_tabu = np.full(nr_runs, k_greedy)
    #t_tabu_total = 0
    t_tabu_total = np.zeros(nr_runs)
    target_chi_str = VERIFIED_RESULTS.get(instance_name, "?")
    try:
        target_chi = int(target_chi_str.split('/')[0]) if target_chi_str != "?" else 0
    except Exception:
        target_chi = 0
    
    # If the graph has no edges or very few, k_greedy could be 1
    for i in range(nr_runs):
        if k_greedy > 1:
            for k in range(k_greedy - 1, 1, -1):
                start = time.time()
                success, _ = tabucol(G, k, iterations=10000)
                end = time.time()
                #t_tabu_total += (end - start)
                t_tabu_total[i] += (end - start)

                if success:
                    #best_k_tabu = k
                    best_k_tabu[i] = k
                    print(f"Tabucol found {k} colors!")
                    if target_chi > 0 and k <= target_chi: # Stop if we hit the known optimal
                        break
                else:
                    print(f"Tabucol failed at {k} colors.")
                    break

    mean_time_tabu = np.mean(t_tabu_total)
    mean_tabu = np.mean(best_k_tabu)
    min_tabu = np.min(best_k_tabu)
    max_tabu = np.max(best_k_tabu)
    st_dev_tabu = np.std(best_k_tabu)


    return {
        "Instance": instance_name,
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Greedy k": k_greedy,
        "Greedy Time (s)": f"{t_greedy:.4f}",
        "Tabucol (mean)": f"{mean_tabu:.2f}",
        "Tabucol (min)": min_tabu,
        "Tabucol (max)": max_tabu,
        "Tabucol (stdev)": f"{st_dev_tabu:.2f}",
        "Tabucol Time (mean s)": f"{mean_time_tabu:.4f}",
        #"Tabucol k": best_k_tabu,
        #"Tabucol Time (s)": f"{t_tabu_total:.4f}",
        "Verified Best k": target_chi_str
    }

def generate_markdown_table(df, output_path):
    headers = df.columns.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    with open(output_path, "w") as f:
        f.write("# Graph Coloring Benchmark Results\n\n")
        f.write(header_row + "\n")
        f.write(sep_row + "\n")
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(x) for x in row.values) + " |"
            f.write(row_str + "\n")

def main():
    parser = argparse.ArgumentParser(description="Graph Coloring Benchmark")
    parser.add_argument("--instances-dir", default="inst", help="Directory containing .col instances")
    parser.add_argument("--output", default="results.md", help="Output markdown file for results table")
    args = parser.parse_args()
    
    if not os.path.exists(args.instances_dir):
        print(f"Error: Instances directory '{args.instances_dir}' does not exist.")
        print("Please create it and add .col files to it.")
        return

    results = []
    
    col_files = [f for f in os.listdir(args.instances_dir) if f.endswith('.col')]
    if not col_files:
        print(f"No .col files found in {args.instances_dir}")
        return
        
    ordered_files = []
    categories = [
        ("small", ["queen5_5", "myciel5", "le450_25c"]),
        ("medium", ["david", "anna", "jean", "dsjc250.5", "flat300_28_0"]),
        ("large", ["dsjc500.9", "dsjc1000.5"])
    ]
    
    visited = set()
    for cat_name, instances in categories:
        for inst in instances:
            filename = f"{inst}.col"
            if filename in col_files:
                ordered_files.append((filename, cat_name))
                visited.add(filename)
                
    for filename in sorted(col_files):
        if filename not in visited:
            ordered_files.append((filename, "other"))
            
    for filename, category in ordered_files:
        filepath = os.path.join(args.instances_dir, filename)
        instance_name = os.path.splitext(filename)[0]
        
        print(f"\n[{category.upper()}] Reading {filename}...")
        G = read_col_file(filepath)
        res = run_benchmark(instance_name, G)
        res["Category"] = category
        results.append(res)
    
    if results:
        df = pd.DataFrame(results)
        cols = df.columns.tolist()
        if "Category" in cols:
            cols.insert(0, cols.pop(cols.index("Category")))
            df = df[cols]
    else:
        df = pd.DataFrame(results)
        
    print("\nFINAL COMPARISON TABLE")
    print(df.to_string(index=False))
    
    # Generate Markdown Table
    try:
        generate_markdown_table(df, args.output)
        print(f"\nResults successfully written to {args.output}")
    except Exception as e:
        print(f"Could not write markdown table to {args.output}: {e}")

if __name__ == "__main__":
    main()
