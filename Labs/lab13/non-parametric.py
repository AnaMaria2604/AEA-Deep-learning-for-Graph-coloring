import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("=========================================================================\n")
print("Initial Hypotheses:")
print("  H0 (Null Hypothesis): The performance distributions (median ranks) of all")
print("                        metaheuristics are equal across all problem instances.")
print("  H1 (Alternative Hypothesis): At least one metaheuristic performs significantly")
print("                              differently (different median rank) from the others.")
print("\n=========================================================================")

np.random.seed(42) 
metaheuristic_1 = np.random.uniform(50, 70, 21) # Poorer performance
metaheuristic_2 = np.random.uniform(40, 55, 21) # Moderate performance
metaheuristic_3 = np.random.uniform(40, 56, 21) # Highly comparable to Meta 2
metaheuristic_4 = np.random.uniform(30, 45, 21) # Top performing algorithm

meta_names = ['Metaheuristic 1', 'Metaheuristic 2', 'Metaheuristic 3', 'Metaheuristic 4']
meta_arrays = [metaheuristic_1, metaheuristic_2, metaheuristic_3, metaheuristic_4]

data_matrix = np.array([metaheuristic_1, metaheuristic_2, metaheuristic_3, metaheuristic_4]).T

print("\n=== Step 1: Descriptive Statistics & Mean Rankings ===")
# stats.rankdata executes the core non-parametric ranking process
ranks = stats.rankdata(data_matrix, axis=1)
mean_ranks = np.mean(ranks, axis=0)
for i, name in enumerate(meta_names):
    print(f"{name} - Median Score: {np.median(meta_arrays[i]):.2f}, Mean Rank: {mean_ranks[i]:.4f}")

print("\n=== Step 2: Friedman Test Result ===")
friedman_stat, p_val_friedman = stats.friedmanchisquare(
    metaheuristic_1, 
    metaheuristic_2, 
    metaheuristic_3, 
    metaheuristic_4
)

print(f"Friedman Statistic : {friedman_stat:.4f}")
print(f"p-value            : {p_val_friedman:.4e}")

if p_val_friedman < 0.05:
    print("\n---> Conclusion: \n-----> Reject H0. \n-----> At least one metaheuristic performs significantly different from the others.")
else:
    print("\n---> Conclusion: \n-----> Fail to reject H0. \n-----> No statistically significant performance differences observed.")

print("\n=== Step 3: Post-Hoc Pairwise Analysis (Conover's Test Matrix) ===")
all_scores = np.concatenate(meta_arrays)
all_groups = (['Meta 1'] * 21) + (['Meta 2'] * 21) + (['Meta 3'] * 21) + (['Meta 4'] * 21)
all_blocks = list(range(1, 22)) * 4 

df_friedman = pd.DataFrame({
    'score': all_scores,
    'algorithm': all_groups,
    'instance': all_blocks
})

conover_results = sp.posthoc_conover_friedman(
    data_matrix, 
    p_adjust='bonferroni'
)

conover_results.columns = meta_names
conover_results.index = meta_names
print(conover_results)

print("\n=== Step 4: Pairwise Comparison Conclusions ===")
final_list = []

for i in range(len(meta_names)):
    for j in range(i+1, len(meta_names)):
        print(f"{meta_names[i]} vs {meta_names[j]}:")
        p_val = conover_results.iloc[i, j]
        if p_val < 0.05:
            print(f"---> Reject H0(p-value: {p_val:.4e}).\n")
        else:
            print(f"---> Fail to reject H0(p-value: {p_val:.4e}).\n")
            final_list.append((meta_names[i], meta_names[j], p_val))

print("\n=== Step 5: Pairs of metaheuristics that did NOT show significant performance differences ===")
if final_list:
    for pair in final_list:
        print(f"  - {pair[0]} vs {pair[1]} (p-value: {pair[2]:.4e})")
else:    
    print("All pairwise comparisons showed statistically significant differences in performance.")