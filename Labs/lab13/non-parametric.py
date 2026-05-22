import numpy as np
import scipy.stats as stats

print("\n--- EXERCISE 2: NON-PARAMETRIC TESTS ---")

# 1. Recreate the setup from J.G. Villegas's paper
# 4 Metaheuristics evaluated across 21 independent problem instances
np.random.seed(42)  # Set seed for reproducible distribution simulation

# Simulated data where a lower objective function score implies better optimization
metaheuristic_1 = np.random.uniform(50, 70, 21) # Poorer performance
metaheuristic_2 = np.random.uniform(40, 55, 21) # Moderate performance
metaheuristic_3 = np.random.uniform(40, 56, 21) # Highly comparable to Meta 2
metaheuristic_4 = np.random.uniform(30, 45, 21) # Top performing algorithm

# 2. Perform the Friedman Test using scipy.stats.friedmanchisquare
friedman_stat, p_val_friedman = stats.friedmanchisquare(
    metaheuristic_1, 
    metaheuristic_2, 
    metaheuristic_3, 
    metaheuristic_4
)

print("\n=== Friedman Test Result ===")
print(f"Friedman Statistic : {friedman_stat:.2f}")
print(f"p-value            : {p_val_friedman:.4e}")

# Conclusion rule block
if p_val_friedman < 0.05:
    print("\nConclusion: Reject H0. At least one metaheuristic performs significantly different from the others.")
else:
    print("\nConclusion: Fail to reject H0. No statistically significant performance differences observed.")