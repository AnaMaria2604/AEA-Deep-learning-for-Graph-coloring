import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# 1. Recreate the precise dataset from W. Montelpare et al. (Chapter 31)
# Reaction times for 3 groups of 10 individuals
drug_1 = [12, 15, 16, 13, 14, 15, 17, 11, 18, 14]
drug_2 = [45, 54, 39, 65, 34, 63, 55, 51, 53, 60]
placebo = [25, 26, 28, 22, 26, 27, 23, 26, 24, 25]

print("--- EXERCISE 1: PARAMETRIC TESTS ---")

# 2. Perform One-Way ANOVA using scipy.stats.f_oneway
f_stat, p_val_anova = stats.f_oneway(drug_1, drug_2, placebo)
print("\n=== One-Way ANOVA Result ===")
print(f"F-statistic : {f_stat:.2f}")
print(f"p-value     : {p_val_anova:.4e}")

# 3. Format data into a long-form DataFrame for post-hoc testing
data = drug_1 + drug_2 + placebo
group_labels = ['Drug1']*10 + ['Drug2']*10 + ['Placebo']*10
df = pd.DataFrame({'value': data, 'group': group_labels})

# 4. Perform Tukey's HSD Test using statsmodels' pairwise_tukeyhsd()
tukey_results = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=0.05)
print("\n=== Tukey HSD Post-Hoc Analysis ===")
print(tukey_results)

# 5. Perform Scheffé's Test using scikit-posthocs' posthoc_scheffe()
scheffe_results = sp.posthoc_scheffe(df, val_col='value', group_col='group')
print("\n=== Scheffé Post-Hoc Analysis (p-value matrix) ===")
print(scheffe_results)