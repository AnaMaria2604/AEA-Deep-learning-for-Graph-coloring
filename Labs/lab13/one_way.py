import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

print("--- EXERCISE 1: ANALYSIS WITH IMAGE DATA ---")

# 1. Extracted data points from your provided table image
group_1 = [3, 5, 3, 2, 4, 6, 9, 3, 8, 10]
group_2 = [10, 8, 15, 9, 11, 16, 17, 15, 7, 10]
group_3 = [29, 15, 14, 15, 17, 10, 8, 11, 18, 19]

# 2. Perform One-Way ANOVA using scipy.stats.f_oneway
f_stat, p_val_anova = stats.f_oneway(group_1, group_2, group_3)

print("\n=== One-Way ANOVA Result ===")
print(f"F-statistic : {f_stat:.4f}")
print(f"p-value     : {p_val_anova:.6f}")

# 3. Restructure data into a long-form DataFrame for post-hoc testing
scores = group_1 + group_2 + group_3
groups = ['Group 1']*10 + ['Group 2']*10 + ['Group 3']*10
df = pd.DataFrame({'score': scores, 'group': groups})

# 4. Perform Tukey's HSD Test using statsmodels' pairwise_tukeyhsd()
tukey_results = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
print("\n=== Tukey HSD Post-Hoc Analysis ===")
print(tukey_results)

# 5. Perform Scheffé's Test using scikit-posthocs' posthoc_scheffe()
scheffe_results = sp.posthoc_scheffe(df, val_col='score', group_col='group')
print("\n=== Scheffé Post-Hoc Analysis (p-value matrix) ===")
print(scheffe_results)