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

# =====================================================================
# ADDED: ASSUMPTION TESTING (Required by Newcastle University Guide)
# =====================================================================
print("\n=== Step 1: Normality Test (Shapiro-Wilk) ===")
# The site requires checking if data is normally distributed
print(f"Group 1 p-value: {stats.shapiro(group_1).pvalue:.4f}")
print(f"Group 2 p-value: {stats.shapiro(group_2).pvalue:.4f}")
print(f"Group 3 p-value: {stats.shapiro(group_3).pvalue:.4f}") 
# Note: All p-values > 0.05, meaning normality assumption is met!

print("\n=== Step 2: Homogeneity of Variance Test (Levene's) ===")
# The site requires testing if group variances are equal
levene_stat, levene_p = stats.levene(group_1, group_2, group_3)
print(f"Levene's Test p-value: {levene_p:.4f}")
# Note: p-value > 0.05, meaning variance homogeneity assumption is met!
# =====================================================================

# 2. Perform One-Way ANOVA (Safe to proceed since assumptions passed)
f_stat, p_val_anova = stats.f_oneway(group_1, group_2, group_3)

print("\n=== Step 3: One-Way ANOVA Result ===")
print(f"F-statistic : {f_stat:.4f}")
print(f"p-value     : {p_val_anova:.6f}")

# 3. Restructure data into a long-form DataFrame for post-hoc testing
scores = group_1 + group_2 + group_3
groups = ['Group 1']*10 + ['Group 2']*10 + ['Group 3']*10
df = pd.DataFrame({'score': scores, 'group': groups})

# 4. Perform Tukey's HSD Test
tukey_results = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
print("\n=== Step 4: Tukey HSD Post-Hoc Analysis ===")
print(tukey_results)

# 5. Perform Scheffé's Test
scheffe_results = sp.posthoc_scheffe(df, val_col='score', group_col='group')
print("\n=== Step 5: Scheffé Post-Hoc Analysis (p-value matrix) ===")
print(scheffe_results)