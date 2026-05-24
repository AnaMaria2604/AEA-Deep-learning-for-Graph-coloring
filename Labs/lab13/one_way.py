import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

group_1 = [3, 5, 3, 2, 4, 6, 9, 3, 8, 10] # 10 samples from Group 1
group_2 = [10, 8, 15, 9, 11, 16, 17, 15, 7, 10] # 10 samples from Group 2
group_3 = [29, 15, 14, 15, 17, 10, 8, 11, 18, 19] # 10 samples from Group 3
groups = ['Group 1']*10 + ['Group 2']*10 + ['Group 3']*10  
groups = [group_1, group_2, group_3]
scores = group_1 + group_2 + group_3

print("=========================================================================\n")
print("Initial Hypotheses:")
print("  H0 (Null Hypothesis): μ1 = μ2 = μ3")
print("  H1 (Alternative Hypothesis): At least one group mean is significantly different.")
print("\n=========================================================================")

print("\n=== Step 1: Normality Test (Shapiro-Wilk) ===")
for i, group in enumerate(groups, start=1):
    print(f"Group {i} - P-value: {stats.shapiro(group).pvalue:.4f}, Mean: {np.mean(group):.2f}, Std Dev: {np.std(group, ddof=1):.2f}")

if (stats.shapiro(group_1).pvalue > 0.05 and 
    stats.shapiro(group_2).pvalue > 0.05 and 
    stats.shapiro(group_3).pvalue > 0.05):
    print("\n---> Conclusion: \n-----> All groups appear to be normally distributed(the p-values > 0.05) \n-----> Fail to reject H0")

print("\n=== Step 2: Homogeneity of Variance Test (Levene's) ===")
levene_stat, levene_p = stats.levene(group_1, group_2, group_3)
print(f"Levene's Test p-value: {levene_p:.4f}")
if levene_p > 0.05:
    print("\n---> Conclusion: \n-----> Variances are homogeneous (p-value > 0.05). \n-----> Fail to reject H0.")

print("\n=== Step 3: One-Way ANOVA Result ===")
f_stat, p_val_anova = stats.f_oneway(group_1, group_2, group_3)
print(f"F-statistic : {f_stat:.4f}")
print(f"p-value     : {p_val_anova:.6f}")

if p_val_anova < 0.05:
    print("\n---> Conclusion: \n-----> Reject H0. \n-----> At least one group mean is significantly different from the others.")
else:    
    print("\n---> Conclusion: \n-----> Fail to reject H0. \n-----> No statistically significant differences between group means observed.")

groups = ['Group 1']*10 + ['Group 2']*10 + ['Group 3']*10 
df = pd.DataFrame({'score': scores, 'group': groups})

print("\n=== Step 4: Tukey HSD Post-Hoc Analysis ===")
tukey_results = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
print(tukey_results)

if tukey_results.reject.any():
    print("\n---> Conclusion: \n-----> Reject H0 for at least one pairwise comparison. \n-----> There are significant differences between some group means.")
else:    
    print("\n---> Conclusion: \n-----> Fail to reject H0 for all pairwise comparisons. \n-----> No significant differences between group means detected.")

print("\n=== Step 5: Scheffé Post-Hoc Analysis (p-value matrix) ===")
scheffe_results = sp.posthoc_scheffe(df, val_col='score', group_col='group')
print(scheffe_results)