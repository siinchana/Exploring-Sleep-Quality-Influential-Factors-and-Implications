from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

# Read data from CSV file into a pandas DataFrame
df = pd.read_csv('/content/sleeepp.csv')
df = pd.read_csv(csv_path)

# Create a contingency table between 'antecedents' and 'consequents'
contingency_table = pd.crosstab(df['antecedents'], df['consequents'])

# Perform chi-square test to get the chi-squared statistic
chi2_stat, _, _, _ = chi2_contingency(contingency_table)

# Calculate Cramér's V
n = contingency_table.sum().sum()  # total number of observations
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

# Display the result
print(f"Cramér's V: {cramers_v}")
