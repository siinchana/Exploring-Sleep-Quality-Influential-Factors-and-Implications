import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('/content/sleeepp.csv')

# Visualize the distribution of each feature using histograms
df.hist(figsize=(12, 8))
plt.show()

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot a heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Select features with positive correlation
positive_correlation_threshold = 0.5
selected_features = corr_matrix.index[corr_matrix['Quality of Sleep'] > positive_correlation_threshold]

# Display selected features with their correlation coefficients
selected_df = df[selected_features]
selected_corr = corr_matrix.loc[selected_features, selected_features]

print("Selected Features:")
print(selected_df.head())

print("\nCorrelation Matrix for Selected Features:")
print(selected_corr)

# Visualize the correlation matrix of selected features
plt.figure(figsize=(10, 8))
sns.heatmap(selected_corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix for Selected Features")
plt.show()
