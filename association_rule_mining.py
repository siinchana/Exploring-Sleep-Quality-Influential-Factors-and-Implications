import pandas as pd
import itertools
from sklearn.metrics import jaccard_score

# Function to calculate Jaccard distance
def calculate_jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union if union != 0 else 0

# Function to calculate Gini index
def calculate_gini_index(probabilities):
    sorted_probabilities = sorted(probabilities)
    n = len(sorted_probabilities)
    cumulative_sum = sum((2 * i - n - 1) * p for i, p in enumerate(sorted_probabilities))
    gini_index = cumulative_sum / (n * sum(sorted_probabilities))
    return gini_index

# Define input and output file paths
input_file = 'your_input_file.csv'
output_file = 'output_results.csv'

# Read association rules from CSV file
association_rules = pd.read_csv(input_file)

# Initialize empty lists to store results
jaccard_values = []
gini_indices = []

# Calculate Jaccard distance and Gini index for each rule
for index, rule in association_rules.iterrows():
    antecedent_set = set(rule['Antecedent'].split(','))
    consequent_set = set(rule['Consequent'].split(','))

    # Jaccard distance calculation
    jaccard_value = calculate_jaccard_distance(antecedent_set, consequent_set)
    jaccard_values.append(jaccard_value)

    # Gini index calculation
    support_values = [rule['Support'], 1 - rule['Support']]
    gini_index = calculate_gini_index(support_values)
    gini_indices.append(gini_index)

# Add new columns for Jaccard and Gini values to the DataFrame
association_rules['JaccardValue'] = jaccard_values
association_rules['GiniIndex'] = gini_indices

# Save the results to a new CSV file
association_rules.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
