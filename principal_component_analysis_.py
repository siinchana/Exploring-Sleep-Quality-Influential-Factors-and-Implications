import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset from CSV file
data = pd.read_csv('dataset.csv')

# Preprocess data - ensure these are actual columns in your dataset
numeric_features = ['Age', 'Sleep duration', 'Quality of sleep', 'Physical activity',
                    'Stress level', 'Blood pressure', 'Heart rate', 'Daily steps']
categorical_features = ['Gender', 'BMI category', 'Sleep disorder']

# Create preprocessors for numeric and categorical data separately
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply preprocessors to columns based on data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Set up PCA with a specific number of components
pca = PCA(n_components=5)  # Adjust based on how many components you need

# Create a pipeline for the dataset
pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca)
])

# Fit and transform the dataset
transformed_data = pca_pipeline.fit_transform(data)

# Analyze component loadings to identify top features contributing to sleep-related components
component_loadings = pca_pipeline.named_steps['pca'].components_

# Convert component loadings to DataFrame for better readability
features = numeric_features + categorical_features
loadings = pd.DataFrame(component_loadings, columns=features)

# Sort features by absolute values in loadings (importance) for the first component
top_features = loadings.abs().sort_values(by=0, axis=1, ascending=False).iloc[0, :5]  # Top 5 features

# Print out the top features contributing to sleep quality/efficiency from the dataset
print("Top features contributing to sleep quality/efficiency in the dataset:")
print(top_features)
