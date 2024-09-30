import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


df = pd.read_csv('/content/sleeepp.csv')

# Assuming 'Quality_of_Sleep' is the target variable
target_variable = 'Quality of Sleep'

# Extract features (X) and target variable (y)
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can choose a different classifier if needed)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Number of features to select (you can adjust this based on your requirements)
num_features_to_select = 5

# Create RFE model
rfe = RFE(estimator=clf, n_features_to_select=num_features_to_select)

# Fit the RFE model and transform the training data
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Print selected features
selected_features = np.array(X.columns)[rfe.support_]
print("Selected Features:", selected_features)

# Train and evaluate the model with the selected features
clf.fit(X_train_rfe, y_train)
accuracy = clf.score(X_test.iloc[:, rfe.support_], y_test)
print("Model Accuracy with Selected Features:", accuracy)
