import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('/content/sleeepp.csv')

# Prepare features and labels
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle NaN values in training data
nan_indices = np.isnan(y_train)
X_train = X_train[~nan_indices]
y_train = y_train[~nan_indices]

# Ensure the number of samples match after handling NaN values
assert len(X_train) == len(y_train), "Number of samples in X_train and y_train should match."

# Train the SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
