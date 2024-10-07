# regression_script.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ----------------------------
# Step 1: Load the Data
# ----------------------------

# Let's load the train and test data
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# Check out the first few rows of the datasets to get an idea of what we're dealing with
print("Training Data Sample:\n", train_data.head())
print("\nTest Data Sample:\n", test_data.head())

# ----------------------------
# Step 2: Data Cleaning
# ----------------------------

# Some quick data cleaning
# 1. Let's deal with missing values in the "Item_Weight" and "Outlet_Size" columns.
# Impute missing values using the median for numerical columns and most frequent for categorical columns.

train_data['Item_Weight'].fillna(train_data['Item_Weight'].median(), inplace=True)
test_data['Item_Weight'].fillna(test_data['Item_Weight'].median(), inplace=True)

train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0], inplace=True)
test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0], inplace=True)

# 2. Clean up inconsistent categories in 'Item_Fat_Content'
train_data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)
test_data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)

# ----------------------------
# Step 3: Feature Engineering
# ----------------------------

# Drop columns that won't help with predictions
X_train = train_data.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y_train = train_data['Item_Outlet_Sales']

X_test = test_data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

# ----------------------------
# Step 4: Preprocessing
# ----------------------------

# Define which columns are numerical and which are categorical
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing pipeline
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# ----------------------------
# Step 5: Model Building
# ----------------------------

# Let's build a basic linear regression model using scikit-learn's Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the training set for evaluation purposes
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train_split, y_train_split)

# ----------------------------
# Step 6: Evaluation
# ----------------------------

# Make predictions on the validation set
y_pred = model.predict(X_val_split)

# Calculate evaluation metrics
mse = mean_squared_error(y_val_split, y_pred)
r2 = r2_score(y_val_split, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# ----------------------------
# Step 7: Predict on Test Data
# ----------------------------

# Now let's make predictions on the test data
test_predictions = model.predict(X_test)

# Save predictions to a CSV file for submission
submission = pd.DataFrame({'Item_Identifier': test_data['Item_Identifier'], 'Outlet_Identifier': test_data['Outlet_Identifier'], 'Item_Outlet_Sales': test_predictions})
submission.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'.")

