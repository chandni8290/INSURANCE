import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data
df = pd.read_csv('insurance.csv')

# Encode categorical columns
for col in ['sex', 'smoker', 'region']:
    df[col] = df[col].astype('category').cat.codes

# Define features and target
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.lb')
print("✅ Model trained and saved as model.lb")
