import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load clean dataset
df = pd.read_csv('dataset.csv')

# Encode categorical variables
encoders = {}
categorical_cols = ['species', 'breed_size', 'location']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Targets
X = df[['species', 'breed_size', 'age', 'location']]
y_cost = df['total_annual_cost']
y_risk = df['risk_tier']

# Train Cost Regressor
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
cost_model.fit(X, y_cost)

# Train Risk Classifier
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X, y_risk)

# Save artifacts
with open('cost_model.pkl', 'wb') as f:
    pickle.dump(cost_model, f)

with open('risk_model.pkl', 'wb') as f:
    pickle.dump(risk_model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Models trained and saved successfully: cost_model.pkl, risk_model.pkl, encoders.pkl")