import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("house_rent.csv")

# Select important features
features = [
    "BHK",
    "Size",
    "Bathroom",
    "City",
    "Furnishing Status",
    "Tenant Preferred"
]

X = df[features]
y = df["Rent"]

# Encode categorical columns
encoders = {}
for col in ["City", "Furnishing Status", "Tenant Preferred"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
r2 = r2_score(y_test, model.predict(X_test))
print("R² Score:", r2)

# Save model & encoders
pickle.dump(model, open("rf_rent_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ House Rent Random Forest model trained")
