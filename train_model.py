import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
try:
    df = pd.read_csv('DATASET CAR PRICE.csv')
    df.columns = df.columns.str.strip()  # Clean column names
    print("✅ CSV Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: 'DATASET CAR PRICE.csv' not found.")
    exit()

# 2. Features and Target
# We drop 'Price' (Target) and 'Car ID' (Identifier)
X = df.drop(['Price', 'Car ID'], axis=1)
y = df['Price']

# Identify categories vs numbers automatically
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Preprocessing & Model Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Training
print("⏳ Training Model...")
model_pipeline.fit(X_train, y_train)

# 6. Metrics Calculation
preds = model_pipeline.predict(X_test)
acc = r2_score(y_test, preds)
mse_val = mean_squared_error(y_test, preds)

# 7. Save the Brain (car_price_models.pkl)
model_data = {
    "model": model_pipeline,
    "accuracy": round(acc * 100, 2),
    "mse": round(mse_val, 2),
    "algo": "Random Forest Regressor",
    "feature_names": X.columns.tolist()
}

with open('car_price_models.pkl', 'wb') as f:
    pickle.dump(model_data, f, protocol=4)

# 8. Save the UI Metadata (ui_data.pkl)
ui_info = {
    'brands': sorted(df['Brand'].unique().tolist()),
    'fuel_types': sorted(df['Fuel Type'].unique().tolist()),
    'transmissions': sorted(df['Transmission'].unique().tolist()),
    'col_names': {
        'brand': 'Brand',
        'fuel': 'Fuel Type',
        'trans': 'Transmission'
    }
}
with open('ui_data.pkl', 'wb') as f:
    pickle.dump(ui_info, f, protocol=4)

print(f"✅ SUCCESS! Accuracy: {model_data['accuracy']}% | MSE: {model_data['mse']}")