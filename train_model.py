import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
df = pd.read_csv('DATASET CAR PRICE.csv')
df.columns = df.columns.str.strip()

# 2. Features and Target
X = df.drop(['Price', 'Car ID'], axis=1)
y = df['Price']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 4. Train
model_pipeline.fit(X_train, y_train)
acc = r2_score(y_test, model_pipeline.predict(X_test))
mse = mean_squared_error(y_test, model_pipeline.predict(X_test))

# 5. Save Files
model_data = {
    "model": model_pipeline,
    "accuracy": round(acc * 100, 2),
    "mse": round(mse, 2),
    "algo": "Random Forest",
    "feature_names": X.columns.tolist()
}

with open('car_price_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)

ui_info = {
    'brands': sorted(df['Brand'].unique().tolist()),
    'fuel_types': sorted(df['Fuel Type'].unique().tolist()),
    'transmissions': sorted(df['Transmission'].unique().tolist())
}
with open('ui_data.pkl', 'wb') as f:
    pickle.dump(ui_info, f)
    print(f"✅ SUCCESS! Accuracy: {model_data['accuracy']}% | MSE: {model_data['mse']}")

print(f"Done! Accuracy: {model_data['accuracy']}%")
