from xgboost import XGBRegressor
from sklearn import metrics
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test, model_save_path):
    print("Training XGBoost Regressor...")
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)
    
    print(f"Model Evaluation -> Mean Absolute Error: {mae:.2f} | R-Squared: {r2:.4f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully at: {model_save_path}")
    
    return model