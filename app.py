import os
import joblib
import numpy as np
import warnings
from src.data_processing import load_and_preprocess_data, split_data
from src.model_training import train_and_evaluate

# Suppress XGBoost feature name warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

EXERCISE_DATA = 'data/exercise.csv'
CALORIES_DATA = 'data/calories.csv'
MODEL_PATH = 'models/xgboost_model.pkl'

def main():
    print("==================================================")
    print("🏃‍♂️ AI Calorie Estimation System Initialization 🏃‍♂️")
    print("==================================================\n")

    # 1. Model Training / Loading Pipeline
    if not os.path.exists(MODEL_PATH):
        print("[INFO] No trained model found. Initiating training pipeline...")
        
        X, y = load_and_preprocess_data(EXERCISE_DATA, CALORIES_DATA)
        
        if X is None or y is None:
            print("[ERROR] Pipeline halted due to missing data. Please check data/ folder.")
            return
            
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_and_evaluate(X_train, X_test, y_train, y_test, MODEL_PATH)
    else:
        print("[INFO] Loading pre-trained XGBoost model from disk...\n")
        model = joblib.load(MODEL_PATH)
        print("[SUCCESS] Model loaded successfully!\n")
    
    # 2. Interactive Prediction CLI
    print("==================================================")
    print("        Live Calorie Prediction Interface         ")
    print("==================================================")
    print("Type 'q' at any prompt to quit the application.\n")
    
    while True:
        try:
            gender_input = input("Gender (M/F): ").strip().lower()
            if gender_input == 'q': break
            gender = 0 if gender_input == 'm' else 1
            
            age_input = input("Age (years): ")
            if age_input.lower() == 'q': break
            age = float(age_input)
            
            height_input = input("Height (cm): ")
            if height_input.lower() == 'q': break
            height = float(height_input)
            
            weight_input = input("Weight (kg): ")
            if weight_input.lower() == 'q': break
            weight = float(weight_input)
            
            duration_input = input("Activity Duration (minutes): ")
            if duration_input.lower() == 'q': break
            duration = float(duration_input)
            
            heart_input = input("Average Heart Rate (bpm): ")
            if heart_input.lower() == 'q': break
            heart_rate = float(heart_input)
            
            temp_input = input("Body Temperature (Celsius): ")
            if temp_input.lower() == 'q': break
            body_temp = float(temp_input)
            
            # Format input for prediction: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
            input_features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
            estimated_calories = model.predict(input_features)[0]
            
            print(f"\n🔥 >> ESTIMATED CALORIES BURNED: {estimated_calories:.2f} kcal << 🔥\n")
            print("-" * 50)
            
        except ValueError:
            print("\n[WARNING] Invalid input. Please enter numerical values.\n")

if __name__ == "__main__":
    main()