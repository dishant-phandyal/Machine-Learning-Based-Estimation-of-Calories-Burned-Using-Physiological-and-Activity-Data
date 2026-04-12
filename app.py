import os
import joblib
import numpy as np
import warnings
from src.data_processing import load_and_preprocess_data, split_data
from src.model_training import train_and_evaluate

warnings.filterwarnings("ignore", category=UserWarning)

EXERCISE_DATA = 'data/exercise.csv'
CALORIES_DATA = 'data/calories.csv'
MODEL_PATH = 'models/xgboost_model.pkl'

def main():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Initiating training pipeline...\n")
        X, y = load_and_preprocess_data(EXERCISE_DATA, CALORIES_DATA)
        
        if X is None or y is None:
            print("Pipeline halted due to missing data.")
            return
            
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_and_evaluate(X_train, X_test, y_train, y_test, MODEL_PATH)
    else:
        print("Loading pre-trained model...\n")
        model = joblib.load(MODEL_PATH)
    
    print("\n--- Personalized Calorie Burn Predictor ---")
    print("Enter the physiological parameters below (or type 'q' to quit)")
    
    while True:
        try:
            gender_input = input("\nGender (M/F) or 'q' to quit: ").strip().lower()
            if gender_input == 'q':
                break
                
            gender = 0 if gender_input == 'm' else 1
            age = float(input("Age (years): "))
            height = float(input("Height (cm): "))
            weight = float(input("Weight (kg): "))
            duration = float(input("Activity Duration (minutes): "))
            heart_rate = float(input("Average Heart Rate (bpm): "))
            body_temp = float(input("Body Temperature (Celsius): "))
            
            input_features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
            estimated_calories = model.predict(input_features)[0]
            print(f"\n>> Estimated Calories Burned: {estimated_calories:.2f} kcal <<")
            
        except ValueError:
            print("Invalid input. Please enter numerical values where requested.")

if __name__ == "__main__":
    main()