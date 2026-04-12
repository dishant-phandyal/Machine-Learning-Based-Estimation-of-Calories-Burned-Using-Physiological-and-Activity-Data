import os
import joblib
import numpy as np
import warnings
import logging
from src.logger import logging  # Imports the custom logger we just made
from src.data_processing import load_and_preprocess_data, split_data
from src.model_training import train_and_evaluate

warnings.filterwarnings("ignore", category=UserWarning)

EXERCISE_DATA = 'data/exercise.csv'
CALORIES_DATA = 'data/calories.csv'
# Updated to use the artifacts folder
MODEL_PATH = 'artifacts/model.pkl' 

def main():
    logging.info("Calorie Estimation System Initialized")
    print("==================================================")
    print("🏃‍♂️ AI Calorie Estimation System Initialization 🏃‍♂️")
    print("==================================================\n")

    if not os.path.exists(MODEL_PATH):
        logging.info("No trained model found in artifacts/. Initiating training pipeline.")
        print("[INFO] No trained model found. Initiating training pipeline...")
        
        X, y = load_and_preprocess_data(EXERCISE_DATA, CALORIES_DATA)
        
        if X is None or y is None:
            logging.error("Pipeline halted: Data files missing or corrupt.")
            print("[ERROR] Pipeline halted due to missing data. Please check data/ folder.")
            return
            
        logging.info("Data loaded successfully. Splitting data.")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        logging.info("Starting model training.")
        model = train_and_evaluate(X_train, X_test, y_train, y_test, MODEL_PATH)
        logging.info("Model training complete and saved to artifacts/.")
    else:
        logging.info("Pre-trained model found in artifacts/. Loading model.")
        print("[INFO] Loading pre-trained XGBoost model from disk...\n")
        model = joblib.load(MODEL_PATH)
        print("[SUCCESS] Model loaded successfully!\n")
    
    print("==================================================")
    print("        Live Calorie Prediction Interface         ")
    print("==================================================")
    
    while True:
        try:
            gender_input = input("\nGender (M/F) or 'q' to quit: ").strip().lower()
            if gender_input == 'q': 
                logging.info("User exited the application.")
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
            logging.info(f"Prediction successful: {estimated_calories:.2f} kcal generated for user input.")
            
            print(f"\n🔥 >> ESTIMATED CALORIES BURNED: {estimated_calories:.2f} kcal << 🔥\n")
            print("-" * 50)
            
        except ValueError:
            logging.warning("Invalid user input detected (ValueError).")
            print("\n[WARNING] Invalid input. Please enter numerical values.\n")

if __name__ == "__main__":
    main()