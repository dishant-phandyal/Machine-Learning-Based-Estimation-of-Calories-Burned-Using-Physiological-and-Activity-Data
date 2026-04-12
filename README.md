# 🏃‍♂️ Machine Learning-Based Estimation of Calories Burned

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Overview
This project is a high-performance machine learning regression system designed to estimate the amount of calories burned during physical activity. By analyzing physiological parameters (such as heart rate and body temperature) alongside physical attributes (age, height, weight), the model provides accurate real-time calorie burn estimations.

Structured with a focus on modularity, the core training pipeline is entirely decoupled from the inference engine. This allows the model to be easily exported and integrated into scalable backend services or fitness-tracking applications.

## ✨ Key Features
- **High Accuracy Regression:** Utilizes an `XGBRegressor` (eXtreme Gradient Boosting) to capture complex, non-linear relationships between heart rate, duration, and energy expenditure.
- **Automated Pipeline:** Seamlessly handles data merging, categorical encoding, and feature engineering.
- **Persistent Inference:** The model is saved as a `.pkl` file after initial training, ensuring instant, zero-latency loading for subsequent predictions.
- **Interactive CLI:** Includes a built-in terminal interface for rapid testing and real-time physiological data input.

## 🗂️ Project Architecture
```text
Machine-Learning-Based-Estimation-of-Calories-Burned/
│
├── data/                   # Raw datasets (Not included in repo for privacy)
│   ├── calories.csv        # Contains User_ID and Calories burned
│   └── exercise.csv        # Contains biometric and duration data
│
├── models/                 # Serialized model artifacts
│   └── xgboost_model.pkl   # Auto-generated post-training
│
├── src/                    # Core modules
│   ├── __init__.py         
│   ├── data_processing.py  # Data ingestion and cleaning logic
│   └── model_training.py   # XGBoost training and evaluation pipeline
│
├── app.py                  # Main entry point and CLI interface
└── requirements.txt        # Project dependencies
```

## ⚙️ Installation & Setup

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/yourusername/calorie-estimation-ml.git](https://github.com/yourusername/calorie-estimation-ml.git)
cd calorie-estimation-ml
```

**2. Create and activate a Virtual Environment:**
```powershell
# Create the virtual environment
python -m venv calorie_env

# Activate the environment (Windows)
.\calorie_env\Scripts\activate

# Activate the environment (Mac/Linux)
source calorie_env/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Prepare the Data:**
Ensure that `calories.csv` and `exercise.csv` are placed directly inside the `data/` folder. *(Note: Datasets can be sourced from standard physiological databases like Kaggle's FMEL dataset).*

## 🚀 Usage

Run the main application script:
```bash
python app.py
```

### Execution Flow:
1. **First Run:** The system detects the absence of a compiled model. It will ingest the `data/` files, process them, train the XGBoost regressor, print the evaluation metrics (MAE & R-Squared), and serialize the model to the `models/` directory.
2. **Subsequent Runs:** The system bypasses training and instantly loads the pre-trained model into memory.
3. **Inference Loop:** The terminal will prompt you to enter biometric data (Gender, Age, Heart Rate, etc.) and instantly return the estimated calories burned.

## 📊 Model Evaluation
The system utilizes the following metrics to ensure reliability:
* **Mean Absolute Error (MAE):** Measures the average magnitude of errors in the predictions, without considering their direction.
* **R-Squared ($R^2$):** Represents the proportion of the variance for the dependent variable that's explained by the independent variables. 

*(Specific metric outputs will vary depending on the size and variance of the local dataset used during training).*

## 🔮 Future Scope
- Transition the CLI application into a RESTful API using **FastAPI** or **Flask** to support scalable web and mobile integrations.
- Implement deep learning architectures (e.g., Feedforward Neural Networks) for comparison against the XGBoost baseline.
- Expand feature engineering to include external environmental factors (humidity, ambient temperature).

---
*Developed with a passion for high-performance data applications.*
```