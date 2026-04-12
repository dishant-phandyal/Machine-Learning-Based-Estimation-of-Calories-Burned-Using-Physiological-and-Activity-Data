# 🏃‍♂️ AI-Powered Calorie Estimation System (ML Project)

**Software and tools requirement**
* GitHub account
* VS Code account
* Python 3.10+

---

## 📌 Overview

This project aims to predict the amount of calories burned during physical activity based on various physiological and demographic factors such as heart rate, body temperature, age, and workout duration. 
The project follows a modular machine learning pipeline architecture, including data ingestion, transformation, model training, and a real-time interactive inference engine.

## 🚀 Project Objectives

* Analyze physiological and exercise datasets.
* Build a robust, decoupled ML pipeline.
* Perform feature engineering and strict categorical encoding.
* Train and evaluate advanced regression models (XGBoost).
* Deploy a reusable, scalable, and instant-loading project structure via CLI.

## 🧠 Problem Statement

Given a user's biometric attributes and real-time workout statistics, accurately predict the **Calories** burned during that specific session.

## 🗂️ Project Structure

```text
Machine-Learning-Based-Estimation-of-Calories-Burned/
│
├── artifacts/               # Generated files (Serialized artifacts)
│   └── model.pkl            # Pre-trained model saved after first run
│
├── data/                    # Raw dataset files
│   ├── calories.csv
│   └── exercise.csv
│
├── logs/                    # Timestamped system execution logs
│
├── src/                     # Core pipeline modules
│   ├── __init__.py
│   ├── logger.py            # Custom logging configuration
│   ├── data_processing.py   # Data ingestion & transformation logic
│   └── model_training.py    # Model trainer & evaluator
│
├── calorie_env/             # Virtual environment (ignored in git)
├── app.py                   # Main pipeline execution and CLI interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Tech Stack

* **Python 3.11**
* **Pandas, NumPy** (Data Manipulation)
* **Scikit-learn** (Splitting & Metrics)
* **XGBoost** (Core Regression Model)
* **Joblib** (Model Serialization)

## 🔄 ML Pipeline Workflow

**1. Data Ingestion**
* Reads `exercise.csv` and `calories.csv` from the source directory.
* Merges datasets dynamically using `User_ID`.

**2. Data Transformation**
* Handles string formatting and whitespace stripping.
* Applies encoding to categorical features (mapping 'male'/ 'female' to numeric boolean values).
* Strictly casts data types to prevent `object` mapping errors within the model matrix.
* Separates independent features (X) from the target variable (y).

**3. Model Training & Execution**
* Splits data into train & test subsets (80/20 split).
* Trains an `XGBRegressor` to capture complex, non-linear physiological relationships.
* Evaluates performance using Mean Absolute Error (MAE) and R-Squared ($R^2$).
* Saves the best model (`xgboost_model.pkl`) to disk for zero-latency reloading on subsequent runs.

## 📊 Features Used

**Numerical Features**
* `Age` (years)
* `Height` (cm)
* `Weight` (kg)
* `Duration` (minutes)
* `Heart_Rate` (bpm)
* `Body_Temp` (°C)

**Categorical Features**
* `Gender` (Male/Female)

## 🧪 How to Run the Project

**Step 1: Clone the repository**
```bash
git clone <your-repo-link>
cd Machine-Learning-Based-Estimation-of-Calories-Burned
```

**Step 2: Create & activate virtual environment**
```bash
python -m venv calorie_env
.\calorie_env\Scripts\activate   # Windows
# source calorie_env/bin/activate  # Mac/Linux
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run the pipeline**
```bash
python app.py
```

## 📈 Sample Output

After running the pipeline for the first time, the following artifacts are generated and the CLI initializes:
```text
models/
 └── xgboost_model.pkl

==================================================
        Live Calorie Prediction Interface         
==================================================
Gender (M/F): m
Age (years): 25
Height (cm): 175
Weight (kg): 75
Activity Duration (minutes): 30
Average Heart Rate (bpm): 120
Body Temperature (Celsius): 39

🔥 >> ESTIMATED CALORIES BURNED: 142.50 kcal << 🔥
```

## 📌 Key Highlights

* **High-Accuracy Regression:** Utilizes Gradient Boosting Decision Trees.
* **Modular and Scalable Architecture:** Clean separation of concerns between processing, training, and UI.
* **Persistent State:** Bypasses heavy retraining phases by caching the compiled model.
* **Interactive CLI:** Built-in error handling for seamless user data entry.

## ⚠️ Common Issues & Fixes

| Issue | Solution |
| :--- | :--- |
| `FileNotFoundError` | Ensure `calories.csv` and `exercise.csv` are placed strictly inside the `data/` folder. |
| `KeyError: 'object'` | Ensure `data_processing.py` includes the explicit string mapping and `.astype(int)` logic for the Gender column. |
| Execution Policy Error | Run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` in PowerShell before activating the `venv`. |
| Libraries not found | Ensure your IDE (VS Code) has the correct `calorie_env` interpreter selected in the bottom right corner. |

## 📚 Future Improvements

* Add Flask/FastAPI backend deployment.
* Develop a graphical web dashboard using Streamlit.
* Integrate CI/CD pipeline for automated testing.
* Implement Hyperparameter tuning (GridSearchCV).
* Docker containerization for OS-agnostic deployment.

## 👨‍💻 Authors

**B.Tech CS (AIML) - Semester VIII Project Team**
* **Dishant Phandyal** (Reg: 220220216)
* Vivek Chauhan
* Aditya Gupta
* Merajudaulah Shekh

## ⭐ Acknowledgements

* XGBoost & Scikit-learn documentation
* Kaggle FMEL physiological dataset
* ML pipeline architectural best practices

## 📬 Contact

Feel free to connect for collaboration or queries regarding this pipeline.
