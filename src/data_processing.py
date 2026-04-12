import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(exercise_path, calories_path):
    try:
        calories_data = pd.read_csv(calories_path)
        exercise_data = pd.read_csv(exercise_path)
        
        # Merge datasets on User_ID
        df = pd.merge(exercise_data, calories_data, on='User_ID')
        
        # --- BULLETPROOF GENDER CONVERSION ---
        # 1. Convert to string, lowercase it, and strip any hidden spaces
        df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
        # 2. Map 'female' to 1 and 'male' to 0
        df['Gender'] = df['Gender'].map({'female': 1, 'male': 0})
        # 3. Force the column to be a strict integer
        df['Gender'] = df['Gender'].fillna(0).astype(int)
        
        # Separate features (X) and target label (y)
        X = df.drop(columns=['User_ID', 'Calories'])
        y = df['Calories']
        
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def split_data(X, y, test_size=0.2, random_state=2):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)