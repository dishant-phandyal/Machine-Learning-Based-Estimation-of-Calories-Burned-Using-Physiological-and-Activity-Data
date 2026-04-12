import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(exercise_path, calories_path):
    try:
        calories_data = pd.read_csv(calories_path)
        exercise_data = pd.read_csv(exercise_path)
        
        df = pd.merge(exercise_data, calories_data, on='User_ID')
        df.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
        
        X = df.drop(columns=['User_ID', 'Calories'], axis=1)
        y = df['Calories']
        
        return X, y
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def split_data(X, y, test_size=0.2, random_state=2):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)