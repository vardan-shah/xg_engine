import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# Import our data loader from Step 1
from data_loader import load_world_cup_shots

def engineer_xg_features(df):
    print("Calculating Shot Geometry (Distance and Angle)...")
    # 1. The exact center of the goal line in StatsBomb is X=120, Y=40
    # We use the Pythagorean theorem to calculate the exact distance of the shot
    df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)
    
    # 2. Calculate the absolute horizontal angle from the center of the pitch (Y=40)
    df['angle_to_goal'] = np.abs(40 - df['y'])
    
    # 3. Headers are significantly harder to score than foot shots. We create a binary feature (1 or 0)
    df['is_header'] = (df['shot_body_part'] == 'Head').astype(int)
    
    # Drop any weird rows that couldn't calculate geometry
    return df.dropna(subset=['distance_to_goal', 'angle_to_goal'])

def train_xg_model():
    # Load and prep the data
    df = load_world_cup_shots()
    df = engineer_xg_features(df)
    
    print("\nInitializing XGBoost Machine Learning Pipeline...")
    # Define our Features (X) and Target (y)
    features = ['distance_to_goal', 'angle_to_goal', 'is_header']
    X = df[features]
    y = df['is_goal']
    
    # Train/Test Split (Keep 20% hidden for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training XGBoost on {len(X_train)} shots. Testing on {len(X_test)} hidden shots...")
    
    # Initialize and train the XGBoost algorithm
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities (xG) on the test set
    # .predict_proba() returns [Chance of Miss, Chance of Goal]. We want index 1.
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate using ROC-AUC (The industry standard for evaluating probability models)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✅ Expected Goals (xG) Model Successfully Trained!")
    print(f"🧠 Model AUC Score: {auc:.3f}")
    
    return model

if __name__ == "__main__":
    train_xg_model()