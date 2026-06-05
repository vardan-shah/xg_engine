import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_loader import load_world_cup_shots

def engineer_xg_features(raw_df):
    print("Calculating Shot Geometry...")
    # Prevent in-place mutation warnings
    df = raw_df.copy()
    
    df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)
    df['angle_to_goal'] = np.abs(40 - df['y']) # Absolute lateral deviation
    
    df['is_header'] = df['shot_body_part'].apply(
        lambda x: 1 if (isinstance(x, str) and x == 'Head') or (isinstance(x, dict) and x.get('name') == 'Head') else 0
    )
    return df.dropna(subset=['distance_to_goal', 'angle_to_goal'])

def train_and_predict_all():
    """
    Full pipeline called by Streamlit's @st.cache_data.
    Loads data, engineers features, trains with honest eval,
    and returns the enriched DataFrame with xg_model predictions.
    """
    raw_df = load_world_cup_shots()
    df = engineer_xg_features(raw_df)

    FEATURES = ['distance_to_goal', 'angle_to_goal', 'is_header']
    X = df[FEATURES]
    y = df['is_goal']

    # Stratify ensures exact goal ratio in both splits for reproducible AUC
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"🧠 xG Model Validation AUC Score: {auc:.3f}")

    # Predict xG for ALL shots so the dashboard has per-player totals
    df['xg_model'] = model.predict_proba(X)[:, 1]
    return df

if __name__ == "__main__":
    train_and_predict_all()