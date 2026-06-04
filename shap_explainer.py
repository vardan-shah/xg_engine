import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# Import your functions from the previous scripts
from data_loader import load_world_cup_shots
from xg_model import engineer_xg_features

def explain_model_brain():
    print("Fetching data and training XGBoost model...")
    # Load and prep data
    df = load_world_cup_shots()
    df = engineer_xg_features(df)
    
    features = ['distance_to_goal', 'angle_to_goal', 'is_header']
    X = df[features]
    y = df['is_goal']
    
    # Train the model on all data for the visualization
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)
    
    print("\nCalculating SHAP values (cracking open the AI brain)...")
    # SHAP TreeExplainer reads the decision trees inside XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # --- VISUALIZATION ---
    # We use a dark theme to match our pitch visualization!
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6))
    
    # Generate the SHAP Beeswarm Plot
    shap.summary_plot(shap_values, X, show=False, plot_size=(10,6))
    
    # Clean up the titles
    plt.title("What Makes a Goal? (SHAP Feature Impact)", fontsize=16, fontweight='bold', pad=20)
    
    # Save the image
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("\n✅ Success! SHAP Visualization saved as 'shap_summary.png'")
    
    plt.show()

if __name__ == "__main__":
    explain_model_brain()