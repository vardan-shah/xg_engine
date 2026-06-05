# 🏆 Expected Goals (xG) Engine & AI Scout

**Executive Summary:** I built a production-grade Expected Goals (xG) engine using StatsBomb event tracking data, XGBoost, and SHAP to analyze spatial shot geometries from the 2022 FIFA World Cup. The model achieves a mathematically robust Validation AUC score of 0.833 on a stratified holdout set. The final layer introduces a Streamlit web dashboard that pipes custom ML telemetry directly into an Anthropic LLM to auto-generate natural language scouting reports.

## 🛠️ Tech Stack & Infrastructure
* **Data Source:** StatsBomb Open Data API (`statsbombpy`)
* **Data Ingestion & Engineering:** `Pandas`, `NumPy`
* **Machine Learning Framework:** `XGBoost` (Extreme Gradient Boosting)
* **Model Interpretability:** `SHAP` (SHapley Additive exPlanations)
* **Spatial Visualizations:** `mplsoccer` (StatsBomb pitch layout), `Matplotlib`
* **Application & GenAI:** `Streamlit`, `Anthropic API` (Claude 3.5 Haiku), `python-dotenv`

## 🚀 Core Features & ML Pipeline
1. **Intelligent Data Ingestion:** Extracts and isolates over 1,400 raw shot events across all 64 tournament matches. Actively filters out `period == 5` (penalty shootouts) to perfectly align spatial tracking data with official FIFA Golden Boot records.
2. **XGBoost Classifier:** Maps highly non-linear threshold steps across engineered spatial geometries (Pythagorean distance to goal, absolute lateral deviation, and binary header status).
3. **SHAP Interpretability:** Explains the underlying mathematics of the 'black box' model via TreeExplainer, mathematically proving that high distance vectors and aerial attempts apply heavy negative pressure on Expected Goals.
4. **Interactive Dashboard:** Features a dynamic pitch visualization where shot marker sizes automatically scale based on the underlying danger of the shot (`xg_model * 1200 + 40`).
5. **AI Automated Scouting:** Compiles the player's underlying array values (total shots, actual goals, accumulated model xG, and Finishing Efficiency Delta) into a JSON payload and dynamically queries Claude to generate professional tactical evaluations.

## 💻 How to Run Locally
1. Clone this repository.
2. Install the required analytics stack: 
   `pip install pandas scikit-learn xgboost matplotlib statsbombpy mplsoccer streamlit anthropic python-dotenv`
3. Create a `.env` file in the root directory and add your Anthropic API Key: 
   `ANTHROPIC_API_KEY=your-actual-api-key`
4. Launch the application:
   `streamlit run app.py`