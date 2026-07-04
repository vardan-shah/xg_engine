# 🏆 Expected Goals (xG) Engine & AI Scout

An AI-powered football analytics platform that predicts **Expected Goals (xG)** from FIFA World Cup 2022 shot events using **XGBoost**, visualizes player shooting patterns with interactive spatial analytics, and automatically generates professional scouting reports using **Groq's Llama 3.3** large language model.

---

# 📌 Project Overview

This project combines **machine learning**, **sports analytics**, **data visualization**, and **generative AI** into a complete football scouting workflow.

Using StatsBomb Open Data, the application engineers spatial shooting features, trains an Expected Goals (xG) model with XGBoost, visualizes player shot distributions on an interactive football pitch, and produces AI-generated scouting reports based on player performance metrics.

The application is built as an interactive Streamlit dashboard suitable for exploratory analysis and football scouting.

---

# ⚙️ Tech Stack

### Data & Feature Engineering
- StatsBomb Open Data (`statsbombpy`)
- Pandas
- NumPy

### Machine Learning
- XGBoost
- Scikit-learn

### Model Explainability
- SHAP (SHapley Additive exPlanations)

### Visualization
- Streamlit
- Matplotlib
- mplsoccer

### AI Integration
- Groq API
- Llama 3.3 70B Versatile
- python-dotenv

---

# 🚀 Features

## Expected Goals (xG) Model

- Engineers football-specific spatial features:
  - Distance to goal
  - Shot angle
  - Header indicator
- Trains an XGBoost classifier on FIFA World Cup shot events
- Generates xG probabilities for every shot
- Uses a stratified train/test split for evaluation

---

## Interactive Dashboard

The Streamlit application provides:

- Player selection
- Total shots
- Goals scored
- Accumulated xG
- Finishing Efficiency (Goals − xG)
- Interactive spatial shot map
- xG-scaled shot markers

---

## AI Scouting Engine

Automatically generates professional scouting reports using Groq's Llama 3.3 model.

Each report includes:

- Executive Summary
- Finishing Ability
- Shot Selection
- Spatial Tendencies
- Strengths
- Weaknesses
- Tactical Fit
- Development Recommendations

The AI analysis is generated directly from structured player performance metrics rather than manually written observations.

---

# 📊 Machine Learning Pipeline

1. Load FIFA World Cup 2022 shot events from StatsBomb Open Data.
2. Engineer spatial shooting features.
3. Train an XGBoost Expected Goals classifier.
4. Predict xG values for every shot.
5. Aggregate player-level metrics.
6. Visualize shooting patterns on a StatsBomb pitch.
7. Generate AI-powered scouting reports using Groq Llama 3.3.

---

# 📁 Project Structure

```text
.
├── app.py
├── data_loader.py
├── pitch_visualizer.py
├── shap_explainer.py
├── xg_model.py
├── README.md
├── .env
└── requirements.txt
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/vardan-shah/xg_engine.git
cd xg_engine
```

Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib mplsoccer statsbombpy streamlit shap groq python-dotenv
```

Create a `.env` file

```text
GROQ_API_KEY=your_api_key_here
```

Run the application

```bash
streamlit run app.py
```

---

# 📂 Dataset

- **Competition:** FIFA World Cup Qatar 2022
- **Source:** StatsBomb Open Data

The dataset contains shot event information used to engineer spatial features and train the Expected Goals model.

---

# 🔮 Future Improvements

- Player percentile rankings
- Radar charts
- Team comparison dashboard
- Shot heatmaps
- PDF scouting report export
- Model comparison (XGBoost vs LightGBM vs CatBoost)
- Live match data integration
