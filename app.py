import os
import json
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic
from xg_model import train_and_predict_all

# Load credentials from .env
load_dotenv()

st.set_page_config(page_title="World Cup xG Engine & AI Scout", layout="wide")

@st.cache_data
def load_and_model_data():
    return train_and_predict_all()

st.title("🏆 FIFA World Cup 2022 - Advanced Scouting & AI Engine")
st.markdown("Analyze spatial data structures, model predictions, and auto-generate AI scouting profiles.")

with st.spinner("Synchronizing StatsBomb tracking fields and processing XGBoost matrices..."):
    df = load_and_model_data()

# Sidebar Setup
st.sidebar.header("Scouting Target")
players = sorted(df['player'].unique())
default_index = players.index("Lionel Andrés Messi Cuccittini") if "Lionel Andrés Messi Cuccittini" in players else 0
selected_player = st.sidebar.selectbox("Select Player:", players, index=default_index)

# Data Isolation
player_df = df[df['player'] == selected_player]
goals = player_df[player_df['is_goal'] == 1]
misses = player_df[player_df['is_goal'] == 0]

# Metrics Computations
total_shots = len(player_df)
actual_goals = len(goals)
total_xg = float(player_df['xg_model'].sum())
goals_above_xg = actual_goals - total_xg

# Layout Metrics Block
st.subheader(f"Performance Analysis: {selected_player}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Shots Taken", total_shots)
m2.metric("Actual Goals Scored", actual_goals)
m3.metric("Accumulated Model xG", f"{total_xg:.2f}")
m4.metric("Finishing Efficiency Delta", f"{goals_above_xg:+.2f}", delta_color="normal")

# Layout Spatial Grid & AI Reporting Split
col_chart, col_ai = st.columns([1.1, 0.9])

with col_chart:
    st.markdown("#### Spatial Shot Map")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#111111', line_color='#444444')
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor('#111111') 
    
    # Scaled sizes based on shot quality
    if not goals.empty:
        g_sizes = goals['xg_model'] * 1200 + 40
        pitch.scatter(goals.x, goals.y, s=g_sizes, marker='*', c='#00ff00', edgecolors='black', ax=ax, label=f'Goals ({actual_goals})')
    if not misses.empty:
        m_sizes = misses['xg_model'] * 1200 + 40
        pitch.scatter(misses.x, misses.y, s=m_sizes, marker='o', c='#ff0000', edgecolors='black', alpha=0.5, ax=ax, label='Misses / Saves')
        
    ax.legend(facecolor='#111111', edgecolor='none', labelcolor='white', loc='upper left')
    st.pyplot(fig)

with col_ai:
    st.markdown("#### AI Scouting Performance Dossier")
    
    # 1. Structure raw telemetry JSON array
    header_shots = int(player_df['is_header'].sum())
    avg_distance = float(player_df['distance_to_goal'].mean()) if total_shots > 0 else 0
    
    raw_stats_payload = {
        "player_name": selected_player,
        "team": player_df['team'].iloc[0] if total_shots > 0 else "Unknown",
        "performance_metrics": {
            "shots": total_shots,
            "goals": actual_goals,
            "expected_goals_xg": round(total_xg, 2),
            "goals_above_xg_delta": round(goals_above_xg, 2),
            "aerial_header_attempts": header_shots,
            "average_shot_distance_yards": round(avg_distance, 1)
        }
    }
    
    with st.expander("🔍 View Raw Prompt Payload"):
        st.json(raw_stats_payload)
        
    if st.button("Generate AI Scouting Brief", type="primary"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or "your-actual-api-key" in api_key:
            st.error("Missing credentials. Please configure your ANTHROPIC_API_KEY in the `.env` file.")
        else:
            with st.spinner("Consulting Chief AI Football Analyst..."):
                try:
                    client = Anthropic(api_key=api_key)
                    
                    system_prompt = (
                        "You are a Senior Football Analytics Director and Chief Scout. "
                        "Analyze the provided player shooting telemetry parsed from our custom XGBoost model. "
                        "Provide a professional, analytical scouting profile using precise sports terminology. "
                        "Focus on shot selection efficiency, finishing mechanics, and spatial tendencies."
                    )
                    
                    user_prompt = f"Analyze this model output payload for scouting evaluation:\n{json.dumps(raw_stats_payload, indent=2)}"
                    
                    # FIX: Using the correct, valid Anthropic model string
                    message = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    
                    report_content = message.content[0].text
                    st.markdown(report_content)
                    
                    st.download_button(
                        label="Download Scouting Report",
                        data=report_content,
                        file_name=f"{selected_player.replace(' ', '_')}_scouting_report.txt",
                        mime="text/plain"
                    )
                    
                # SENIOR FIX: Specific exception handling for clean UI feedback
                except anthropic.AuthenticationError:
                    st.error("API Execution Failure: Authentication Error. Check your API key.")
                except anthropic.RateLimitError:
                    st.error("API Execution Failure: Rate Limit Exceeded or Insufficient Credits.")
                except anthropic.APIConnectionError:
                    st.error("API Execution Failure: Network Connection Error.")
                except Exception as e:
                    st.error(f"Unexpected System Error: {str(e)}")