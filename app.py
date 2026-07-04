import os
import json
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from dotenv import load_dotenv
from groq import Groq
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
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            st.error("Missing GROQ_API_KEY in your .env file.")

        else:
            with st.spinner("Generating AI scouting report..."):
                try:
                    client = Groq(api_key=api_key)

                    system_prompt = """
You are an elite football scout working for a Champions League club.

Analyze ONLY the supplied statistics.

Return the report in this format:

## Executive Summary

## Finishing Ability

## Shot Selection

## Spatial Tendencies

## Strengths

## Weaknesses

## Tactical Fit

## Development Recommendations

Do not invent statistics.
"""

                    user_prompt = f"""
Player data:

{json.dumps(raw_stats_payload, indent=2)}
"""

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        temperature=0.4,
                        max_tokens=700,
                    )

                    report = response.choices[0].message.content

                    st.markdown(report)

                    st.download_button(
                        "Download Scouting Report",
                        report,
                        file_name=f"{selected_player.replace(' ','_')}_report.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.exception(e)