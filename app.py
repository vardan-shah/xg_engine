import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
# Import your existing data loader!
from data_loader import load_world_cup_shots

# 1. Page Configuration
st.set_page_config(page_title="World Cup xG Engine", layout="wide")

# 2. Cache the data so the API doesn't redownload every time you click a button
@st.cache_data
def load_data():
    return load_world_cup_shots()

# 3. Main UI Header
st.title("🏆 FIFA World Cup 2022 - Scouting Dashboard")
st.markdown("Analyze shot locations and Expected Goals for every player in the tournament.")

# 4. Load Data
with st.spinner("Loading StatsBomb Event Data..."):
    df = load_data()

# 5. Sidebar Navigation
st.sidebar.header("Scouting Settings")
players = sorted(df['player'].unique())

# Default to your favorite player if he's in the list!
default_index = players.index("Lionel Andrés Messi Cuccittini") if "Lionel Andrés Messi Cuccittini" in players else 0
selected_player = st.sidebar.selectbox("Search Player:", players, index=default_index)

# 6. Filter Data for the selected player
player_df = df[df['player'] == selected_player]
goals = player_df[player_df['is_goal'] == 1]
misses = player_df[player_df['is_goal'] == 0]

# 7. Dashboard Metrics
st.subheader(f"Shot Map: {selected_player}")
col1, col2 = st.columns(2)
col1.metric("Total Shots", len(player_df))
col2.metric("Actual Goals", len(goals))

# 8. Render the Pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 7))
fig.patch.set_facecolor('#22312b') 

# Scatter plot
pitch.scatter(goals.x, goals.y, s=250, marker='*', c='#00ff00', edgecolors='black', ax=ax, label='Goal')
pitch.scatter(misses.x, misses.y, s=100, marker='o', c='#ff0000', edgecolors='black', alpha=0.6, ax=ax, label='Miss/Save')

ax.legend(facecolor='#22312b', edgecolor='none', labelcolor='white', loc='upper left')

# Send the matplotlib figure to the Streamlit web app
st.pyplot(fig)