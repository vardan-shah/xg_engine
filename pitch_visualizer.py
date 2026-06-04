import matplotlib.pyplot as plt
from mplsoccer import Pitch
from data_loader import load_world_cup_shots

def plot_player_shots(player_name="Lionel Andrés Messi Cuccittini"):
    # 1. Load our cleaned 2022 World Cup data
    df = load_world_cup_shots()
    
    # 2. Filter for only the requested player's shots
    player_df = df[df['player'] == player_name]
    print(f"\nFound {len(player_df)} shots for {player_name}.")
    
    # 3. Separate the data into Goals and Misses so we can color-code them
    goals = player_df[player_df['is_goal'] == 1]
    misses = player_df[player_df['is_goal'] == 0]
    
    # 4. Draw a dark-mode StatsBomb pitch using mplsoccer
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor('#22312b') 
    
    # 5. Scatter plot the shots onto the pitch
    # Goals = Bright Green Stars
    pitch.scatter(goals.x, goals.y, s=250, marker='*', c='#00ff00', edgecolors='black', ax=ax, label='Goal')
    # Misses/Saves = Red Circles
    pitch.scatter(misses.x, misses.y, s=100, marker='o', c='#ff0000', edgecolors='black', alpha=0.6, ax=ax, label='Miss/Save')
    
    # 6. Add professional titling
    plt.title(f"{player_name} - 2022 World Cup Shots", color='white', fontsize=16, fontweight='bold')
    ax.legend(facecolor='#22312b', edgecolor='none', labelcolor='white', loc='upper left')
    
    # 7. Save the high-res image and display it
    plt.savefig('messi_shots.png', dpi=300, bbox_inches='tight')
    print("\n✅ Success! Pitch visualization saved as 'messi_shots.png'")
    
    plt.show()

if __name__ == "__main__":
    plot_player_shots()