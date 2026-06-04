import pandas as pd
from statsbombpy import sb
import warnings

# Suppress harmless API warnings for clean terminal output
warnings.filterwarnings("ignore")

def load_world_cup_shots():
    print("Fetching FIFA World Cup 2022 match list from StatsBomb...")
    
    # Competition 43 is the Men's World Cup, Season 106 is 2022
    matches = sb.matches(competition_id=43, season_id=106)
    match_ids = matches['match_id'].tolist()
    
    all_shots_df = pd.DataFrame()
    
    print(f"Found {len(match_ids)} matches. Downloading event data (this will take 1-2 minutes)...")
    
    # Loop through every single match in the tournament
    for i, match_id in enumerate(match_ids):
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(match_ids)} matches...")
            
        # Get all events for the specific match
        events = sb.events(match_id=match_id)
        
        # We only care about rows where the event type is a 'Shot'
        # ✅ FIX: Exclude Period 5 (Penalty Shootouts) so our data matches official stats
        if 'type' in events.columns and 'Shot' in events['type'].values:
            shots = events[(events['type'] == 'Shot') & (events['period'] != 5)].copy()
            all_shots_df = pd.concat([all_shots_df, shots], ignore_index=True)
            
    print(f"\nExtracted {len(all_shots_df)} raw shots. Cleaning data...")
    
    # 1. Extract X and Y coordinates. StatsBomb stores them as a list [x, y] in the 'location' column.
    all_shots_df['x'] = all_shots_df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    all_shots_df['y'] = all_shots_df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    
    # 2. Select only the columns we need for our ML Feature Matrix
    keep_cols = [
        'match_id', 'player', 'team', 'minute', 
        'x', 'y', 'shot_body_part', 'shot_outcome', 'shot_statsbomb_xg'
    ]
    
    # Filter safely
    existing_cols = [col for col in keep_cols if col in all_shots_df.columns]
    clean_df = all_shots_df[existing_cols].copy()
    
    # 3. Create our binary Target Variable (y): 1 if Goal, 0 if Miss/Save/Block
    clean_df['is_goal'] = (clean_df['shot_outcome'] == 'Goal').astype(int)
    
    # 4. Drop any weird rows with missing coordinate data
    clean_df = clean_df.dropna(subset=['x', 'y'])
    
    return clean_df

if __name__ == "__main__":
    load_world_cup_shots()