import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# In a real scenario, you'd read from a CSV, but here we'll create a DataFrame from the provided data
# For demonstration, I'll use a subset of the data you provided

# Let's assume the data is loaded into a DataFrame called df
# df = pd.read_csv('F1_prediction.csv')
# Since we don't have the full CSV, I'll create a small example with the data you provided

# Create a sample DataFrame from the given data
data = pd.read_csv('F1_final.csv')
df = data.copy()

print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Check data types and basic information
print("\nData info:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Make a copy for feature engineering
df_engineered = df.copy()

# 1. Fix encoding issues in driver names
# Some driver names have encoding issues (like RÃ¤ikkÃ¶nen instead of Räikkönen)
# This may need manual cleaning in a real application

# 2. Handle the Pitstop Time column - there's a specific value 43.77181752 that appears to be a placeholder 
# for missing or invalid data. Let's replace it with NaN and then handle it
df_engineered['Pitstop Time'] = df_engineered['Pitstop Time'].replace(43.77181752, np.nan)

# Calculate median pitstop time for each team to fill missing values more accurately
team_median_pitstop = df_engineered.groupby('Team')['Pitstop Time'].median()
df_engineered['Pitstop Time'] = df_engineered.apply(
    lambda row: team_median_pitstop[row['Team']] if pd.isna(row['Pitstop Time']) else row['Pitstop Time'], 
    axis=1
)

# If there are still NaN values after this, fill them with the overall median
overall_median_pitstop = df_engineered['Pitstop Time'].median()
df_engineered['Pitstop Time'].fillna(overall_median_pitstop, inplace=True)

# 3. Create new features

# 3.1 Starting position relative features
df_engineered['Position_Gain'] = df_engineered['Starting Grid'] - df_engineered['Position']
df_engineered['Position_Gain_Percentage'] = (df_engineered['Position_Gain'] / df_engineered['Starting Grid']) * 100

# Starting grid position categories
df_engineered['Front_Row'] = (df_engineered['Starting Grid'] <= 2).astype(int)
df_engineered['Top_5_Start'] = (df_engineered['Starting Grid'] <= 5).astype(int)
df_engineered['Top_10_Start'] = (df_engineered['Starting Grid'] <= 10).astype(int)
df_engineered['Back_Grid'] = (df_engineered['Starting Grid'] > 15).astype(int)

# 3.2 Driver-Track history
# For each driver-track combination, calculate historical performance
driver_track_history = df_engineered.groupby(['Driver', 'Track'])['Position'].agg(['mean', 'count']).reset_index()
driver_track_history.columns = ['Driver', 'Track', 'Driver_Track_Avg_Position', 'Driver_Track_Experience']

# Merge these statistics back to the dataframe
df_engineered = pd.merge(df_engineered, driver_track_history, on=['Driver', 'Track'], how='left')

# 3.3 Driver performance metrics
driver_performance = df_engineered.groupby('Driver')['Position'].agg(['mean', 'std', 'median']).reset_index()
driver_performance.columns = ['Driver', 'Driver_Avg_Position', 'Driver_Position_Std', 'Driver_Median_Position']
df_engineered = pd.merge(df_engineered, driver_performance, on='Driver', how='left')

# 3.4 Team performance metrics by year
team_year_performance = df_engineered.groupby(['Team', 'Year'])['Position'].mean().reset_index()
team_year_performance.columns = ['Team', 'Year', 'Team_Year_Avg_Position']
df_engineered = pd.merge(df_engineered, team_year_performance, on=['Team', 'Year'], how='left')

# 3.5 Track-specific overtaking difficulty
track_overtaking = df_engineered.groupby('Track')['Position_Gain'].agg(['mean', 'std']).reset_index()
track_overtaking.columns = ['Track', 'Track_Avg_Position_Gain', 'Track_Position_Gain_Std']
df_engineered = pd.merge(df_engineered, track_overtaking, on='Track', how='left')

# 3.6 Weather-specific performance
# Weather is already encoded as 0 or 1, but let's create derived features
df_engineered['Weather_Label'] = df_engineered['Weather'].map({0: 'Wet', 1: 'Dry'})

# Calculate driver performance in different weather conditions
driver_weather_performance = df_engineered.groupby(['Driver', 'Weather'])['Position'].mean().reset_index()
driver_weather_performance.columns = ['Driver', 'Weather', 'Driver_Weather_Avg_Position']
df_engineered = pd.merge(df_engineered, driver_weather_performance, on=['Driver', 'Weather'], how='left')

# 3.7 Pitstop performance relative to competition
race_avg_pitstop = df_engineered.groupby(['Year', 'Track'])['Pitstop Time'].mean().reset_index()
race_avg_pitstop.columns = ['Year', 'Track', 'Race_Avg_Pitstop']
df_engineered = pd.merge(df_engineered, race_avg_pitstop, on=['Year', 'Track'], how='left')
df_engineered['Pitstop_Delta'] = df_engineered['Pitstop Time'] - df_engineered['Race_Avg_Pitstop']

# 3.8 Driver-Team synergy
driver_team_synergy = df_engineered.groupby(['Driver', 'Team'])['Position'].agg(['mean', 'count']).reset_index()
driver_team_synergy.columns = ['Driver', 'Team', 'Driver_Team_Avg_Position', 'Driver_Team_Experience']
df_engineered = pd.merge(df_engineered, driver_team_synergy, on=['Driver', 'Team'], how='left')

# 3.9 Season performance trend
# Sort by Year and create a performance index for each driver
df_engineered = df_engineered.sort_values(['Year', 'Track', 'Driver'])
df_engineered['Season_Race_Number'] = df_engineered.groupby(['Year', 'Driver']).cumcount() + 1

# Calculate rolling average performance (last 3 races)
df_engineered['Last3_Avg_Position'] = df_engineered.groupby('Driver')['Position'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# 3.10 Points-per-finish ratio (driver's ability to score points when finishing)
points_per_finish = df_engineered.groupby('Driver').apply(
    lambda x: np.sum(x['Points']) / len(x) if len(x) > 0 else 0
).reset_index()
points_per_finish.columns = ['Driver', 'Points_Per_Race']
df_engineered = pd.merge(df_engineered, points_per_finish, on='Driver', how='left')

# 3.11 Qualifying performance - gap between starting position and historical average
df_engineered['Qualifying_Vs_Avg'] = df_engineered['Starting Grid'] - df_engineered['Driver_Avg_Position']

# 3.12 Create interaction features
df_engineered['Driver_Team_Confidence'] = df_engineered['Driver_Team_Avg_Position'] * df_engineered['Team confidence']
df_engineered['Weather_Team_Factor'] = df_engineered['Team confidence'] * (2 * df_engineered['Weather'] - 1)  # Higher values for good teams in dry conditions

# 3.13 Normalize important numerical features
features_to_normalize = [
    'Starting Grid', 'Pitstop Time', 'Driver_Avg_Position', 'Team_Year_Avg_Position',
    'Driver_Track_Avg_Position', 'Driver_Weather_Avg_Position'
]

scaler = StandardScaler()
for feature in features_to_normalize:
    if feature in df_engineered.columns:
        df_engineered[f'{feature}_Scaled'] = scaler.fit_transform(df_engineered[[feature]])

# 3.14 Championship points (running total for the season)
df_engineered['Cumulative_Season_Points'] = df_engineered.groupby(['Year', 'Driver'])['Points'].cumsum()

# 3.15 Competitiveness metrics
df_engineered['Competitive_Edge'] = df_engineered['Team confidence'] * (22 - df_engineered['Driver_Avg_Position']) / 22

# Calculate the number of new features added
original_columns = set(df.columns)
new_columns = set(df_engineered.columns) - original_columns
print(f"\nNumber of new features added: {len(new_columns)}")
print("New features:")
for col in sorted(new_columns):
    print(f"- {col}")

# Export the engineered dataset
df_engineered.to_csv('F1_prediction_engineered.csv', index=False)

print("\nFeature engineering complete. Data exported to 'F1_prediction_engineered.csv'")

# Return the first few rows of the engineered dataset
df_engineered.head()