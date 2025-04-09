import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load the original dataset to calculate average values for each driver
df = pd.read_csv('F1_prediction_engineered.csv')

# Convert problematic columns to numeric, coercing errors to NaN
numeric_columns = [
    'Pitstop Time', 'Team confidence', 'Position_Gain', 'Position_Gain_Percentage',
    'Driver_Track_Avg_Position', 'Driver_Track_Experience', 'Driver_Avg_Position', 
    'Driver_Position_Std', 'Driver_Median_Position', 'Team_Year_Avg_Position',
    'Track_Avg_Position_Gain', 'Track_Position_Gain_Std', 'Driver_Weather_Avg_Position',
    'Race_Avg_Pitstop', 'Pitstop_Delta', 'Driver_Team_Avg_Position', 'Driver_Team_Experience',
    'Season_Race_Number', 'Last3_Avg_Position', 'Points_Per_Race', 'Qualifying_Vs_Avg',
    'Driver_Team_Confidence', 'Weather_Team_Factor', 'Starting Grid_Scaled', 'Pitstop Time_Scaled',
    'Driver_Avg_Position_Scaled', 'Team_Year_Avg_Position_Scaled', 'Driver_Track_Avg_Position_Scaled',
    'Driver_Weather_Avg_Position_Scaled', 'Cumulative_Season_Points', 'Competitive_Edge'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Load the saved models and scaler
rf_model = joblib.load('model/rf_best_model.pkl')
dt_model = joblib.load('model/dt_best_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Driver and track encodings
driver_encoding = {
    'Max Verstappen': 69, 'Sergio Pérez': 105, 'Charles Leclerc': 13, 
    'Carlos Sainz': 12, 'Lando Norris': 57, 'George Russell': 30, 
    'Lewis Hamilton': 58, 'Esteban Ocon': 24, 'Fernando Alonso': 27, 
    'Pierre Gasly': 90, 'Valtteri Bottas': 114, 'Zhou Guanyu': 35, 
    'Sebastian Vettel': 103, 'Daniel Ricciardo': 18, 'Lance Stroll': 56, 
    'Nicholas Latifi': 76,  
    'Yuki Tsunoda': 119,  'Oscar Piastri': 84, 
    'Logan Sargeant': 60, 
}

track_encoding = {
    'Albert Park Grand Prix Circuit': 0, 'Autodromo Enzo e Dino Ferrari': 1,
    'Autodromo Nazionale di Monza': 3, 'Circuit Gilles Villeneuve': 10,
    'Circuit de Barcelona-Catalunya': 13, 'Circuit de Monaco': 14,
    'Circuit de Spa-Francorchamps': 16, 'Hockenheimring': 19,
    'Hungaroring': 20, 'Indianapolis Motor Speedway': 21,
    'Nürburgring': 29, 'Red Bull Ring': 30,
    'Sepang International Circuit': 31, 'Silverstone Circuit': 33,
    'Suzuka Circuit': 35, 'Bahrain International Circuit': 7,
    'Shanghai International Circuit': 32, 'Istanbul Park': 22,
    'Marina Bay Street Circuit': 27, 'Yas Marina Circuit': 37,
    'Circuit of the Americas': 17, 'Sochi Autodrom': 34,
    'Autódromo Hermanos Rodríguez': 4, 'Baku City Circuit': 8,
    'Circuit Paul Ricard': 12, 'Autodromo Internazionale del Mugello': 2,
    'Jeddah Corniche Circuit': 23, 'Miami International Autodrome': 28,
    'Las Vegas Strip Street Circuit': 25
}

# Reverse mappings for decoding
driver_decoding = {v: k for k, v in driver_encoding.items()}
track_decoding = {v: k for k, v in track_encoding.items()}

# Safe mean function that handles NaN values
def safe_mean(series):
    return series.dropna().mean() if not series.dropna().empty else 0

# Safe mode function that handles empty series
def safe_mode(series):
    try:
        return series.mode()[0] if not series.empty else 0
    except IndexError:
        return 0

# Precompute average values for each driver
def precompute_driver_averages():
    driver_averages = {}
    
    for driver_name, driver_enc in driver_encoding.items():
        driver_data = df[df['Driver encoded'] == driver_enc]
        
        if len(driver_data) > 0:
            # Calculate average values for this driver
            avg_values = {
                "Pitstop Time": safe_mean(driver_data['Pitstop Time']),
                "Team confidence": safe_mean(driver_data['Team confidence']),
                "Position_Gain": safe_mean(driver_data['Position_Gain']),
                "Position_Gain_Percentage": safe_mean(driver_data['Position_Gain_Percentage']),
                "Driver_Track_Avg_Position": safe_mean(driver_data['Driver_Track_Avg_Position']),
                "Driver_Track_Experience": safe_mean(driver_data['Driver_Track_Experience']),
                "Driver_Avg_Position": safe_mean(driver_data['Driver_Avg_Position']),
                "Driver_Position_Std": safe_mean(driver_data['Driver_Position_Std']),
                "Driver_Median_Position": safe_mean(driver_data['Driver_Median_Position']),
                "Team_Year_Avg_Position": safe_mean(driver_data['Team_Year_Avg_Position']),
                "Driver_Weather_Avg_Position": safe_mean(driver_data['Driver_Weather_Avg_Position']),
                "Race_Avg_Pitstop": safe_mean(driver_data['Race_Avg_Pitstop']),
                "Pitstop_Delta": safe_mean(driver_data['Pitstop_Delta']),
                "Driver_Team_Avg_Position": safe_mean(driver_data['Driver_Team_Avg_Position']),
                "Driver_Team_Experience": safe_mean(driver_data['Driver_Team_Experience']),
                "Season_Race_Number": safe_mean(driver_data['Season_Race_Number']),
                "Last3_Avg_Position": safe_mean(driver_data['Last3_Avg_Position']),
                "Points_Per_Race": safe_mean(driver_data['Points_Per_Race']),
                "Qualifying_Vs_Avg": safe_mean(driver_data['Qualifying_Vs_Avg']),
                "Driver_Team_Confidence": safe_mean(driver_data['Driver_Team_Confidence']),
                "Weather_Team_Factor": safe_mean(driver_data['Weather_Team_Factor']),
                "Starting Grid_Scaled": safe_mean(driver_data['Starting Grid_Scaled']),
                "Pitstop Time_Scaled": safe_mean(driver_data['Pitstop Time_Scaled']),
                "Driver_Avg_Position_Scaled": safe_mean(driver_data['Driver_Avg_Position_Scaled']),
                "Team_Year_Avg_Position_Scaled": safe_mean(driver_data['Team_Year_Avg_Position_Scaled']),
                "Driver_Track_Avg_Position_Scaled": safe_mean(driver_data['Driver_Track_Avg_Position_Scaled']),
                "Driver_Weather_Avg_Position_Scaled": safe_mean(driver_data['Driver_Weather_Avg_Position_Scaled']),
                "Cumulative_Season_Points": safe_mean(driver_data['Cumulative_Season_Points']),
                "Competitive_Edge": safe_mean(driver_data['Competitive_Edge']),
                "Team encoded": safe_mode(driver_data['Team encoded']),
                "Country encoded": safe_mode(driver_data['Country encoded'])
            }
            
            # Store the averages for this driver
            driver_averages[driver_enc] = avg_values
        else:
            # If driver not found in the dataset, use global averages
            avg_values = {
                "Pitstop Time": safe_mean(df['Pitstop Time']),
                "Team confidence": safe_mean(df['Team confidence']),
                "Position_Gain": safe_mean(df['Position_Gain']),
                "Position_Gain_Percentage": safe_mean(df['Position_Gain_Percentage']),
                "Driver_Track_Avg_Position": safe_mean(df['Driver_Track_Avg_Position']),
                "Driver_Track_Experience": safe_mean(df['Driver_Track_Experience']),
                "Driver_Avg_Position": safe_mean(df['Driver_Avg_Position']),
                "Driver_Position_Std": safe_mean(df['Driver_Position_Std']),
                "Driver_Median_Position": safe_mean(df['Driver_Median_Position']),
                "Team_Year_Avg_Position": safe_mean(df['Team_Year_Avg_Position']),
                "Driver_Weather_Avg_Position": safe_mean(df['Driver_Weather_Avg_Position']),
                "Race_Avg_Pitstop": safe_mean(df['Race_Avg_Pitstop']),
                "Pitstop_Delta": safe_mean(df['Pitstop_Delta']),
                "Driver_Team_Avg_Position": safe_mean(df['Driver_Team_Avg_Position']),
                "Driver_Team_Experience": safe_mean(df['Driver_Team_Experience']),
                "Season_Race_Number": safe_mean(df['Season_Race_Number']),
                "Last3_Avg_Position": safe_mean(df['Last3_Avg_Position']),
                "Points_Per_Race": safe_mean(df['Points_Per_Race']),
                "Qualifying_Vs_Avg": safe_mean(df['Qualifying_Vs_Avg']),
                "Driver_Team_Confidence": safe_mean(df['Driver_Team_Confidence']),
                "Weather_Team_Factor": safe_mean(df['Weather_Team_Factor']),
                "Starting Grid_Scaled": safe_mean(df['Starting Grid_Scaled']),
                "Pitstop Time_Scaled": safe_mean(df['Pitstop Time_Scaled']),
                "Driver_Avg_Position_Scaled": safe_mean(df['Driver_Avg_Position_Scaled']),
                "Team_Year_Avg_Position_Scaled": safe_mean(df['Team_Year_Avg_Position_Scaled']),
                "Driver_Track_Avg_Position_Scaled": safe_mean(df['Driver_Track_Avg_Position_Scaled']),
                "Driver_Weather_Avg_Position_Scaled": safe_mean(df['Driver_Weather_Avg_Position_Scaled']),
                "Cumulative_Season_Points": safe_mean(df['Cumulative_Season_Points']),
                "Competitive_Edge": safe_mean(df['Competitive_Edge']),
                "Team encoded": safe_mode(df['Team encoded']),
                "Country encoded": safe_mode(df['Country encoded'])
            }
            driver_averages[driver_enc] = avg_values
            
    return driver_averages

# Precompute track average values
def precompute_track_averages():
    track_averages = {}
    
    for track_name, track_enc in track_encoding.items():
        track_data = df[df['Track_encoded'] == track_enc]
        
        if len(track_data) > 0:
            # Calculate average values for this track
            avg_values = {
                "Track_Avg_Position_Gain": safe_mean(track_data['Track_Avg_Position_Gain']),
                "Track_Position_Gain_Std": safe_mean(track_data['Track_Position_Gain_Std']),
            }
            
            # Store the averages for this track
            track_averages[track_enc] = avg_values
        else:
            # If track not found in the dataset, use global averages
            avg_values = {
                "Track_Avg_Position_Gain": safe_mean(df['Track_Avg_Position_Gain']),
                "Track_Position_Gain_Std": safe_mean(df['Track_Position_Gain_Std']),
            }
            track_averages[track_enc] = avg_values
            
    return track_averages

# Precompute the averages
driver_averages = precompute_driver_averages()
track_averages = precompute_track_averages()

# Function to calculate derived features from starting grid position
def calculate_grid_features(starting_grid):
    front_row = 1 if starting_grid <= 2 else 0
    top_5_start = 1 if starting_grid <= 5 else 0
    top_10_start = 1 if starting_grid <= 10 else 0
    back_grid = 1 if starting_grid > 15 else 0
    
    return front_row, top_5_start, top_10_start, back_grid

# Handle potential issues with model prediction
def safe_predict(model, input_data):
    try:
        return model.predict(input_data)[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return 1  # Default to position 1 if prediction fails

# Handle potential issues with prediction probabilities
def safe_predict_proba(model, input_data):
    try:
        predic =  model.predict_proba(input_data)[0]
        predic -= 1
        return predic
    except Exception as e:
        print(f"Probability prediction error: {e}")
        # Return a simple array with highest probability for position 1
        proba = np.zeros(20)  # Assuming max 20 positions
        proba[0] = 1.0


    return proba
# Prediction function
def predict_position(driver_name, track_name, starting_grid, weather, model_choice):
    # Get encoded values
    driver_enc = driver_encoding[driver_name]
    track_enc = track_encoding[track_name]
    
    # Calculate grid-based features
    front_row, top_5_start, top_10_start, back_grid = calculate_grid_features(starting_grid)
    
    # Get driver and track averages
    driver_avg = driver_averages[driver_enc]
    track_avg = track_averages[track_enc]
    
    # Create feature vector
    feature_vector = {
        "Track_encoded": track_enc,
        "Country encoded": driver_avg["Country encoded"],
        "Driver encoded": driver_enc,
        "Team encoded": driver_avg["Team encoded"],
        "Starting Grid": starting_grid,
        "Pitstop Time": driver_avg["Pitstop Time"],
        "Weather": weather,
        "Team confidence": driver_avg["Team confidence"],
        "Position_Gain": driver_avg["Position_Gain"],
        "Position_Gain_Percentage": driver_avg["Position_Gain_Percentage"],
        "Front_Row": front_row,
        "Top_5_Start": top_5_start,
        "Top_10_Start": top_10_start,
        "Back_Grid": back_grid,
        "Driver_Track_Avg_Position": driver_avg["Driver_Track_Avg_Position"],
        "Driver_Track_Experience": driver_avg["Driver_Track_Experience"],
        "Driver_Avg_Position": driver_avg["Driver_Avg_Position"],
        "Driver_Position_Std": driver_avg["Driver_Position_Std"],
        "Driver_Median_Position": driver_avg["Driver_Median_Position"],
        "Team_Year_Avg_Position": driver_avg["Team_Year_Avg_Position"],
        "Track_Avg_Position_Gain": track_avg["Track_Avg_Position_Gain"],
        "Track_Position_Gain_Std": track_avg["Track_Position_Gain_Std"],
        "Driver_Weather_Avg_Position": driver_avg["Driver_Weather_Avg_Position"],
        "Race_Avg_Pitstop": driver_avg["Race_Avg_Pitstop"],
        "Pitstop_Delta": driver_avg["Pitstop_Delta"],
        "Driver_Team_Avg_Position": driver_avg["Driver_Team_Avg_Position"],
        "Driver_Team_Experience": driver_avg["Driver_Team_Experience"],
        "Season_Race_Number": driver_avg["Season_Race_Number"],
        "Last3_Avg_Position": driver_avg["Last3_Avg_Position"],
        "Points_Per_Race": driver_avg["Points_Per_Race"],
        "Qualifying_Vs_Avg": driver_avg["Qualifying_Vs_Avg"],
        "Driver_Team_Confidence": driver_avg["Driver_Team_Confidence"],
        "Weather_Team_Factor": driver_avg["Weather_Team_Factor"],
        "Starting Grid_Scaled": driver_avg["Starting Grid_Scaled"],
        "Pitstop Time_Scaled": driver_avg["Pitstop Time_Scaled"],
        "Driver_Avg_Position_Scaled": driver_avg["Driver_Avg_Position_Scaled"],
        "Team_Year_Avg_Position_Scaled": driver_avg["Team_Year_Avg_Position_Scaled"],
        "Driver_Track_Avg_Position_Scaled": driver_avg["Driver_Track_Avg_Position_Scaled"],
        "Driver_Weather_Avg_Position_Scaled": driver_avg["Driver_Weather_Avg_Position_Scaled"],
        "Cumulative_Season_Points": driver_avg["Cumulative_Season_Points"],
        "Competitive_Edge": driver_avg["Competitive_Edge"]
    }
    
    # Convert to DataFrame for consistency with model input
    input_df = pd.DataFrame([feature_vector])
    
    # Scale the features
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        print(f"Scaling error: {e}")
        # If scaling fails, use unscaled data
        input_scaled = input_df.values
    
    # Make prediction
    if model_choice == "Random Forest":
        prediction = safe_predict(rf_model, input_scaled)
    else:  # Decision Tree
        prediction = safe_predict(dt_model, input_scaled)
 
   
    
    # Format the output
    prediction_text = f"Predicted finish position: {int(prediction-1)}\n\n"
    
   
    
    # Add driver and track information
    prediction_text += f"\nDriver: {driver_name}\n"
    prediction_text += f"Track: {track_name}\n"
    prediction_text += f"Starting Position: {starting_grid}\n"
    prediction_text += f"Weather: {'Wet' if weather == 0 else 'Dry'}\n"
    
    return prediction_text
def create_interface():
    with gr.Blocks(theme="glass",title="F1 Race Position Predictor") as interface:

        # Main title
        gr.Markdown("# Formula 1 Race Position Predictor")
        gr.Markdown("Select a driver, track, starting position, and weather conditions to predict the finishing position.")
        
        # Gradio Row layout for alignment
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                driver_dropdown = gr.Dropdown(
                    choices=list(driver_encoding.keys()),
                    label="Select Driver",
                    value="",
                )
                
                track_dropdown = gr.Dropdown(
                    choices=list(track_encoding.keys()),
                    label="Select Track",
                    value="",
                )
                
                starting_grid = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=1,
                    label="Starting Grid Position",
                )
                
                weather = gr.Radio(
                    choices=[0, 1],
                    value=1,
                    label="Weather Conditions",
                    info="(0: Wet, 1: Dry)"
                )
                
                model_choice = gr.Radio(
                    choices=["Random Forest", "Decision Tree"],
                    value="Random Forest",
                    label="Prediction Model",
                )
                
                predict_button = gr.Button("Predict Finish Position", elem_id="predict_button")
            
            # Styled column for output
            with gr.Column(scale=1, min_width=300):
                output_text = gr.Textbox(
                    label="Prediction Results",
                    lines=10,
                    interactive=False,
                    placeholder="The predicted finish position will be displayed here."
                )
        
        # Action on button click
        predict_button.click(
            fn=predict_position,
            inputs=[driver_dropdown, track_dropdown, starting_grid, weather, model_choice],
            outputs=output_text
        )
    
    return interface






# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()