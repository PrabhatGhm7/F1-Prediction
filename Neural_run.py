import pandas as pd
import numpy as np
import joblib
import gradio as gr
from tensorflow.keras.models import load_model

# Load dataset for averages
df = pd.read_csv('F1_prediction_engineered.csv')
numeric_columns = [
    "Pitstop Time", "Team confidence", "Position_Gain", "Position_Gain_Percentage",
    "Front_Row", "Top_5_Start", "Top_10_Start", "Back_Grid",
    "Driver_Track_Avg_Position", "Driver_Track_Experience", "Driver_Avg_Position",
    "Driver_Position_Std", "Driver_Median_Position", "Team_Year_Avg_Position",
    "Track_Avg_Position_Gain", "Track_Position_Gain_Std", "Driver_Weather_Avg_Position",
    "Race_Avg_Pitstop", "Pitstop_Delta", "Driver_Team_Avg_Position", "Driver_Team_Experience",
    "Season_Race_Number", "Last3_Avg_Position", "Points_Per_Race", "Qualifying_Vs_Avg",
    "Driver_Team_Confidence", "Weather_Team_Factor", "Starting Grid_Scaled",
    "Pitstop Time_Scaled", "Driver_Avg_Position_Scaled", "Team_Year_Avg_Position_Scaled",
    "Driver_Track_Avg_Position_Scaled", "Driver_Weather_Avg_Position_Scaled",
    "Cumulative_Season_Points", "Competitive_Edge"
]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Load models and preprocessing tools
nn_model = load_model('f1_neural_network_model.h5')
best_nn_model = load_model('best_f1_nn_model.h5')
imputer = joblib.load('nn_imputer.pkl')
scaler = joblib.load('nn_scaler.pkl')

# Encodings
driver_encoding = {
    'Max Verstappen': 69, 'Sergio Pérez': 105, 'Charles Leclerc': 13,
    'Carlos Sainz': 12, 'Lando Norris': 57, 'George Russell': 30,
    'Lewis Hamilton': 58, 'Esteban Ocon': 24, 'Fernando Alonso': 27,
    'Pierre Gasly': 90, 'Valtteri Bottas': 114, 'Zhou Guanyu': 35,
    'Sebastian Vettel': 103, 'Daniel Ricciardo': 18, 'Lance Stroll': 56,
    'Nicholas Latifi': 76, 'Yuki Tsunoda': 119, 'Oscar Piastri': 84,
    'Logan Sargeant': 60
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

# Precompute averages
def safe_mean(series):
    return series.dropna().mean() if not series.dropna().empty else 0

def safe_mode(series):
    return series.mode()[0] if not series.empty else 0

driver_averages = {
    driver_enc: {
        col: safe_mean(df[df['Driver encoded'] == driver_enc][col]) for col in numeric_columns
    } | {
        "Team encoded": safe_mode(df[df['Driver encoded'] == driver_enc]["Team encoded"]),
        "Country encoded": safe_mode(df[df['Driver encoded'] == driver_enc]["Country encoded"])
    } for driver_enc in driver_encoding.values()
}

track_averages = {
    track_enc: {
        "Track_Avg_Position_Gain": safe_mean(df[df['Track_encoded'] == track_enc]['Track_Avg_Position_Gain']),
        "Track_Position_Gain_Std": safe_mean(df[df['Track_encoded'] == track_enc]['Track_Position_Gain_Std'])
    } for track_enc in track_encoding.values()
}

# Helper functions
def calculate_grid_features(starting_grid):
    return (1 if starting_grid <= 2 else 0, 1 if starting_grid <= 5 else 0,
            1 if starting_grid <= 10 else 0, 1 if starting_grid > 15 else 0)

def safe_predict(model, input_data, driver_name, starting_grid):
    raw_pred = float(model.predict(input_data, verbose=0)[0][0])
    print(f"Raw prediction before adjustment: {raw_pred}")

    # Driver-specific adjustments
    # Elite drivers who often win from pole or top positions
    elite_drivers = [
        "Max Verstappen",
        "Lewis Hamilton",
        "Fernando Alonso",
        "Charles Leclerc"
    ]

    # Strong drivers: Consistent point scorers and regular podium finishers,
    # frequently in contention when conditions are right.
    strong_drivers = [
        "Sergio Pérez",
        "Lando Norris",
        "Carlos Sainz Jr.",
        "George Russell",
        "Pierre Gasly",
        "Esteban Ocon",
        "Oscar Piastri",
        "Yuki Tsunoda",
        "Alexander Albon",
        "Nico Hulkenberg",
    ]

    # Weak drivers: Drivers who, based on the 2020-2024 data, generally
    # score fewer points and have struggled to compete at the upper levels.
    weak_drivers = [
        "Lance Stroll",
        "Daniel Ricciardo",
        "Zhou Guanyu",
        "Kevin Magnussen",
        "Liam Lawson",
        "Logan Sargeant",
        "Nyck de Vries"
    ]

    
    # Adjust raw prediction based on driver performance group and starting grid.
    if driver_name in elite_drivers:
        if starting_grid <= 2:
            # Elite drivers starting at the very front get a significant boost.
            raw_pred -= 2.5
            print("Adjustment: Elite driver starting in P1/P2; subtracted 2.5 from raw prediction.")
        elif starting_grid <= 5:
            # Less boost if an elite driver starts in P3 to P5.
            raw_pred -= 2.1
            print("Adjustment: Elite driver starting in P3-P5; subtracted 1.5 from raw prediction.")
        elif starting_grid <= 20:
            # Less boost if an elite driver starts in P3 to P5.
            raw_pred -= 2.8
            print("Adjustment: Elite driver starting in P3-P5; subtracted 1.5 from raw prediction.")

    elif driver_name in strong_drivers:
        if starting_grid <= 3:
            # Strong drivers in the top three get a moderate boost.
            raw_pred -= 1.8
            print("Adjustment: Strong driver starting in P1-P3; subtracted 0.7 from raw prediction.")
        elif starting_grid >= 10:
            # If a strong driver has a poor grid, a slight penalty applies.
            raw_pred -= 2.8
            print("Adjustment: Strong driver starting beyond P9; added 0.5 to raw prediction.")
    elif driver_name in weak_drivers:
        if starting_grid <= 5:
            # A weak driver starting very high might get a minor bonus to reflect an over-performance.
            raw_pred += 1.0
            print("Adjustment: Weak driver starting in P1-P5; subtracted 0.5 from raw prediction.")
        elif starting_grid > 10:
            # Weak drivers starting far back get a penalty to reflect lower expected performance.
            raw_pred += 1.5
            print("Adjustment: Weak driver starting beyond P10; added 1.0 to raw prediction.")

    # In all cases, if the starting grid is very poor, apply an additional general penalty.
    if starting_grid > 15:
        raw_pred -= 0.5
        print("Additional adjustment: Starting grid > 15; added 0.5 to raw prediction.")

    # Finalize prediction: round and clamp to valid range (1 to 20).
    prediction = int(round(raw_pred))
    prediction = max(1, min(20, prediction))

    print(f"Adjusted raw prediction: {raw_pred}")
    print(f"Final prediction: {prediction}")
    return prediction

    

# Prediction function
def predict_position(driver_name, track_name, starting_grid, weather, model_choice):
    driver_enc = driver_encoding[driver_name]
    track_enc = track_encoding[track_name]
    front_row, top_5_start, top_10_start, back_grid = calculate_grid_features(starting_grid)
    driver_avg = driver_averages[driver_enc]
    track_avg = track_averages[track_enc]

    feature_vector = {
        "Track_encoded": track_enc, "Country encoded": driver_avg["Country encoded"],
        "Driver encoded": driver_enc, "Team encoded": driver_avg["Team encoded"],
        "Starting Grid": starting_grid, "Starting_Grid_sq": starting_grid ** 2,
        "Starting_Grid_cu": starting_grid ** 3, "Pitstop Time": driver_avg["Pitstop Time"],
        "Weather": weather, "Weather_Starting_Grid": weather * starting_grid,
        "Team confidence": driver_avg["Team confidence"], "Position_Gain": driver_avg["Position_Gain"],
        "Position_Gain_Percentage": driver_avg["Position_Gain_Percentage"], "Front_Row": front_row,
        "Top_5_Start": top_5_start, "Top_10_Start": top_10_start, "Back_Grid": back_grid,
        "Driver_Track_Avg_Position": driver_avg["Driver_Track_Avg_Position"],
        "Driver_Track_Experience": driver_avg["Driver_Track_Experience"],
        "Driver_Avg_Position": driver_avg["Driver_Avg_Position"],
        "Driver_Position_Std": driver_avg["Driver_Position_Std"],
        "Driver_Median_Position": driver_avg["Driver_Median_Position"],
        "Team_Year_Avg_Position": driver_avg["Team_Year_Avg_Position"],
        "Track_Avg_Position_Gain": track_avg["Track_Avg_Position_Gain"],
        "Track_Position_Gain_Std": track_avg["Track_Position_Gain_Std"],
        "Driver_Weather_Avg_Position": driver_avg["Driver_Weather_Avg_Position"],
        "Race_Avg_Pitstop": driver_avg["Race_Avg_Pitstop"], "Pitstop_Delta": driver_avg["Pitstop_Delta"],
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

    input_df = pd.DataFrame([feature_vector])
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    model = nn_model if model_choice == "Neural Network" else best_nn_model
    prediction = safe_predict(model, input_scaled, driver_name, starting_grid)
    return (f"Predicted finish position: {prediction}\n\n"
            f"Driver: {driver_name}\nTrack: {track_name}\nStarting Position: {starting_grid}\n"
            f"Weather: {'Wet' if weather == 0 else 'Dry'}")
    
with gr.Blocks(theme="glass", title="F1 Race Position Predictor") as interface:
    # Header Section
    gr.Markdown("# Formula 1 Race Position Predictor")
    gr.Markdown("Predict a driver's finishing position based on key race factors.")

    # Layout with Row and Columns
    with gr.Row():
        # Input Column
        with gr.Column():
            # Race Parameters Section
            gr.Markdown("### Race Parameters")
            driver_dropdown = gr.Dropdown(
                choices=list(driver_encoding.keys()), 
                label="Driver", 
                value=""
            )
            track_dropdown = gr.Dropdown(
                choices=list(track_encoding.keys()), 
                label="Track", 
                value=""
            )
            starting_grid = gr.Slider(
                minimum=1, 
                maximum=20, 
                step=1, 
                value=1, 
                label="Starting Grid Position"
            )
            weather = gr.Radio(
                choices=[("Dry", 1), ("Wet", 0)], 
                value=1, 
                label="Weather"
            )

            # Model Selection Section
            gr.Markdown("### Model Selection")
            model_choice = gr.Radio(
                choices=["Neural Network", "Best Neural Network"], 
                value="Best Neural Network", 
                label="Model"
            )

            # Predict Button
            predict_button = gr.Button("Predict", variant="primary")

        # Output Column
        with gr.Column():
            output_text = gr.Textbox(label="Prediction", lines=5)

    # Connect Button to Prediction Function
    predict_button.click(
        fn=predict_position, 
        inputs=[driver_dropdown, track_dropdown, starting_grid, weather, model_choice], 
        outputs=output_text
    )

# Launch the interface
interface.launch()