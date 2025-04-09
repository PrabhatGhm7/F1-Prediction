import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load and clean the dataset
df = pd.read_csv('F1_prediction_engineered.csv')
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Driver', 'Team', 'Track', 'Country', 'Position'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Enhanced feature engineering
df['Starting_Grid_sq'] = df['Starting Grid'] ** 2  # Polynomial feature for non-linearity
df['Starting_Grid_cu'] = df['Starting Grid'] ** 3  # Further emphasize starting grid
df['Weather_Starting_Grid'] = df['Weather'] * df['Starting Grid']  # Interaction term

# Define feature columns
feature_columns = [
    "Track_encoded", "Country encoded", "Driver encoded", "Team encoded",
    "Starting Grid", "Starting_Grid_sq", "Starting_Grid_cu", "Pitstop Time",
    "Weather", "Weather_Starting_Grid", "Team confidence", "Position_Gain",
    "Position_Gain_Percentage", "Front_Row", "Top_5_Start", "Top_10_Start", "Back_Grid",
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

# Prepare features and target
X = df[feature_columns].apply(pd.to_numeric, errors='coerce')
y = df['Position'].astype(float)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
joblib.dump(imputer, 'nn_imputer.pkl')  # Save imputer

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'nn_scaler.pkl')  # Save scaler

# Define improved neural network model
model = Sequential([
    Dense(256, input_shape=(X_train_scaled.shape[1],)),  # Increased neurons
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),  # Higher dropout to prevent overfitting
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    Dense(1)  # Linear output for regression
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_f1_nn_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=12,  # Smaller batch size for better generalization
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# Save the final model
model.save('f1_neural_network_model.h5')

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('Training and Validation Loss.png')

plt.show()

import seaborn as sns


# Flatten y_test and y_pred for plotting
y_test_flat = y_test.ravel()
y_pred_flat = y_pred.ravel()

# Evaluate the model
mse = mean_squared_error(y_test_flat, y_pred_flat)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

# Print evaluation metrics
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Set Seaborn style
sns.set_style(style="whitegrid")

# 1. Training & Validation Loss
plt.figure(figsize=(10, 6))
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], marker='o', label='Training Loss', color='royalblue')
plt.plot(epochs, history.history['val_loss'], marker='o', label='Validation Loss', color='darkorange')
plt.title('Training vs Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_loss.png')
plt.show()

# 2. Actual vs Predicted Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_flat, y=y_pred_flat, color='mediumseagreen')
plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# 3. Residuals Distribution
residuals = y_test_flat - y_pred_flat
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='tomato')
plt.axvline(0, color='black', linestyle='--')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('residuals_distribution.png')
plt.show()

# 4. Highlighted Loss Curve with Best Epoch
plt.figure(figsize=(10, 6))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
min_val_epoch = val_loss.index(min(val_loss)) + 1
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', marker='o')
plt.axvline(min_val_epoch, linestyle='--', color='green', alpha=0.6, label=f'Best Epoch: {min_val_epoch}')
plt.title('Loss over Epochs with Best Epoch Highlighted')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_with_best_epoch.png')
plt.show()

# 5. Residuals vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_flat, y=residuals, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png')
plt.show()


