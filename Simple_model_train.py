import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns



# Load dataset (assuming 'df' is your dataset)
df = pd.read_csv('F1_prediction_engineered.csv')

# Step 1: Preprocess data
# Encode categorical features using LabelEncoder

# Clean column names to avoid any hidden spaces or issues
df.columns = df.columns.str.strip()

# Handle missing values by dropping rows with missing critical columns
df = df.dropna(subset=['Driver', 'Team', 'Track', 'Country'])

# Check for NaN and infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Select features (including the encoded categorical ones) and the target
X = df[[
    "Track_encoded",
    "Country encoded",
    "Driver encoded",
    "Team encoded",
    "Starting Grid",
    "Pitstop Time",
    "Weather",
    "Team confidence",
    "Position_Gain",
    "Position_Gain_Percentage",
    "Front_Row",
    "Top_5_Start",
    "Top_10_Start",
    "Back_Grid",
    "Driver_Track_Avg_Position",
    "Driver_Track_Experience",
    "Driver_Avg_Position",
    "Driver_Position_Std",
    "Driver_Median_Position",
    "Team_Year_Avg_Position",
    "Track_Avg_Position_Gain",
    "Track_Position_Gain_Std",
    "Driver_Weather_Avg_Position",
    "Race_Avg_Pitstop",
    "Pitstop_Delta",
    "Driver_Team_Avg_Position",
    "Driver_Team_Experience",
    "Season_Race_Number",
    "Last3_Avg_Position",
    "Points_Per_Race",
    "Qualifying_Vs_Avg",
    "Driver_Team_Confidence",
    "Weather_Team_Factor",
    "Starting Grid_Scaled",
    "Pitstop Time_Scaled",
    "Driver_Avg_Position_Scaled",
    "Team_Year_Avg_Position_Scaled",
    "Driver_Track_Avg_Position_Scaled",
    "Driver_Weather_Avg_Position_Scaled",
    "Cumulative_Season_Points",
    "Competitive_Edge"
]]

y = df['Position'].astype(float)  # Treat Position as a regression target

# Create a copy of the DataFrame to avoid the SettingWithCopyWarning
X = X.copy()  # This is a temporary step to ensure no warnings when working with the data

# Now proceed with your modifications, using .loc to ensure you're working in place
X.loc[:, 'Starting_Grid'] = X['Starting Grid'] ** 50
X.loc[:, 'Track_encoded'] = X['Track_encoded'] ** 30
X.loc[:, 'Weather'] = X['Weather'] ** 40


# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Check for non-numeric values in X_train
X_train = X_train.apply(pd.to_numeric, errors='coerce')

# Check for non-numeric values in X_test
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Use SimpleImputer to fill missing values with the column's mean
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'

# Fit and transform the training set
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test set (using the same imputer learned from the training set)
X_test_imputed = imputer.transform(X_test)



# Step 3: Normalize the numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model initialization
# Decision Tree with GridSearch
dt_model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
dt_param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(X_train_scaled, y_train)

# Random Forest with GridSearch
rf_model = RandomForestClassifier(random_state=42,class_weight='balanced')
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train_scaled, y_train)

# Step 5: Model evaluation
# Decision Tree
dt_best_model = dt_grid_search.best_estimator_
dt_predictions = dt_best_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(classification_report(y_test, dt_predictions))

# Random Forest
rf_best_model = rf_grid_search.best_estimator_
rf_predictions = rf_best_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")
print(classification_report(y_test, rf_predictions))

# Step 6: Visualize Feature Importance (Random Forest)
plt.figure(figsize=(10, 6))
feature_importances = rf_best_model.feature_importances_
features = X.columns
plt.barh(features, feature_importances)
plt.title('Feature Importance (Random Forest)')
plt.show()


# Step 6: Visualize Feature Importance (Random Forest)
plt.figure(figsize=(10, 6))
feature_importances = dt_best_model.feature_importances_
features = X.columns
plt.barh(features, feature_importances)
plt.title('Feature Importance (Decision Tree)')
plt.show()

# Confusion Matrix for Decision Tree
dt_cm = confusion_matrix(y_test, dt_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix (Decision Tree)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Feature Importance Visualization for Decision Tree
plt.figure(figsize=(10, 6))
dt_feature_importances = dt_best_model.feature_importances_
dt_features = X.columns
sorted_idx_dt = np.argsort(dt_feature_importances)

plt.barh(dt_features[sorted_idx_dt], dt_feature_importances[sorted_idx_dt], color='skyblue')
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Feature Importance Visualization for Random Forest
plt.figure(figsize=(10, 6))
rf_feature_importances = rf_best_model.feature_importances_
rf_features = X.columns
sorted_idx_rf = np.argsort(rf_feature_importances)

plt.barh(rf_features[sorted_idx_rf], rf_feature_importances[sorted_idx_rf], color='lightcoral')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# Plot Accuracy Comparison
model_names = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()



# Save the best Random Forest model, Decision Tree model, and StandardScaler
joblib.dump(rf_best_model, 'rf_best_model.pkl')
joblib.dump(dt_best_model, 'dt_best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Confirm saving
print("Models and scaler saved successfully.")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure seaborn styling (no deprecated usage)
sns.set_theme(style="whitegrid")

# Random Forest Feature Importance (sorted)
plt.figure(figsize=(10, 6))
rf_importances = rf_best_model.feature_importances_
rf_sorted_idx = np.argsort(rf_importances)
plt.barh(X.columns[rf_sorted_idx], rf_importances[rf_sorted_idx], color='salmon')
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_random_forest.png')
plt.show()

# Decision Tree Feature Importance (sorted)
plt.figure(figsize=(10, 6))
dt_importances = dt_best_model.feature_importances_
dt_sorted_idx = np.argsort(dt_importances)
plt.barh(X.columns[dt_sorted_idx], dt_importances[dt_sorted_idx], color='skyblue')
plt.title('Feature Importance (Decision Tree)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_decision_tree.png')
plt.show()

# Confusion Matrix - Decision Tree
dt_cm = confusion_matrix(y_test, dt_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix (Decision Tree)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_decision_tree.png')
plt.show()

# Confusion Matrix - Random Forest
rf_cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix (Random Forest)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_random_forest.png')
plt.show()

# Accuracy Comparison
model_names = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, accuracies, color=['dodgerblue', 'mediumseagreen'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison', fontsize=14)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.show()
