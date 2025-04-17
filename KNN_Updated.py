import pandas as pd
import numpy as np
import math as m
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, max_error

# Load data
data = pd.read_excel("Data.xlsx")

# Input features
input_features = data[["D", "P", "N", "J"]].values

# Target variables: CT and CP
target = data[["CT", "CP"]].values

# Split the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN Regressor model
knn_model = KNeighborsRegressor(n_neighbors=4)

# Apply cross-validation with ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Metrics storage
mse_scores = []
r2_scores = []
abs_scores = []
med_abs_scores = []
max_error_scores = []

# Cross-validation loop
for train_index, val_index in cv.split(X_train_scaled):
    X_train_cv, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val = y_train[train_index], y_train[val_index]
    
    # Fit the model
    knn_model.fit(X_train_cv, y_train_cv)
    
    # Predict on validation set
    y_pred_val = knn_model.predict(X_val)
    
    # Compute metrics
    mse = mean_squared_error(y_val, y_pred_val)
    eval_r2 = r2_score(y_val, y_pred_val)
    eval_abs = mean_absolute_error(y_val, y_pred_val)
    med_abs_error = median_absolute_error(y_val, y_pred_val)
    #max_error_val = max_error(y_val, y_pred_val)
    
    # Store metrics
    mse_scores.append(mse)
    r2_scores.append(eval_r2)
    abs_scores.append(eval_abs)
    med_abs_scores.append(med_abs_error)
    #max_error_scores.append(max_error_val)

# Calculate the average scores over cross-validation
avg_mse = np.mean(mse_scores)
avg_r2 = np.mean(r2_scores)
avg_abs = np.mean(abs_scores)
avg_med_abs = np.mean(med_abs_scores)
#avg_max_error = np.mean(max_error_scores)

# Output cross-validation metrics
print("Cross-Validation Evaluation Metrics:")
print("1. Mean Squared Error:", avg_mse)
print("2. r2 Score:", avg_r2)  # Greater than 0.95 or 95% is best
print("3. Mean Absolute Error:", avg_abs)
print("4. Median Absolute Error:", avg_med_abs)
#print("5. Average Max Error:", avg_max_error)

# Train model on full training set
knn_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_test = knn_model.predict(X_test_scaled)

# Compute test set metrics
mse_test = mean_squared_error(y_test, y_pred_test)
eval_r2_test = r2_score(y_test, y_pred_test)
eval_abs_test = mean_absolute_error(y_test, y_pred_test)
med_abs_error_test = median_absolute_error(y_test, y_pred_test)
#max_error_test = max_error(y_test, y_pred_test)

# Output test set metrics
print("Evaluation Metrics on Test Set:")
print("1. Mean Squared Error:", mse_test)
print("2. r2 Score:", eval_r2_test)  # Greater than 0.95 or 95% is best
print("3. Mean Absolute Error:", eval_abs_test)
print("4. Median Absolute Error:", med_abs_error_test)
#print("5. Max Error:", max_error_test)

# Predict CT and CP for new input
input_diameter = float(input("Enter Required Diameter (in inches): "))
input_pitch = float(input("Enter Required Pitch (in inches): "))
input_RPM = float(input("Enter RPM value of Propeller: "))
input_J = float(input("Enter J value: "))

scaled_input = scaler.transform([[input_diameter, input_pitch, input_RPM, input_J]])
predicted_values = knn_model.predict(scaled_input)

predicted_thrust_constant = predicted_values[0][0]  # CT
predicted_power_coefficient = predicted_values[0][1]  # CP

print("Predicted Thrust Constant (CT):", predicted_thrust_constant)
print("Predicted Power Coefficient (CP):", predicted_power_coefficient)

rad_speed = (input_RPM * 2 * m.pi) / 60
Thrust_g = predicted_thrust_constant * rad_speed * rad_speed
Thrust_N = 0.000980665 * Thrust_g

print("Calculated Thrust (N):", Thrust_N)

import math
Torque_Coeff = predicted_power_coefficient/(2*math.pi)

Torque = (Torque_Coeff/predicted_thrust_constant)*(input_diameter*0.0254*Thrust_N)
print("The Torque Value is(Nm):", Torque)
