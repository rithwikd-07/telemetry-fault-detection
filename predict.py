import joblib
import pandas as pd


# Load the trained model
model = joblib.load("models/random_forest.pkl")

print("=== MACHINE FAILURE PREDICTION ===")

try:
    # Get sensor values
    air_temperature = float(input("Air temperature [K]: "))
    process_temperature = float(input("Process temperature [K]: "))
    rotational_speed = float(input("Rotational speed [rpm]: "))
    torque = float(input("Torque [Nm]: "))
    tool_wear = float(input("Tool wear [min]: "))

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get estimated probability
    failure_probability = model.predict_proba(input_data)[0][1]

    print("\n=== PREDICTION RESULT ===")
    print(f"Failure probability estimate: {failure_probability * 100:.2f}%")

    if prediction == 1:
        print("MACHINE FAILURE PREDICTED")
    else:
        print("NORMAL OPERATION")

except ValueError:
    print("\nInvalid input. Please enter numbers only.")