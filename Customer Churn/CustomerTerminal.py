import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("model.sav", "rb"))

# Function to preprocess input data
def preprocess_input(data, feature_names):
    # Convert binary categorical variables to numeric
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
    data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data)

    # Add missing columns if any
    missing_cols = set(feature_names) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # Ensure columns are in the same order as in the training data
    data = data[feature_names]

    return data

# Function to predict churn
def predict_churn(input_data, feature_names):
    # Preprocess input data
    input_data_processed = preprocess_input(input_data, feature_names)
    # Predict churn
    prediction = model.predict(input_data_processed)
    probability = model.predict_proba(input_data_processed)[:,1]
    return prediction, probability

# Get input from the user
def get_input():
    input_data = {}
    input_data['SeniorCitizen'] = input("Enter SeniorCitizen (0 or 1): ")
    input_data['MonthlyCharges'] = input("Enter MonthlyCharges: ")
    input_data['TotalCharges'] = input("Enter TotalCharges: ")
    input_data['gender'] = input("Enter gender (Male or Female): ")
    input_data['Partner'] = input("Enter Partner (Yes or No): ")
    input_data['Dependents'] = input("Enter Dependents (Yes or No): ")
    input_data['PhoneService'] = input("Enter PhoneService (Yes or No): ")
    input_data['MultipleLines'] = input("Enter MultipleLines (Yes, No, or No phone service): ")
    input_data['InternetService'] = input("Enter InternetService (DSL, Fiber optic, or No): ")
    input_data['OnlineSecurity'] = input("Enter OnlineSecurity (Yes, No, or No internet service): ")
    input_data['OnlineBackup'] = input("Enter OnlineBackup (Yes, No, or No internet service): ")
    input_data['DeviceProtection'] = input("Enter DeviceProtection (Yes, No, or No internet service): ")
    input_data['TechSupport'] = input("Enter TechSupport (Yes, No, or No internet service): ")
    input_data['StreamingTV'] = input("Enter StreamingTV (Yes, No, or No internet service): ")
    input_data['StreamingMovies'] = input("Enter StreamingMovies (Yes, No, or No internet service): ")
    input_data['Contract'] = input("Enter Contract (Month-to-month, One year, or Two year): ")
    input_data['PaperlessBilling'] = input("Enter PaperlessBilling (Yes or No): ")
    input_data['PaymentMethod'] = input("Enter PaymentMethod (Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)): ")
    input_data['tenure'] = input("Enter tenure (in months): ")

    return input_data

# Main function to predict churn based on user input
def main():
    input_data = get_input()
    feature_names = model.feature_names_in_
    prediction, probability = predict_churn(pd.DataFrame([input_data]), feature_names)
    
    if prediction[0] == 1:
        print("This customer is likely to be churned!!")
    else:
        print("This customer is likely to continue!!")
    
    print("Confidence: {:.2f}%".format(probability[0] * 100))

# Run the main function
if __name__ == "__main__":
    main()
