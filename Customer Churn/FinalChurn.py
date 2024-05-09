import pandas as pd
import pickle
import streamlit as st

# Load the trained model
model = pickle.load(open("model.sav", "rb"))

# Function to preprocess input data
def preprocess_input(data, feature_names):
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

# Get input from the user using Streamlit
def get_input():
    input_data = {}
    input_data['SeniorCitizen'] = st.selectbox("Enter SeniorCitizen (0 or 1):", ['0', '1'])
    input_data['MonthlyCharges'] = st.text_input("Enter MonthlyCharges:")
    input_data['TotalCharges'] = st.text_input("Enter TotalCharges:")
    input_data['gender'] = st.selectbox("Enter gender:", ['Male', 'Female'])
    input_data['Partner'] = st.selectbox("Enter Partner:", ['Yes', 'No'])
    input_data['Dependents'] = st.selectbox("Enter Dependents:", ['Yes', 'No'])
    input_data['PhoneService'] = st.selectbox("Enter PhoneService:", ['Yes', 'No'])
    input_data['MultipleLines'] = st.selectbox("Enter MultipleLines:", ['Yes', 'No', 'No phone service'])
    input_data['InternetService'] = st.selectbox("Enter InternetService:", ['DSL', 'Fiber optic', 'No'])
    input_data['OnlineSecurity'] = st.selectbox("Enter OnlineSecurity:", ['Yes', 'No', 'No internet service'])
    input_data['OnlineBackup'] = st.selectbox("Enter OnlineBackup:", ['Yes', 'No', 'No internet service'])
    input_data['DeviceProtection'] = st.selectbox("Enter DeviceProtection:", ['Yes', 'No', 'No internet service'])
    input_data['TechSupport'] = st.selectbox("Enter TechSupport:", ['Yes', 'No', 'No internet service'])
    input_data['StreamingTV'] = st.selectbox("Enter StreamingTV:", ['Yes', 'No', 'No internet service'])
    input_data['StreamingMovies'] = st.selectbox("Enter StreamingMovies:", ['Yes', 'No', 'No internet service'])
    input_data['Contract'] = st.selectbox("Enter Contract:", ['Month-to-month', 'One year', 'Two year'])
    input_data['PaperlessBilling'] = st.selectbox("Enter PaperlessBilling:", ['Yes', 'No'])
    input_data['PaymentMethod'] = st.selectbox("Enter PaymentMethod:", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    input_data['tenure'] = st.text_input("Enter tenure (in months):")

    return input_data

# Main function to predict churn based on user input
def main():
    input_data = get_input()
    feature_names = model.feature_names_in_
    prediction, probability = predict_churn(pd.DataFrame([input_data]), feature_names)
    
    if prediction[0] == 1:
        st.write("This customer is likely to be churned!!")
    else:
        st.write("This customer is likely to continue!!")
    
    st.write("Confidence: {:.2f}%".format(probability[0] * 100))

# Run the main function
if __name__ == "__main__":
    main()
