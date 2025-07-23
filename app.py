import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("Employee Salary Prediction App")

st.write(
    "Upload your CSV file (e.g., `adult.csv`), change parameters, and predict if a person's salary is >50K or <=50K. "
    "The app is based on your notebook's pipeline (data cleaning, Label Encoding, MinMaxScaler, KNN classifier)."
)
# --- Data Upload ---
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write("Sample data:")
    st.dataframe(data.head(10))

    # --- Data Cleaning, Preprocessing ---
    # Replace missing/unknowns
    if "workclass" in data.columns:
        data['workclass'].replace({'?': 'Others'}, inplace=True)
    if "occupation" in data.columns:
        data['occupation'].replace({'?': 'Others'}, inplace=True)
    if "education" in data.columns:
        # Remove seldom categories as in your notebook
        data = data[~data['education'].isin(['1st-4th','5th-6th','Preschool'])]
    if "workclass" in data.columns:
        # Remove rows unlikely for prediction
        data = data[~data["workclass"].isin(['Without-pay', 'Never-worked'])]

    # Remove outlier age
    if "age" in data.columns:
        data = data[(data['age']>=17) & (data['age']<=75)]

    # Ensure "education-num" column naming as in notebooks
    if 'educational-num' in data.columns:
        data.rename(columns={'educational-num':'education-num'}, inplace=True)
    # Drop text education col if both present
    if 'education' in data.columns and 'education-num' in data.columns:
        data.drop(columns=['education'], inplace=True)
    
    # --- Encode categorical columns ---
    cat_columns = [
        "workclass", "marital-status", "occupation", "relationship",
        "race", "gender", "native-country"
    ]
    encoders = {}
    for col in cat_columns:
        if col in data.columns:
            lbl = LabelEncoder()
            data[col] = lbl.fit_transform(data[col])
            encoders[col] = lbl

    # --- Split into features and target ---
    if 'income' in data.columns:
        X = data.drop(columns=['income'])
        y = data['income']
    elif 'Income' in data.columns:
        X = data.drop(columns=['Income'])
        y = data['Income']
    else:
        st.warning("No 'income' column found. Please check your CSV.")
        st.stop()

    # --- Scale features ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train Classifier ---
    knn = KNeighborsClassifier()
    knn.fit(X_scaled, y)

    st.success("Model trained! Now enter details below to predict salary class:")

    # --- Collect User Input for Prediction ---
    def input_features():
        input_dict = {}
        for col in X.columns:
            # Numerical columns
            if np.issubdtype(data[col].dtype, np.number):
                min_v = int(data[col].min())
                max_v = int(data[col].max())
                mean_v = float(data[col].mean())
                val = st.number_input(f"{col}", min_value=min_v, max_value=max_v, value=int(mean_v))
                input_dict[col] = val
            # Encoded categorical columns (provide label)
            elif col in encoders:
                labels = list(encoders[col].classes_)
                selected = st.selectbox(f"{col}", labels)
                code = encoders[col].transform([selected])[0]
                input_dict[col] = code
            else:
                st.warning(f"Unknown feature type for {col}, please update code.")
                st.stop()
        return pd.DataFrame([input_dict])

    input_df = input_features()
    # Scale user input
    input_scaled = scaler.transform(input_df)

    if st.button("Predict Salary Class"):
        pred_salary = knn.predict(input_scaled)[0]
        st.subheader(f"**Prediction: {pred_salary}**")

    st.write("---")
    st.markdown("**Note:** This web app retrains a fresh KNN model every time a CSV is uploaded for demonstration, so use it for interactive testing and exploration.")

else:
    st.info("Please upload a CSV file to begin.")

