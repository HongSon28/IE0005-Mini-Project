import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
cabin = st.text_input("Your cabin, NA if unknown")


# Preprocess (adjust based on your training pipeline)
sex = 1 if sex == "male" else 0

cabin_bool = 0 if cabin == "NA" else 1

age_group = 0;

data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "SibSp": sibsp,
        "Fare": fare,
        "Age": age,
    }])

data["Age"] = data["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, 200]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
data["Age Group"] = pd.cut(data["Age"], bins, labels = labels)

label_encoder = LabelEncoder()
data['Age Group'] = label_encoder.fit_transform(data['Age Group'])

data = data.drop(['Age'], axis = 1)

data['CabinBool'] = cabin_bool

# Predict button
if st.button("Predict"):
    perc = model.predict_proba(data)[0][1] * 100

    st.write("You have ", perc, "% chance of survive")
    
    if perc >= 66:
        st.success("🎉 You most likely survived, congrats!")
    else:
        if perc <=33:
            st.error("💀 You likely to not survived, good luck next time!")
        else:
            st.warning("You may survive")