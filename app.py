import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model/phishing_model.joblib")

model = load_model()
feature_names = list(model.feature_names_in_)

st.title("🛡️ Phishing Website Detection")
st.write(
    "Enter website-related feature values to predict whether a website "
    "is legitimate or phishing."
)

with st.form("prediction_form"):
    input_data = {}

    columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        with columns[index % 3]:
            input_data[feature] = st.number_input(
                feature.replace("_", " "),
                min_value=-1,
                max_value=1,
                value=0,
                step=1
            )

    submitted = st.form_submit_button("Analyse Website")

if submitted:
    input_frame = pd.DataFrame([input_data])
    prediction = model.predict(input_frame)[0]
    probability = model.predict_proba(input_frame)[0][1]

    if prediction == 1:
        st.error(f"Potential phishing website detected — risk score: {probability:.1%}")
    else:
        st.success(f"Likely legitimate website — phishing risk score: {probability:.1%}")

    st.warning(
        "Educational model only. Do not rely on this result as the sole "
        "security decision for a real website."
    )