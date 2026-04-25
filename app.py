import streamlit as st
import joblib
from xgboost import XGBClassifier
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="DNA Confidence Classifier", page_icon="🧬")

@st.cache_resource
def load_models():
    vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    nusvc = joblib.load('model/nusvc_diabetes_model.joblib')
    xgb = XGBClassifier()
    xgb.load_model('model/xgb_diabetes_model.json')
    return vectorizer, nusvc, xgb

vectorizer, nusvc, xgb = load_models()

st.title("🧬 DNA Analysis with Confidence")

dna_input = st.text_area("Enter DNA Sequence:", height=150)

if st.button("Analyze Sequence"):
    if dna_input.strip():
        X_new = vectorizer.transform([dna_input])

        # Get Probabilities
        # [0] is Non-diabetic, [1] is Diabetic
        prob_nu = nusvc.predict_proba(X_new)[0]
        prob_xgb = xgb.predict_proba(X_new)[0]

        
        st.divider()
        
        col1, col2 = st.columns(2)

        # Helper function to display results
        def display_confidence(name, probs):
            class_idx = np.argmax(probs)
            confidence = probs[class_idx] * 100
            label = "Diabetic" if class_idx == 1 else "Non-diabetic"
            
            st.subheader(f"{name}")
            st.metric(label=f"Predicted: {label}", value=f"{confidence:.2f}%")
            
            # Visual Progress Bar
            color = "red" if class_idx == 1 else "green"
            st.progress(probs[1]) # Progress bar fills based on Diabetic probability
            st.caption(f"Confidence score for Diabetic: {probs[1]*100:.1f}%")

        with col1:
            display_confidence("NuSVC Model", prob_nu)

        with col2:
            display_confidence("XGBoost Model", prob_xgb)
            
    else:
        st.error("Please enter a sequence.")