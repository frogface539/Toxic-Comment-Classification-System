import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("models/final_model.h5")
with open("models/final_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_COLORS = {
    "toxic": "#FF4B4B",        
    "severe_toxic": "#FF7F50", 
    "obscene": "#FFA500",     
    "threat": "#FFD700",       
    "insult": "#00BFFF",       
    "identity_hate": "#9370DB" 
}

# App Config
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered", )
st.markdown("<h1 style='text-align: center; color: #FFFDD0;'>💬 Toxic Comment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a comment below and detect toxicity across 6 categories using a deep learning model.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
comment = st.text_area("📝 Enter a comment for classification:", height=150)

threshold = st.slider("🎚️ Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("🔍 Predict Toxicity"):
    if comment.strip() == "":
        st.warning("⚠️ Please enter a comment.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)[0]

        st.markdown("### 🧾 Results")

        for label, prob in zip(LABELS, preds):
            is_toxic = prob >= threshold
            bar_color = LABEL_COLORS[label]
            emoji = "✅" if not is_toxic else "⚠️"

            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<b style='color:{bar_color}'>{label}</b> {emoji}", unsafe_allow_html=True)
            with col2:
                st.progress(float(prob))


        st.markdown("---")
        st.success("✅ Prediction complete. Adjust threshold if needed.")


st.markdown("<hr style='margin-top: 40px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.9em;'>Made with ♥ by Lakshay</p>", unsafe_allow_html=True)
