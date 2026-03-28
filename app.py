import streamlit as st
from src.predict import predict_news

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🛡️ Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered detection system</div>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
text = st.text_area("📰 Enter News Article", height=150)
st.markdown('</div>', unsafe_allow_html=True)

# Button
if st.button("🚀 Analyze News"):
    if text.strip() == "":
        st.warning("Please enter some news!")
    else:
        with st.spinner("Analyzing... 🤖"):
            pred, prob = predict_news(text)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Result
        if pred == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

        # Confidence
        st.write("### Confidence Score")
        st.progress(float(prob))
        st.write(f"{prob:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)