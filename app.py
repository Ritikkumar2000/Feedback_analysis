# app.py
import streamlit as st
import pickle
import re
import nltk

# ============================================
# FIX: Download NLTK data for Streamlit Cloud
# ============================================
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
# ========================
# LOAD MODELS
# ========================

@st.cache_resource
def load_models():
    """Load trained model and vectorizer"""
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ========================
# TEXT PREPROCESSING
# ========================

def clean_text(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# ========================
# PREDICTION FUNCTION
# ========================

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment category for given text"""
    cleaned = clean_text(text)
    
    if len(cleaned) == 0:
        return None, None
    
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    # Get probability if available
    try:
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
    except:
        confidence = None
    
    return prediction, confidence

# ========================
# STREAMLIT UI
# ========================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Feedback Sentiment Analyzer",
        page_icon="💬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 42px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 18px;
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .suggestion {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">💬 Feedback Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered sentiment classification for user feedback</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        """
        This app uses machine learning to classify user feedback into four categories:
        
        • **Positive**: Favorable feedback
        • **Negative**: Critical feedback
        • **Positive with Suggestion**: Favorable with improvement ideas
        • **Negative with Suggestion**: Critical with improvement ideas
        
        The model is trained using Logistic Regression with TF-IDF features.
        """
    )
    
    st.sidebar.title("📊 Model Info")
    st.sidebar.write("**Algorithm**: Logistic Regression")
    st.sidebar.write("**Features**: TF-IDF (5000 features)")
    st.sidebar.write("**Training Samples**: 1,022")
    st.sidebar.write("**Classes**: 4")
    
    # Load models
    try:
        model, vectorizer = load_models()
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train the model first using 'sentiment_model_training.py'")
        return
    
    # Main content
    st.write("")
    st.write("### Enter User Feedback")
    
    # Text input
    user_input = st.text_area(
        "Type or paste feedback below:",
        height=150,
        placeholder="Example: The product works great but the UI could be more intuitive..."
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
    
    # Prediction
    if predict_button:
        if not user_input or len(user_input.strip()) < 10:
            st.warning("⚠️ Please enter at least 10 characters of feedback.")
        else:
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_sentiment(user_input, model, vectorizer)
                
                if prediction is None:
                    st.error("❌ Unable to process the text. Please try different input.")
                else:
                    # Display result
                    st.write("---")
                    st.write("### 📋 Analysis Results")
                    
                    # Determine styling based on category
                    if "Positive" in prediction and "Suggestion" not in prediction:
                        css_class = "positive"
                        emoji = "😊"
                    elif "Negative" in prediction and "Suggestion" not in prediction:
                        css_class = "negative"
                        emoji = "😞"
                    else:
                        css_class = "suggestion"
                        emoji = "💡"
                    
                    # Result box
                    st.markdown(
                        f'<div class="result-box {css_class}">{emoji} {prediction}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence score
                    if confidence:
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                    
                    # Additional insights
                    st.write("### 💭 Interpretation")
                    if prediction == "Positive":
                        st.success("This feedback expresses satisfaction without suggesting changes.")
                    elif prediction == "Negative":
                        st.error("This feedback expresses dissatisfaction without providing solutions.")
                    elif prediction == "Positive with Suggestion":
                        st.info("This feedback is positive but includes actionable improvement ideas.")
                    elif prediction == "Negative with Suggestion":
                        st.warning("This feedback is critical but offers constructive suggestions.")
    
    # Example feedback section
    st.write("---")
    st.write("### 📝 Example Feedback")
    
    examples = {
        "Positive": "The product exceeded my expectations! Highly recommend it.",
        "Negative": "Very disappointed with the quality. Not worth the price.",
        "Positive with Suggestion": "Great product overall, but would be perfect with a mobile app.",
        "Negative with Suggestion": "The interface is confusing. Adding a tutorial would help new users."
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Positive Examples:**")
        if st.button("Try Positive Example"):
            st.session_state.example_text = examples["Positive"]
        if st.button("Try Positive + Suggestion"):
            st.session_state.example_text = examples["Positive with Suggestion"]
    
    with col2:
        st.write("**Negative Examples:**")
        if st.button("Try Negative Example"):
            st.session_state.example_text = examples["Negative"]
        if st.button("Try Negative + Suggestion"):
            st.session_state.example_text = examples["Negative with Suggestion"]
    
    # Display selected example
    if 'example_text' in st.session_state:
        st.info(f"**Selected Example:** {st.session_state.example_text}")

if __name__ == "__main__":
    main()
