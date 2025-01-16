from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import google.generativeai as genai
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models with error handling
try:
    sentiment_model = pipeline("sentiment-analysis", model="./Sentiment_Model")
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_model = None

try:
    classifier_model = pipeline("zero-shot-classification", model="./Classifier_model")
except Exception as e:
    print(f"Error loading classifier model: {e}")
    classifier_model = None

vader_classifier = SentimentIntensityAnalyzer()

# Configure Gemini Pro using environment variable for security
genai_api_key = os.getenv("GENAI_API_KEY", "AIzaSyCA4-Pmug1UNb85sJrLN3xlXLNbPCIHIvc")
genai.configure(api_key=genai_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Helper functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scale_sentiment(score, category):
    if category == "positive":
        if score >= 0.8: return 5
        elif score >= 0.6: return 4
        elif score >= 0.4: return 3
        elif score >= 0.2: return 2
        else: return 1
    elif category == "negative":
        if score <= -0.8: return 5
        elif score <= -0.6: return 4
        elif score <= -0.4: return 3
        elif score <= -0.2: return 2
        else: return 1
    else:
        return 3

def ensemble_sentiment_analysis(text):
    cleaned_text = preprocess_text(text)
    if sentiment_model is None:
        return {"sentiment": "unknown", "scale": 0}
    transformer_result = sentiment_model(cleaned_text)[0]
    transformer_compound = {
        "5 stars": 1.0,
        "4 stars": 0.7,
        "3 stars": 0.0,
        "2 stars": -0.7,
        "1 star": -1.0
    }.get(transformer_result['label'], 0.0)

    vader_result = vader_classifier.polarity_scores(cleaned_text)
    combined_score = 0.7 * transformer_compound + 0.3 * vader_result['compound']

    if combined_score > 0.2: sentiment_class = "positive"
    elif combined_score < -0.2: sentiment_class = "negative"
    else: sentiment_class = "neutral"

    sentiment_scale = scale_sentiment(combined_score, sentiment_class)
    return {"sentiment": sentiment_class, "scale": sentiment_scale}

def classify_industry(text):
    if classifier_model is None:
        return "unknown"
    candidate_labels = ["finance", "medical", "e-commerce"]
    result = classifier_model(text, candidate_labels)
    return result["labels"][0]  # Return the top label

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Perform sentiment analysis and classification
        sentiment_result = ensemble_sentiment_analysis(user_message)
        classification_result = classify_industry(user_message)

        # Generate support context
        support_context = f"""
        This is a customer support inquiry with {sentiment_result['sentiment']} sentiment.
        The primary category is {classification_result}.
        Please respond in a {sentiment_result['sentiment']} and professional tone.
        """

        # Prepare prompt for Gemini model
        prompt = f"""Context: {support_context}
        Customer message: {user_message}
        Response:"""

        # Generate response using Gemini model
        response = gemini_model.generate_content(prompt)

        return jsonify({"message": response.text})

    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error running the Flask app: {e}")