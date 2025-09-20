import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
from transformers import pipeline

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
GEMMA_URL = os.getenv("GEMMA_API_URL")

# Initialize Mental Health Classifier
mhClassifier = pipeline(
    "text-classification",
    model="tahaenesaslanturk/mental-health-classification-v0.2",
    token=HF_TOKEN
)

def check_mental_health(UserText):
    result = mhClassifier(UserText)[0]
    label, score = result['label'], result['score']
    return label, score

def get_chat_response(userText):
    values = {
        "model": "gemma:2b",
        "prompt": f"User: {userText}\nAssistant:",
        "stream": True
    }

    try:
        response = requests.post(GEMMA_URL, json=values, stream=True, timeout=60)
    except Exception as e:
        return f"[Error] Could not reach Gemma API: {e}"

    full_reply = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_reply += data["response"]
            if data.get("done"):
                break
    return full_reply

def chatbot_pipeline(UserText):
    label, score = check_mental_health(UserText)

    # Critical alert
    if label.lower() in ["suicidewatch", "suicidal", "depression", "stress", "anxiety"] and score > 0.7:
        alert_msg = (
            f"⚠️ Critical situation detected! You may be experiencing '{label}' "
            f"with confidence {score:.2f}.\n"
            "Please call AASRA Helpline (India) at 9152987821 for support."
        )
        return None, alert_msg

    # Normal conversation
    reply = get_chat_response(UserText)
    return reply, None

# Streamlit UI
st.set_page_config(page_title="Safe Chatbot", page_icon="🤖")
st.title("💬 Mental Health Aware Chatbot")
st.write("Talk to the chatbot below. It will alert if a critical mental health issue is detected.")

# Input box
user_input = st.text_input("You:", "")

if user_input:
    reply, alert = chatbot_pipeline(user_input)
    
    if alert:
        st.error(alert)
    if reply:
        st.success(reply)