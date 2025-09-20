# 🧠 MindCare Chatbot

A **mental-health aware chatbot** built with Python, Streamlit, HuggingFace, and Gemma AI. It can chat like a normal chatbot but **alerts critical mental health situations** like suicidal thoughts, depression, or anxiety.

---

## 🔹 Features

- Chat interface with **Streamlit**.
- **Mental health classifier** using HuggingFace: `tahaenesaslanturk/mental-health-classification-v0.2`.
- **Critical alerts** for suicidal/depression/stress/anxiety messages.
- **Integration with Gemma AI (`gemma:2b`)** for normal conversation.
- Entire conversation **kept in UI**.

---

## 🔹 Requirements

- Python 3.11+
- Install dependencies:

```bash
    pip install -r requirements.txt




##  🔹  How They Work Together

    The user types a message.

    The message first goes to the Distress Detection model.

        If the score is above the threshold (for example, 0.7), the system Marks according to the text user provide and also give the score form 0-1 

        If the score is below the threshold, the system says: "Normal."

    Then a decision is made:

        If risk is detected, the bot sends a supportive message, helpline information, and flags for escalation.

        If normal, the bot forwards the user input to the chat model, which generates a reply.

    The final output is shown to the user.







User Input 
    │
    ├──► Model 2: Distress Classifier   (Using the oflline model that is )
    │        │
    │        ├──► High Risk → Escalate + Supportive Reply
    │        └──► Low Risk → Send to Chatbot
    │
    └──► Model 1: Chatbot → Normal friendly reply  [ollama run  gemma:2b]




API server (Ollama by default runs on http://localhost:11434):

Create a .env file in the root of your project:

    HF_API_TOKEN=hf_your_huggingface_token_here
    GEMMA_API_URL=http://localhost:11434/api/generate

    Replace hf_your_huggingface_token_here with your HuggingFace token.


##        SETUP

Clone the repo


Install Python dependencies:
pip install -r requirements.txt 


finally  RUN python  code 

