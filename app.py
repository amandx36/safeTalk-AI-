import streamlit as st
import os
import requests
import json
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Mental Health Aware Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Environment variables
HF_TOKEN = os.getenv("HF_API_TOKEN")
GEMMA_URL = os.getenv("GEMMA_API_URL")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        margin-right: auto;
    }
    
    .alert-message {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #bd2130;
    }
    
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
    }
    
    .sidebar-content {
        padding: 20px 0;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_mental_health_classifier():
    """Load the mental health classifier model"""
    try:
        classifier = pipeline(
            "text-classification",
            model="tahaenesaslanturk/mental-health-classification-v0.2",
            token=HF_TOKEN
        )
        return classifier
    except Exception as e:
        st.error(f"Failed to load mental health classifier: {str(e)}")
        return None

def check_mental_health(user_text, classifier):
    """Check mental health status of user input"""
    if not classifier:
        return "unknown", 0.0
    
    try:
        result = classifier(user_text)[0]
        label, score = result['label'], result['score']
        return label, score
    except Exception as e:
        st.error(f"Error in mental health classification: {str(e)}")
        return "error", 0.0

def get_chat_response(user_text):
    """Get response from Gemma AI model"""
    if not GEMMA_URL:
        return "[Error] Gemma API URL not configured"
    
    values = {
        "model": "gemma:2b",
        "prompt": f"User: {user_text}\nAssistant:",
        "stream": True
    }
    
    try:
        response = requests.post(GEMMA_URL, json=values, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[Error] Could not reach Gemma API: {str(e)}"
    
    full_reply = ""
    try:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_reply += data["response"]
                if data.get("done"):
                    break
    except json.JSONDecodeError as e:
        return f"[Error] Failed to parse API response: {str(e)}"
    
    return full_reply.strip() if full_reply.strip() else "I apologize, but I couldn't generate a response."

def chatbot_pipeline(user_text, classifier):
    """Main chatbot pipeline with mental health checking"""
    # Mental health check
    label, score = check_mental_health(user_text, classifier)
    
    # Configuration
    THRESHOLD = 0.7
    CRITICAL_LABELS = ["suicidewatch", "suicidal", "depression", "stress", "anxiety"]
    
    # Check for critical mental health indicators
    if label.lower() in CRITICAL_LABELS and score > THRESHOLD:
        alert_msg = f"""
        🚨 **Critical Mental Health Alert Detected**
        
        **Detected Issue**: {label.title()}
        **Confidence Level**: {score:.1%}
        
        **Immediate Support Available**:
        • **AASRA Helpline (India)**: 9152987821
        • **National Suicide Prevention Lifeline**: 988
        • **Crisis Text Line**: Text HOME to 741741
        
        Please reach out to a mental health professional or trusted person immediately.
        Your life matters, and help is available. 💙
        """
        return None, alert_msg, label, score
    
    # Normal conversation
    reply = get_chat_response(user_text)
    return reply, None, label, score

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'mental_health_stats' not in st.session_state:
        st.session_state.mental_health_stats = {}
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0

def update_mental_health_stats(label):
    """Update mental health statistics"""
    if label in st.session_state.mental_health_stats:
        st.session_state.mental_health_stats[label] += 1
    else:
        st.session_state.mental_health_stats[label] = 1

def display_chat_history():
    """Display chat history with improved formatting"""
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {chat['user_input']}
                <div class="timestamp">{chat['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response or alert
            if chat.get('alert'):
                st.markdown(f"""
                <div class="alert-message">
                    <strong>🚨 Mental Health Alert</strong><br>
                    {chat['alert']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Bot:</strong> {chat['bot_response']}
                    <div class="timestamp">Mental Health Status: {chat['mental_health_label']} ({chat['mental_health_score']:.1%})</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start a conversation!")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Load classifier
    classifier = load_mental_health_classifier()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Mental Health Aware Chatbot</h1>
        <p>A safe space for conversation with built-in mental health monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.header("📊 Session Statistics")
        
        # Display stats
        st.markdown(f"""
        <div class="stats-box">
            <strong>Total Messages:</strong> {st.session_state.total_messages}
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.mental_health_stats:
            st.subheader("Mental Health Indicators")
            for label, count in st.session_state.mental_health_stats.items():
                st.markdown(f"""
                <div class="stats-box">
                    <strong>{label.title()}:</strong> {count} times
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Control buttons
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.mental_health_stats = {}
            st.session_state.total_messages = 0
            st.rerun()
        
        if st.button("💾 Export Chat", type="secondary"):
            if st.session_state.chat_history:
                chat_export = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download Chat History",
                    data=chat_export,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.markdown("---")
        
        # Help section
        st.subheader("🆘 Emergency Contacts")
        st.markdown("""
        **India:**
        - AASRA: 9152987821
        - Vandrevala Foundation: 9999 666 555
        
        **International:**
        - Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        display_chat_history()
        
        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Type your message here...",
                height=100,
                placeholder="How are you feeling today? What's on your mind?"
            )
            submitted = st.form_submit_button("Send Message", type="primary")
            
            if submitted and user_input.strip():
                if not classifier:
                    st.error("Mental health classifier not available. Please check your configuration.")
                    return
                
                # Show processing message
                with st.spinner("Processing your message..."):
                    reply, alert, label, score = chatbot_pipeline(user_input, classifier)
                
                # Create chat entry
                chat_entry = {
                    'user_input': user_input,
                    'bot_response': reply,
                    'alert': alert,
                    'mental_health_label': label,
                    'mental_health_score': score,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Update session state
                st.session_state.chat_history.append(chat_entry)
                st.session_state.total_messages += 1
                update_mental_health_stats(label)
                
                # Rerun to update display
                st.rerun()
    
    with col2:
        st.subheader("ℹ️ Information")
        
        st.info("""
        **How it works:**
        1. Type your message
        2. AI analyzes for mental health indicators
        3. Provides appropriate response or alerts
        4. Chat history is maintained
        """)
        
        st.warning("""
        **Privacy Notice:**
        Your conversations are processed locally and not stored permanently. 
        Chat history is cleared when you refresh the page.
        """)
        
        # Model status
        if classifier:
            st.success("🟢 Mental Health Classifier: Active")
        else:
            st.error("🔴 Mental Health Classifier: Inactive")
        
        if GEMMA_URL:
            st.success("🟢 Gemma AI: Connected")
        else:
            st.error("🔴 Gemma AI: Not configured")

if __name__ == "__main__":
    main()