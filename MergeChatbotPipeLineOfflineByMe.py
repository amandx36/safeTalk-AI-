import os
    # for disabling the merro messages 

# for disabling the oneDNN / TF spam messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disables oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides TensorFlow warnings/info


import requests
import json
from transformers import pipeline

from dotenv import load_dotenv


load_dotenv()  # loads variables from .env

HF_TOKEN = os.getenv("HF_API_TOKEN")
GEMMA_URL = os.getenv("GEMMA_API_URL")


load_dotenv()  # loads variables from .env
# -=-=-=-=-=-=-=-=-=-=mental health classifier =-=-=-==--=-=-=-=-=-=-=
# pipeline = easy shortcut to use ML models.
# You don’t have to:
# - Load tokenizer manually
# - Convert text to numbers
# - Run the model step by step
#
# NOTE: If the model requires authentication, set your HF token in env:
# export HUGGINGFACEHUB_API_TOKEN="your_token_here"
# or run `huggingface-cli login` before running this script.

# mental health classifier (change model if you prefer)
mhClassifier = pipeline(
    "text-classification",
    model="tahaenesaslanturk/mental-health-classification-v0.2",
    token=HF_TOKEN

    # if authentacation is required than you can add you token !!!! :)  
)

# making the function for the checking ;
def check_mental_health(UserText):
    result = mhClassifier(UserText)[0]
    label, MentalHealthScore = result['label'], result['score']
    return label, MentalHealthScore 


# =--=-=-=-=-=-=-=-=-   ai 02 for chatting !!!!-=-=-=-=-=-=-=-=-=-=-=-
def get_chat_response(userText):
    url = GEMMA_URL
  
    # the gemma reply is in the chunk chunk and the give me in the jason format so i have to merge it 
    # {"response":"Hello","done":false}
    # {"response":"!","done":false}
    # {"response":" I","done":false}
    # ...
    # {"done":true,"done_reason":"stop"}

    values = {
        "model": "gemma:2b",
        "prompt": f"User: {userText}\nAssistant:",
        "stream": True    # for the chunked output !!! 
    }

    # now its time to get the response brother !!!! 
    try:
        response = requests.post(url, json=values, stream=True, timeout=60)
    except Exception as e:
        return f"[Error] Could not reach Ollama/Gemma API: {e}"

    # now make an empty string for putting the value of reply from the gemma:2b ai 
    full_reply = ""

    #  concating the output we get from the gemma ai
    for line in response.iter_lines():
        # iter_lines()
        # Reads the response line by line as it comes.
        # It does NOT wait for the full response. Each chunk is available immediately.
        # It just gives you each line (usually in bytes) from the server stream.
        if line:
            # decoding the  output comes from the ai which is the byte code !!! 
            data = json.loads(line.decode("utf-8"))

            # full_reply += data["response"]
            # This is the part that merges the chunks from each line into a single reply.
            if "response" in data:
                full_reply += data["response"]

            # if data.get("done"):
            # This is the stop signal. tells  loop: “Hey, AI finished sending data. You can break now.”
            # closing it 

            if data.get("done"):
                break

    return full_reply


#  MERGED pipeline (mental-check + chat)
def chatbotPipeline(UserText):
    # first  check the mental health score in the text if critical than send the alert message to our  councillor 
    label, score = check_mental_health(UserText)
  
    # i use for the debugging !!! 
    
    
   #  print(f"[DEBUG] Classifier output: {label}, score: {score}")


    # now if the output of the ai give suicidal depression etc words and the score more than the 0.7 than triger the alert which is the further process !!! 
    if label.lower() in ["suicidal", "suicidewatch", "depression", "stress", "anxiety"] and  score > 0.7:
        print(
            f"Critical situation , it look like you may experience personal assistance and you were experiencing {label}. "
            "You can call AASRA Helpline (India) at 9152987821 for support."
        )

        # return none so that sudicidal ai did not return anything !!!!! 
        return None # dont return anything !!!!! 
    
    # if all things fine than continue with the chatting !!!! 
    return get_chat_response(UserText)


# run the chatBot  unless  the user close the chat 
while True:
    user = input("You: ")
    if user.lower() in ["quit", "exit"]:
        break 

    reply = chatbotPipeline(user)
    # Only print the bot reply when there is not any critical case !!!  
    if reply:
        print("Bot:", reply)
        


