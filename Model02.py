from re import M
import requests
import json  # Needed for parsing JSON lines


def getChatResponse(userText):
    url = "http://localhost:11434/api/generate"
  
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
    response = requests.post(url, json=values, stream=True)
    # now make a empty string for putting the value of reply from the gemma:2b ai 
    fullReply = ""

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
                fullReply += data["response"]

            # if data.get("done"):
            # This is the stop signal. It tells the loop: “Hey, AI finished sending data. You can break now.”
            if data.get("done"):
                break

    return fullReply

aman  =  input("Enter the text :== ")
reply =  getChatResponse(aman)
print("AI Reply: "+reply)



