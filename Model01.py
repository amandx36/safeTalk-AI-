from transformers import pipeline 

import os     
# for disabling the merro messages 

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disables oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides TensorFlow warnings/info



# pipeline = easy shortcut to use ML models.
# You don’t have to:

# Load tokenizer manually

# Convert text to numbers

# Run the model step by step

# mental health  calssifier 

mhClassifier =  pipeline("text-classification" , model  = HF_TOKEN)



# pipeline = easy shortcut to use ML models.
# You don’t have to:

# Load tokenizer manually

# Convert text to numbers

# Run the model step by step

# mental health  calssifier 

mhClassifier =  pipeline("text-classification" , model  = "tahaenesaslanturk/mental-health-classification-v0.2" , token =HF_TOKEN)

# 

# pranavpsv/bert-base-uncased-mental-health due  to of authentication and token issue i am using this 
 
# making the function for the checking ;

def checker_mentalHealth(UserText):
    result =  mhClassifier(UserText)[0]
    label , MentalHealthScore = result['label'], result["score"]

    return label , MentalHealthScore 


# pranavpsv/bert-base-uncased-mental-health due  to of authentication and token issue i am using this 
 
# making the function for the checking ;

def checker_mentalHealth(UserText):
    result =  mhClassifier(UserText)[0]
    label , MentalHealthScore = result['label'], result["score"]

    return label , MentalHealthScore 


text =  input("Enter the text for checking :) "  )

output01 ,  output02  =  checker_mentalHealth(text)

print(output01)
print(output02)


