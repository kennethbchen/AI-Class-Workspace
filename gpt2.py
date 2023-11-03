from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import numpy as np
import sys

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# https://github.com/joshuaeckroth/csci431-gpt2

rev = input("Type your review: ")

msg = \
"""
Review: Works really well.
Sentiment: Positive

Review: I hate this.
Sentiment: Negative

Review: """

msg += rev + "\n"
msg += "Sentiment: "

print(msg)

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

i = 0
while i < 3:
    i += 1
    inputs = tokenizer([msg], return_tensors="pt")
    outputs = model.forward(**inputs, labels=inputs.input_ids)
    # grab the last logit
    logits = outputs.logits.detach()
    logits = logits[0, -1, :]
    # append the highest scoring token to the message
    response = tokenizer.decode(torch.argmax(logits))
    print(response, end='')
    msg += response
    # find top-10 tokens
    #top_k = 10
    #top_tokens = np.argsort(logits)[-top_k:]
    # print each with its score
    #for tok in top_tokens:
    #    print(f"{tokenizer.decode(tok):20s} | {logits[tok]:.3f}")


