import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

msg = [
        {
            "role": "system",
            "content": "You are a priest in a confessional listening to the confessions of penitents."
        }
]

while True:

    user_input = input("> ")
    if len(user_input) == 0:
        continue

    if user_input == "exit()":
        break

    msg.append({"role":"user", "content":user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msg,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    res_text = response.choices[0].message.content

    print(res_text)
    msg.append({"role":"assistant","content": res_text})




