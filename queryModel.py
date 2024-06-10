from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def queryOpenai(messages):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=messages
    )
    return response.choices[0].message.content
