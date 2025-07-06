import pandas as pd
import numpy as np
import openai
import time
import streamlit as st

openai.api_key = st.secrets["OPEN_AI_KEY"]

df = pd.read_csv("Diseases_Symptoms.csv", dtype="str")
df['desease_info'] = 'Name: ' + df['Name'] + ', Symptoms: ' + df['Symptoms'] + ', Treatments: ' + df['Treatments'] + ', Contagious: ' + df['Contagious'] + ', Chronic: ' + df['Chronic']
df = df.dropna(subset=['desease_info'])

embeddings = []

for text in df['desease_info']:
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    embeddings.append(embedding)
    time.sleep(1)  # rate limiting

# Save embeddings and dataset
np.save("embeddings.npy", np.array(embeddings))
