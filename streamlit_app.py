import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np
import time
import os

# Set your API key
openai.api_key = st.secrets["OPEN_AI_KEY"]

# --------------------------
# Load and embed dataset
# --------------------------

@st.cache_resource(show_spinner=False)
def load_data():
    df = pd.read_csv("Diseases_Symptoms.csv", dtype=str)
    df['desease_info'] = 'Name: ' + df['Name'] + ', Symptoms: ' + df['Symptoms'] + ', Treatments: ' + df['Treatments'] + ', Contagious: ' + df['Contagious'] + ', Chronic: ' + df['Chronic']
    df = df.dropna(subset=['desease_info'])
    emb_matrix = np.load("embeddings.npy")

    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)

    return df, index

df, faiss_index = load_data()

# --------------------------
# UI SECTION 1: Title & Summary
# --------------------------
st.title("🩺 Symptom Checker Assistant (RAG + GPT-3.5)")
st.markdown("""
This app uses OpenAI's GPT-3.5 Turbo and a medical dataset to answer questions about symptoms by matching them with known diagnoses and treatments.

Just describe your symptoms and the assistant will suggest a likely condition based on retrieved medical entries.
""")

# --------------------------
# UI SECTION 2: Data Overview
# --------------------------
st.header("📊 Data Overview")
st.write("Here are a few example entries from the dataset:")
st.dataframe(df[['Name', 'Symptoms', 'Treatments']].head(6), use_container_width=True)


def retrieve_top_docs(query, k=3):
    query_embedding = openai.Embedding.create(input=[query], model="text-embedding-ada-002")["data"][0]["embedding"]
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return df.iloc[indices[0]]

def generate_response(user_query):
    top_docs = retrieve_top_docs(user_query)
    context = "\n".join(top_docs['desease_info'].tolist())
    prompt = f"""
You are a helpful medical assistant.

Context:
{context}

The user reports: "{user_query}"

Based on the context, suggest the most likely condition and appropriate treatment.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

# --------------------------
# Session State for Chat
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Describe your symptoms and I’ll suggest possible conditions and treatments. For example: “I have a constant knee pain” or “I feel vertigo and hearing loss.”"}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (Streamlit’s chat UI)
if prompt := st.chat_input("Enter your symptoms here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and store assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})