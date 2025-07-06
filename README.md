# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)


This Streamlit app combines **OpenAI's GPT-4** with **retrieval-augmented generation (RAG)** to provide grounded medical suggestions based on symptom descriptions. It uses a local CSV dataset of symptoms, diagnoses, and treatments.

## 💡 What It Does

- Accepts a natural-language symptom description (e.g., "I feel dizzy and my joints hurt").
- Uses OpenAI’s `text-embedding-ada-002` to embed the user query.
- Retrieves the most semantically similar cases from the dataset using **FAISS**.
- Generates an answer using **GPT-4**, grounded in the retrieved data.

## 🔍 Dataset

The app expects a file named `medical_data.csv` with the following columns:

- `symptoms`
- `diagnosis`
- `treatment`

Each row describes a known medical case.

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt



### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
