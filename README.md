# 🩺 Symptom Checker Assistant (Streamlit + GPT-3.5 + RAG)

## 📘 Introduction

This is an interactive medical assistant built with **Streamlit** that uses **OpenAI's GPT-3.5** and **Retrieval-Augmented Generation (RAG)** to help users match their described symptoms with possible **diagnoses and treatments**.

The app uses:
- **A local medical dataset** of symptoms, diagnoses, and treatments
- **OpenAI’s `text-embedding-ada-002`** to perform semantic search
- **FAISS** for fast document retrieval
- **GPT-4** for generating grounded medical suggestions
- **Streamlit Chat UI** for a conversational experience

> ⚠️ **Disclaimer:** This app is for educational/demo purposes and does not provide professional medical advice.

---

## 💬 Example

After launching the app, a user might enter:

> **"I have a constant knee pain and limited mobility"**

The assistant might respond with:

> _"Your symptoms may be consistent with osteoarthritis or a related joint condition. It is typically managed with physical therapy, NSAIDs, and in some cases, joint injections. Please consult a healthcare provider for a diagnosis."_

---

## 📦 Prerequisites

1. **Install dependencies**

```bash
pip install -r requirements.txt