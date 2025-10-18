# akbank-generative-ai-chatbot
RAG mimarisi kullanılarak geliştirilmiş bir yapay zekâ tabanlı chatbot projesi.

# Educational Q&A Chatbot 🤖

## 🎯 Project Aim
This project aims to create an **educational chatbot** capable of answering general knowledge and academic questions.  
It uses the **SQuAD dataset** from Hugging Face and a **Retrieval-Augmented Generation (RAG)** architecture to generate relevant, well-grounded answers.

---

## 📚 Dataset
**Dataset:** [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/squad)

Each entry in this dataset includes:
- A *context* paragraph (text passage)
- A *question* based on the paragraph
- A set of *answers* extracted from the context

The chatbot retrieves the most relevant passages from this dataset and uses a large language model to generate accurate, natural responses.

---

## ⚙️ Technologies
- Python  
- LangChain  
- ChromaDB  
- OpenAI / Gemini API  
- Streamlit  

---

## 🧠 Architecture
1. **Data Loading**: Load a subset of the SQuAD dataset using Hugging Face.  
2. **Embeddings**: Convert text chunks into embeddings using a pre-trained model.  
3. **Vector Store**: Store embeddings in ChromaDB for fast retrieval.  
4. **RAG Pipeline**: Retrieve relevant passages and pass them to an LLM for final answer generation.  
5. **Web Interface**: Users can ask questions via Streamlit.

---

## 🧩 Example Workflow
1. User asks: “Who wrote *Pride and Prejudice*?”
2. The system retrieves relevant passages from SQuAD.
3. The LLM responds: “Jane Austen wrote *Pride and Prejudice*.”

---

## 🌐 Deployment
Will be deployed on Streamlit Cloud.
