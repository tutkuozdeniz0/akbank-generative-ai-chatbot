# rag_pipeline.py
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

def build_rag_pipeline():
    """
    Build the RAG pipeline using SQuAD dataset
    """
    try:
        # 1. Load dataset from Hugging Face
        st.info("📥 Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="train[:2000]")  # Start with 2000 examples
        docs = [d["context"] for d in dataset]
        
        # 2. Split documents into chunks
        st.info("✂️ Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = splitter.create_documents(docs)
        
        # 3. Initialize embeddings
        st.info("🔤 Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # 4. Create vector store
        st.info("🗄️ Creating vector database...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # 5. Create QA chain
        st.info("🔗 Setting up QA chain...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            verbose=False
        )
        
        return qa
        
    except Exception as e:
        raise Exception(f"Error building RAG pipeline: {str(e)}")
