"""
CHAIN.PY - TUTKU SUPPLY CHAIN DATASET İÇİN RAG SİSTEMİ
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import RetrievalQA

from config import GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, VECTOR_DB_PATH
from data_loader import load_tutku_supply_chain_data  # ÖZEL: Tutku dataset loader

class SupplyChainChatbot:
    def __init__(self):
        """Chatbot'u başlat"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        self.vector_store = None
        self.qa_chain = None
        
    def load_and_process_data(self):
        """TUTKU ÖZDENİZ DATASET'İNİ YÜKLE"""
        print("📚 Tutku Özdeniz supply chain dataset'i yükleniyor...")
        
        # ÖZEL: Tutku dataset'ini yükle
        documents = load_tutku_supply_chain_data()
        
        if not documents:
            print("❌ Hiç veri yüklenemedi, demo veri kullanılıyor...")
            documents = [Document(
                page_content="Supply Chain Management: Tedarik zinciri yönetimi temel kavramları.",
                metadata={"source": "demo_data"}
            )]
        
        # PDF'ler için optimize edilmiş bölücü
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # PDF metinleri için
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ {len(chunks)} metin parçası oluşturuldu")
        return chunks
    
    def setup_vector_store(self, chunks):
        """Vektör veritabanını oluştur"""
        print("🗄️ Vektör veritabanı oluşturuluyor...")
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("✅ Vektör veritabanı hazır")
    
    def setup_qa_chain(self):
        """Soru-cevap zincirini kur"""
        print("🔗 QA zinciri kuruluyor...")
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # PDF'ler için daha fazla kaynak
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        print("✅ QA zinciri hazır")
    
    def initialize_chatbot(self):
        """Tüm sistemi başlat"""
        print("🤖 Supply Chain Chatbot başlatılıyor...")
        
        chunks = self.load_and_process_data()
        self.setup_vector_store(chunks)
        self.setup_qa_chain()
        
        print("🎉 Chatbot hazır! Sorularınızı bekliyorum...")
    
    def ask_question(self, question):
        """Kullanıcı sorusuna cevap ver"""
        if not self.qa_chain:
            return "❌ Chatbot henüz hazır değil."
        
        try:
            response = self.qa_chain({"query": question})
            answer = response["result"]
            source_docs = response["source_documents"]
            
            # Kaynakları göster (PDF dosya isimleriyle)
            sources = "\n\n📚 Kaynak Dosyalar:\n"
            for i, doc in enumerate(source_docs[:3]):
                filename = doc.metadata.get('filename', 'Bilinmeyen dosya')
                source_type = doc.metadata.get('type', 'pdf')
                preview = doc.page_content[:100].replace('\n', ' ') + "..."
                
                sources += f"{i+1}. 📄 {filename} ({source_type})\n"
                sources += f"   Önizleme: {preview}\n\n"
            
            return answer + sources
            
        except Exception as e:
            return f"❌ Hata oluştu: {str(e)}"

# Test
if __name__ == "__main__":
    chatbot = SupplyChainChatbot()
    chatbot.initialize_chatbot()
    
    # Test soruları
    test_questions = [
        "Supply chain management nedir?",
        "Tedarik zinciri optimizasyonu nasıl yapılır?",
        "Lojistik yönetimi hakkında bilgi ver"
    ]
    
    for question in test_questions:
        print(f"\n🧪 Soru: {question}")
        print("─" * 50)
        print(chatbot.ask_question(question))
