"""
CHAIN.PY - TUTKU SUPPLY CHAIN DATASET İÇİN RAG SİSTEMİ

Bu dosya Supply Chain Management chatbot'unun beynidir.
RAG (Retrieval-Augmented Generation) mimarisi ile çalışır ve
Tutku Özdeniz'in Hugging Face'teki supply chain dataset'ini kullanır.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import RetrievalQA

from config import GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, VECTOR_DB_PATH

# Data loader importu - hata yönetimi ile
try:
    from data_loader import load_tutku_supply_chain_data
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ data_loader import hatası: {e}")
    DATA_LOADER_AVAILABLE = False
    
    # Fallback fonksiyon
    def load_tutku_supply_chain_data():
        """Data loader yoksa kullanılacak yedek fonksiyon"""
        print("❌ data_loader.py bulunamadı, demo veri kullanılıyor...")
        return []

class SupplyChainChatbot:
    def __init__(self):
        """Chatbot'u başlat - tüm bileşenleri initialize eder"""
        
        # API key kontrolü
        if not GEMINI_API_KEY:
            raise ValueError(
                "❌ GEMINI_API_KEY bulunamadı. "
                "Lütfen .env dosyasına ekleyin veya app.py'de girin. "
                "API key: https://aistudio.google.com/ adresinden alınabilir."
            )
        
        # Embedding modelini başlat - metinleri vektörlere çevirir
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        
        # Dil modelini başlat - cevapları üretir
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,  # Yaratıcılık seviyesi (0-1 arası)
            max_tokens=1000   # Maksimum cevap uzunluğu
        )
        
        # Diğer bileşenler
        self.vector_store = None  # Vektör veritabanı
        self.qa_chain = None      # Soru-cevap zinciri
        
    def load_and_process_data(self):
        """
        TUTKU ÖZDENİZ DATASET'İNİ YÜKLE VE İŞLE
        
        Returns:
            list: İşlenmiş metin parçaları (chunks)
        """
        print("📚 Tutku Özdeniz supply chain dataset'i yükleniyor...")
        
        # Data loader'ın kullanılabilirliğini kontrol et
        if not DATA_LOADER_AVAILABLE:
            print("❌ Data loader kullanılamıyor, demo veriye geçiliyor...")
            documents = [Document(
                page_content="""
                Supply Chain Management (Tedarik Zinciri Yönetimi): 
                Hammaddeden nihai ürüne kadar olan tüm süreçleri planlayan, 
                yürüten ve kontrol eden disiplindir.
                
                Ana bileşenler:
                - Tedarik (Procurement)
                - Üretim (Manufacturing) 
                - Dağıtım (Distribution)
                - Lojistik (Logistics)
                - Envanter Yönetimi (Inventory Management)
                """,
                metadata={"source": "demo_data", "filename": "demo_supply_chain.txt"}
            )]
        else:
            # ÖZEL: Tutku dataset'ini yükle
            documents = load_tutku_supply_chain_data()
            
            if not documents:
                print("❌ Hiç veri yüklenemedi, demo veri kullanılıyor...")
                documents = [Document(
                    page_content="Supply Chain Management: Tedarik zinciri yönetimi temel kavramları.",
                    metadata={"source": "demo_data", "filename": "fallback.txt"}
                )]
        
        # PDF'ler için optimize edilmiş metin bölücü
        # Büyük dokümanları daha küçük parçalara böler
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,      # Her parçanın maksimum boyutu
            chunk_overlap=CHUNK_OVERLAP, # Parçalar arası örtüşme
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # PDF metinleri için
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ {len(chunks)} metin parçası oluşturuldu")
        return chunks
    
    def setup_vector_store(self, chunks):
        """
        VEKTÖR VER
