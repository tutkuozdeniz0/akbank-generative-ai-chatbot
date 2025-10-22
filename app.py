"""
APP.PY - SUPPLY CHAIN MANAGEMENT AI CHATBOT
Ana Streamlit uygulaması - Tedarik Zinciri Yönetimi için AI asistanı
"""

import streamlit as st
import os
from chain import SupplyChainChatbot
from config import GEMINI_API_KEY

# Sayfa ayarları
st.set_page_config(
    page_title="Supply Chain AI Asistanı",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="expanded"
)

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık ve açıklama
    st.title("🚚 Supply Chain Management AI Asistanı")
    st.markdown("---")
    st.markdown("""
    🤖 **Tedarik Zinciri ve Lojistik konularında uzman AI asistanı**
    
    Supply chain yönetimi, lojistik, envanter optimizasyonu ve ilgili konularda 
    sorularınızı yanıtlayabilirim. Hugging Face'teki özel datasetimden 
    güncel bilgilerle size yardımcı oluyorum!
    """)
    
    # Session state initialization
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'is_ready' not in st.session_state:
        st.session_state.is_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - Ayarlar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        st.markdown("---")
        
        # API Key girişi
        st.subheader("🔑 API Yapılandırması")
        api_key = st.text_input(
            "Google Gemini API Key:",
            value=GEMINI_API_KEY or "",
            type="password",
            placeholder="AIzaSyB...",
            help="https://aistudio.google.com/ adresinden alabilirsiniz"
        )
        
        # Chatbot'u başlat butonu
        st.markdown("---")
        if st.button("🚀 Chatbot'u Başlat", use_container_width=True, type="primary"):
            if not api_key:
                st.error("❌ Lütfen Google Gemini API key giriniz")
            else:
                with st.spinner("🤖 Supply Chain asistanı hazırlanıyor... Bu işlem birkaç dakika sürebilir."):
                    try:
                        # Environment variable ayarla
                        os.environ["GEMINI_API_KEY"] = api_key
                        
                        # Chatbot'u başlat
                        st.session_state.chatbot = SupplyChainChatbot()
                        st.session_state.chatbot.initialize_chatbot()
                        st.session_state.is_ready = True
                        st.session_state.chat_history = []
                        
                        st.success("✅ Chatbot başarıyla hazırlandı!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Hata oluştu: {str(e)}")
                        st.info("🔧 API key'inizi ve internet bağlantınızı kontrol edin")
        
        # Bilgi bölümü
        st.markdown("---")
        with st.expander("ℹ️ Kullanım Kılavuzu"):
            st.markdown("""
            **📖 Nasıl Kullanılır:**
            1. API key'inizi girin
            2. "Chatbot'u Başlat" butonuna tıklayın
            3. Sorunuzu yazın ve Enter'a basın
            
            **💡 Örnek Sorular:**
            - Tedarik zinciri nedir?
            - Just-In-Time modeli nasıl çalışır?
            - Stok optimizasyonu yöntemleri nelerdir?
            - Lojistik maliyetleri nasıl azaltılır?
            """)
        
        # Dataset bilgisi
        with st.expander("📊 Dataset Bilgisi"):
            st.markdown("""
            **Veri Kaynağı:** Hugging Face
            **Dataset:** tutkuozdeniz/supply-chain-management
            **İçerik:** PDF makaleler, sunumlar, eğitim materyalleri
            **Konular:** Supply chain, lojistik, envanter yönetimi
            """)
    
    # Ana içerik alanı
    if not st.session_state.is_ready:
        # Başlatılmamış durum
        st.info("👈 Lütfen sol taraftan API key'inizi girin ve **Chatbot'u Başlat** butonuna tıklayın")
        
        # Örnek sorular
        st.markdown("### 💡 Örnek Sorular:")
        example_questions = [
            "📦 **Tedarik zinciri yönetimi nedir ve neden önemlidir?**",
            "⚡ **Just-In-Time (JIT) üretim modelini açıklar mısın?**",
            "📊 **Stok devir hızı nasıl hesaplanır ve optimize edilir?**",
            "🚛 **Lojistik ve dağıtım arasındaki temel farklar nelerdir?**",
            "💰 **Tedarik zinciri maliyet optimizasyonu nasıl yapılır?**"
        ]
        
        for question in example_questions:
            st.write(question)
            
        # Teknoloji stack bilgisi
        st.markdown("---")
        st.markdown("### 🛠️ Teknoloji Stack'i")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AI Model", "Google Gemini")
        with col2:
            st.metric("Framework", "Streamlit")
        with col3:
            st.metric("Mimari", "RAG")
        with col4:
            st.metric("Dataset", "Hugging Face")
            
    else:
        # Chatbot hazır - sohbet arayüzü
        st.success("🎉 Supply Chain asistanı hazır! Aşağıdan sorularınızı sorabilirsiniz.")
        
        # Chat history gösterimi
        if st.session_state.chat_history:
            st.markdown("### 💬 Sohbet Geçmişi")
            for i, (question, answer) in enumerate(st.session_state.chat_history[-5:], 1):
                with st.expander(f"🗨️ Soru {i}: {question[:50]}...", expanded=False):
                    st.markdown(f"**💭 Soru:** {question}")
                    st.markdown(f"**🤖 Cevap:** {answer}")
        
        # Yeni soru girişi
        st.markdown("---")
        st.markdown("### 💭 Yeni Soru Sorun")
        
        question = st.text_area(
            "Supply chain ile ilgili sorunuz:",
            placeholder="Örnek: Tedarik zinciri risk yönetimi nasıl yapılır?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("📨 Soruyu Gönder", use_container_width=True) and question:
                with st.spinner("🔍 Supply chain veritabanında araştırıyorum..."):
                    try:
                        # Chatbot'tan cevap al
                        answer = st.session_state.chatbot.ask_question(question)
                        
                        # Sohbet geçmişine ekle
                        st.session_state.chat_history.append((question, answer))
                        
                        # Cevabı göster
                        st.markdown("### 💬 Cevap:")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"❌ Hata oluştu: {str(e)}")
                        st.info("🔄 Lütfen tekrar deneyin veya chatbot'u yeniden başlatın")
        
        with col2:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()
        
        # Chat temizleme
        if st.session_state.chat_history:
            if st.button("🗑️ Sohbeti Temizle", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()
      
