# 🤖 Supply Chain Management AI Chatbot

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://gemini.google.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

*Akıllı Tedarik Zinciri Asistanı - AI destekli eğitim ve danışmanlık platformu*

</div>

## 🎯 Proje Hakkında

**Supply Chain Management AI Chatbot**, tedarik zinciri yönetimi konularında uzmanlaşmış, RAG (Retrieval-Augmented Generation) mimarisi ile geliştirilmiş bir yapay zeka asistanıdır. Kullanıcıların tedarik zinciri, lojistik, envanter yönetimi ve ilgili konulardaki sorularını yanıtlar.

### 🌟 Özellikler

- 🤖 **Google Gemini AI** destekli akıllı sohbet
- 📚 **Hugging Face Dataset** entegrasyonu
- 🔍 **RAG Mimarisi** ile doğru ve güvenilir cevaplar
- 📄 **PDF & ZIP dosya desteği** - otomatik metin çıkarma
- 🌐 **Streamlit web arayüzü** - kullanıcı dostu
- 🏷️ **Türkçe dil desteği** - yerelleştirilmiş deneyim

## 🏗️ Mimari

```mermaid
graph TB
    A[Kullanıcı Sorusu] --> B[Streamlit Arayüzü]
    B --> C[RAG Pipeline]
    C --> D[Hugging Face Dataset]
    D --> E[PDF/ZIP İşleme]
    E --> F[Vektör Veritabanı]
    F --> G[Google Gemini AI]
    G --> H[Cevap]
    H --> B
🚀 Hızlı Başlangıç
Ön Koşullar
Python 3.8+

Google Gemini API Key (Almak için)

Kurulum
Repository'yi klonlayın:

bash
git clone https://github.com/tutkuozdeniz0/akbank-generative-ai-chatbot.git
cd akbank-generative-ai-chatbot
Gerekli kütüphaneleri yükleyin:

bash
pip install -r requirements.txt
Çevre değişkenlerini ayarlayın:

bash
# .env dosyası oluşturun
echo "GEMINI_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .env
Uygulamayı başlatın:

bash
streamlit run app.py
🐳 Docker ile Çalıştırma
bash
# Docker image oluştur
docker build -t supply-chain-chatbot .

# Container başlat
docker run -p 8501:8501 supply-chain-chatbot
📁 Proje Yapısı
text
akbank-generative-ai-chatbot/
├── app.py                 # Ana Streamlit uygulaması
├── chain.py              # RAG pipeline ve chatbot sınıfı
├── config.py             # Yapılandırma ayarları
├── data_loader.py        # Hugging Face dataset yükleyici
├── requirements.txt      # Python bağımlılıkları
├── Dockerfile           # Docker konteyner yapılandırması
├── .gitignore           # Git ignore ayarları
└── README.md            # Bu dosya
🔧 Kullanım
Yerel Geliştirme
Uygulamayı başlattıktan sonra tarayıcınızda http://localhost:8501 adresine gidin

Sol sidebar'dan Google Gemini API key'inizi girin

"Chatbot'u Başlat" butonuna tıklayın

Supply chain ile ilgili sorularınızı sorun!

Örnek Sorular
"Tedarik zinciri yönetimi nedir?"

"Just-In-Time üretim modelini açıklar mısın?"

"Stok devir hızı nasıl hesaplanır?"

"Lojistik ve dağıtım arasındaki farklar nelerdir?"

"Tedarikçi performans metriklari nelerdir?"

🎨 Özellikler
🤖 Akıllı Yanıt Sistemi
Bağlamsal anlama

Çok dilli destek

Kaynak gösterimi

Gerçek zamanlı işleme

📊 Veri İşleme
PDF doküman işleme

ZIP arşiv desteği

Otomatik metin çıkarma

Vektör tabanlı arama

🔒 Güvenlik
API key güvenliği

Veri şifreleme

Güvenli dosya işleme

🌐 Deployment
Streamlit Cloud
bash
# Otomatik deploy için:
# 1. GitHub repo'yu Streamlit Cloud'a bağla
# 2. Secrets kısmına GEMINI_API_KEY ekle
# 3. Otomatik deploy başlayacak
Hugging Face Spaces
bash
# HF Spaces için:
# 1. Yeni Space oluştur
# 2. Repository'yi bağla
# 3. Secrets kısmına API key ekle
🛠️ Geliştirme
Bağımlılıkları Güncelleme
bash
pip freeze > requirements.txt
Test Etme
bash
python -m pytest tests/
Docker Build
bash
docker build -t supply-chain-chatbot .
docker run -p 8501:8501 --env-file .env supply-chain-chatbot
📊 Dataset
Proje, Hugging Face üzerinde barındırılan özel bir supply chain dataset'i kullanmaktadır:

Dataset: tutkuozdeniz/supply-chain-management

İçerik: PDF makaleler, sunumlar, eğitim materyalleri

Format: PDF, ZIP, metin dosyaları

🤝 Katkıda Bulunma
Katkılarınızı memnuniyetle karşılıyoruz! Lütfen:

Fork edin

Feature branch oluşturun (git checkout -b feature/AmazingFeature)

Commit edin (git commit -m 'Add some AmazingFeature')

Push edin (git push origin feature/AmazingFeature)

Pull Request oluşturun

📝 Lisans
Bu proje MIT lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

👨‍💻 Geliştirici
Tutku Özdeniz

GitHub: @tutkuozdeniz0

Hugging Face: tutkuozdeniz

🙏 Teşekkür
Google Gemini AI ekibine

Hugging Face ekibine

Streamlit ekibine

Akbank Generative AI programına

<div align="center">

</div> ```
