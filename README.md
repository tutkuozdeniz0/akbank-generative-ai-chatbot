# 🤖 Supply Chain Management AI Chatbot

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://gemini.google.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**Akıllı Tedarik Zinciri Asistanı - AI destekli eğitim ve danışmanlık platformu**

</div>

## 🎯 Proje Hakkında

**Supply Chain Management AI Chatbot**, tedarik zinciri yönetimi konularında uzmanlaşmış, RAG (Retrieval-Augmented Generation) mimarisi ile geliştirilmiş bir yapay zeka asistanıdır. 

Kullanıcıların tedarik zinciri, lojistik, envanter yönetimi ve ilgili konulardaki sorularını, Hugging Face üzerindeki özel dataset'ten alınan güncel bilgilerle yanıtlar.

### 🌟 Öne Çıkan Özellikler

- 🤖 **Google Gemini AI** destekli akıllı sohbet
- 📚 **Hugging Face Dataset** entegrasyonu
- 🔍 **RAG Mimarisi** ile doğru ve güvenilir cevaplar
- 📄 **PDF & ZIP dosya desteği** - otomatik metin çıkarma
- 🌐 **Streamlit web arayüzü** - kullanıcı dostu
- 🏷️ **Türkçe dil desteği** - yerelleştirilmiş deneyim
- 🔒 **Güvenli API yönetimi** - environment variables

## 🏗️ Sistem Mimarisi

### Akış Diyagramı

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
