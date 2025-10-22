"""
DATA_LOADER.PY - PDF & ZIP DESTEKLİ HUGGING FACE LOADER


PDF ve ZIP dosyalarını işleyebilir.
"""

import os
import tempfile
import zipfile
from datasets import load_dataset
from langchain.schema import Document
import pandas as pd
from config import HF_DATASET_NAME, HF_SPLIT

# PDF işleme kütüphaneleri
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ PyPDF2 yüklü değil, PDF desteği kapalı")

class TutkuSupplyChainLoader:
    def __init__(self):
        self.dataset = None
        self.documents = []
    
    def load_dataset_from_hf(self):
        """Tutku Özdeniz dataset'ini Hugging Face'ten yükle"""
        print("📥 Hugging Face'ten dataset yükleniyor: tutkuozdeniz/supply-chain-management")
        
        try:
            # Dataset'i yükle
            self.dataset = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
            print(f"✅ Dataset yüklendi: {len(self.dataset)} dosya")
            return self.dataset
            
        except Exception as e:
            print(f"❌ Dataset yüklenemedi: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content):
        """PDF içeriğinden metin çıkar"""
        if not PDF_SUPPORT:
            return "PDF desteği yüklü değil"
        
        try:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name
            
            # PDF'den metin çıkar
            with open(tmp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Geçici dosyayı temizle
            os.unlink(tmp_path)
            return text.strip()
            
        except Exception as e:
            print(f"❌ PDF işleme hatası: {e}")
            return f"PDF işlenemedi: {str(e)}"
    
    def process_zip_file(self, zip_content):
        """ZIP dosyasını işle ve içindeki dosyaları çıkar"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(zip_content)
                tmp_path = tmp_file.name
            
            extracted_texts = []
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.lower().endswith('.pdf'):
                        with zip_ref.open(file_info) as pdf_file:
                            pdf_content = pdf_file.read()
                            text = self.extract_text_from_pdf(pdf_content)
                            if text and "PDF işlenemedi" not in text:
                                extracted_texts.append({
                                    'filename': file_info.filename,
                                    'content': text
                                })
            
            os.unlink(tmp_path)
            return extracted_texts
            
        except Exception as e:
            print(f"❌ ZIP işleme hatası: {e}")
            return []
    
    def process_dataset_files(self):
        """Dataset'teki tüm dosyaları işle ve metne çevir"""
        if self.dataset is None:
            print("❌ Önce dataset yükleyin")
            return []
        
        print("🔍 Dataset dosyaları işleniyor...")
        
        for i, item in enumerate(self.dataset):
            try:
                # Dosya bilgilerini al
                file_name = item.get('file_name', f'file_{i}')
                content = item.get('content', b'')
                
                if not content:
                    continue
                
                # Dosya tipine göre işle
                if file_name.lower().endswith('.pdf'):
                    print(f"   📄 PDF işleniyor: {file_name}")
                    text = self.extract_text_from_pdf(content)
                    if text and "PDF işlenemedi" not in text:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": f"pdf_{i}",
                                "filename": file_name,
                                "type": "pdf"
                            }
                        )
                        self.documents.append(doc)
                
                elif file_name.lower().endswith('.zip'):
                    print(f"   📦 ZIP işleniyor: {file_name}")
                    extracted_files = self.process_zip_file(content)
                    for extracted in extracted_files:
                        doc = Document(
                            page_content=extracted['content'],
                            metadata={
                                "source": f"zip_{i}",
                                "filename": f"{file_name}/{extracted['filename']}",
                                "type": "zip_content"
                            }
                        )
                        self.documents.append(doc)
                
                elif file_name.lower().endswith('.txt'):
                    print(f"   📝 TXT işleniyor: {file_name}")
                    text = content.decode('utf-8') if isinstance(content, bytes) else str(content)
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": f"txt_{i}",
                            "filename": file_name,
                            "type": "text"
                        }
                    )
                    self.documents.append(doc)
                    
            except Exception as e:
                print(f"❌ {file_name} işlenirken hata: {e}")
                continue
        
        print(f"✅ {len(self.documents)} adet doküman işlendi")
        return self.documents
    
    def get_dataset_stats(self):
        """Dataset istatistiklerini göster"""
        if not self.documents:
            return "❌ Henüz doküman işlenmemiş"
        
        stats = {
            'total_documents': len(self.documents),
            'pdf_count': len([d for d in self.documents if d.metadata.get('type') == 'pdf']),
            'zip_content_count': len([d for d in self.documents if d.metadata.get('type') == 'zip_content']),
            'text_count': len([d for d in self.documents if d.metadata.get('type') == 'text']),
            'total_text_length': sum(len(d.page_content) for d in self.documents)
        }
        
        return f"""
        📊 Dataset İstatistikleri:
        - Toplam doküman: {stats['total_documents']}
        - PDF dosyaları: {stats['pdf_count']}
        - ZIP içeriği: {stats['zip_content_count']}
        - Text dosyaları: {stats['text_count']}
        - Toplam metin: {stats['total_text_length']} karakter
        """

# KOLAY KULLANIM FONKSİYONU
def load_tutku_supply_chain_data():
    """Tutku Özdeniz supply chain verilerini yükle ve işle"""
    loader = TutkuSupplyChainLoader()
    
    # 1. Dataset'i HF'ten yükle
    dataset = loader.load_dataset_from_hf()
    
    if dataset is None:
        print("❌ HF dataset yüklenemedi")
        return []
    
    # 2. Dosyaları işle ve metne çevir
    documents = loader.process_dataset_files()
    
    # 3. İstatistikleri göster
    print(loader.get_dataset_stats())
    
    return documents

# Test
if __name__ == "__main__":
    documents = load_tutku_supply_chain_data()
    if documents:
        print(f"\n📖 İlk doküman önizleme:")
        print(f"Kaynak: {documents[0].metadata.get('filename', 'bilinmiyor')}")
        print(f"İçerik: {documents[0].page_content[:200]}...")
