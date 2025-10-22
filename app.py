"""
APP.PY - SUPPLY CHAIN MANAGEMENT AI CHATBOT
Ana Streamlit uygulaması - Tedarik Zinciri Yönetimi için AI asistanı
"""
# ==============================================================================
# BÖLÜM 0: GEREKLİ KÜTÜPHANELERİ YÜKLE
# Colab ortamında bu hücreyi ilk çalıştırmalısın.
# ==============================================================================
print("0. Gerekli kütüphaneler yükleniyor...")
# Langchain'in son sürümleriyle uyumluluk için bu paketleri güncel tutuyoruz.
# Ayrıca PDF işleme için 'unstructured[pdf]' ve 'pypdf'nin doğru kurulduğundan emin olalım.
!pip install -q transformers datasets langchain langchain-community faiss-cpu sentence-transformers pypdf chromadb gradio unstructured[pdf] huggingface_hub
print("Kütüphaneler yüklendi.")

import os
import requests
import gradio as gr
from datasets import load_dataset
# langchain_community'den import etmemiz gereken modülleri güncelledik
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, UnstructuredFileLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob
import shutil

# ==============================================================================
# BÖLÜM 1: YAPILANDIRMA AYARLARI
# Lütfen buradaki yer tutucuları kendi bilgilerinizle doldurun.
# ==============================================================================

# 1. Hugging Face Veri Seti Adı (Terim/Tanım için)
HUGGINGFACE_DATASET_NAME = "tutkuozdeniz0/tedarik-zinciri-terimleri" # Boş bırakmak istersen "" yapabilirsin.

# 2. Hugging Face API Token
HUGGINGFACE_API_TOKEN = ""

# 3. Embeddings Modeli Adı (Metinleri vektörlere çevirmek için)
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Türkçe destekli

# 4. LLM (Büyük Dil Modeli) Adı (Cevap üretmek için)
# Performans ve kalite dengesi için flan-t5-large'ı öneririm eğer GPU yeterliyse.
# Eğer hala çok yavaşsa 'base' veya 'small' deneyebilirsiniz.
LLM_MODEL_NAME = "google/flan-t5-large" # Daha iyi kalite için 'large'
# LLM_MODEL_NAME = "google/flan-t5-base" # Orta kalite/hız
# LLM_MODEL_NAME = "google/flan-t5-small" # Düşük kalite/yüksek hız

LLM_TEMPERATURE = 0.4 # Daha tutarlı ve az spekülatif cevaplar için biraz düşürdük
LLM_MAX_NEW_TOKENS = 700 # Cevap uzunluğunu artırdık, cümlelerin yarım kalmaması için
# LLM_MAX_LENGTH, HuggingFaceHub ile kullanılıyordu. HuggingFacePipeline ile 'max_new_tokens' tercih edilir.

# ChromaDB için kalıcı depolama dizini
PERSIST_DIRECTORY = 'chroma_db'

# Metin parçalama ayarları (PDF'ler için daha büyük parçalar daha iyi olabilir)
CHUNK_SIZE = 800 # PDF'lerde genelde daha uzun cümleler/paragraflar olur
CHUNK_OVERLAP = 150 # Parçalar arası bağlamı korumak için üst üste binme

# Retriever (Parça çekici) ayarları
SEARCH_K = 4 # Daha fazla ilgili parça çekmek için artırdık

# API token ayarı
if HUGGINGFACE_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN
    print("Hugging Face API token başarıyla ayarlandı.")
else:
    print("Uyarı: Hugging Face API token ayarlanmadı. Bazı Hugging Face Hub modelleri kısıtlı olabilir.")

print("\nYapılandırma ayarları tamamlandı.")

# ==============================================================================
# BÖLÜM 2: HUGGING FACE VERİ SETİNİ VE YEREL PDF/DİĞER DOSYALARI YÜKLE
# ==============================================================================
print(f"\n2. Veri kaynakları yükleniyor ve işleniyor...")
all_documents = []

# --- Hugging Face Veri Setini Yükle (Terim/Tanım için) ---
if HUGGINGFACE_DATASET_NAME:
    try:
        dataset = load_dataset(HUGGINGFACE_DATASET_NAME)
        print(f"Hugging Face veri seti '{HUGGINGFACE_DATASET_NAME}' başarıyla yüklendi.")
        print(f"Veri setindeki splitler: {list(dataset.keys())}")

        target_split = 'train'
        if target_split not in dataset and 'data' in dataset:
            target_split = 'data'
        elif target_split not in dataset:
            print(f"Uyarı: Hugging Face veri setinde '{target_split}' veya 'data' split'i bulunamadı.")
            target_split = None

        if target_split:
            for i, entry in enumerate(dataset[target_split]):
                term = entry.get('term', '')
                definition = entry.get('definition', '')
                
                document_content = ""
                if term:
                    document_content += f"Terim: {term}\n"
                if definition:
                    document_content += f"Tanım: {definition}\n"
                
                # Diğer sütunlar varsa buraya ekleyin:
                # explanation = entry.get('explanation', '')
                # if explanation:
                #     document_content += f"Açıklama: {explanation}\n"
                # example = entry.get('example', '')
                # if example:
                #     document_content += f"Örnek: {example}\n"

                if document_content.strip():
                    metadata = {
                        "source": f"HF Dataset: {HUGGINGFACE_DATASET_NAME} (Entry ID: {i})",
                        "term": term, # Terimi metadata olarak saklamak faydalı olabilir
                    }
                    all_documents.append(Document(page_content=document_content.strip(), metadata=metadata))
        
    except Exception as e:
        print(f"Hugging Face veri seti yükleme veya işleme hatası: {e}")
        print("Lütfen HUGGINGFACE_DATASET_NAME'in doğru olduğundan ve erişilebilir olduğundan emin olun.")

# --- Yerel PDF ve Diğer Dosyaları Yükle ---
print("\nYerel PDF ve diğer ek dosyalar yükleniyor...")

# Colab'e yüklediğiniz tüm PDF dosyalarının yollarını buraya ekleyin.
# Colab'de dosya yükledikten sonra sol paneldeki dosya simgesine tıklayıp
# dosya üzerinde sağ tıklayıp "Copy path" seçeneğini kullanabilirsiniz.
# Veya tüm PDF'leri dinamik olarak bulabiliriz:
local_data_dir = "/content/" # Colab'de dosyaların yüklendiği varsayılan dizin
pdf_files = glob.glob(os.path.join(local_data_dir, '**', '*.pdf'), recursive=True)
# .gitattributes ve .zip dosyalarını elemiyoruz
pdf_files = [f for f in pdf_files if not os.path.basename(f).startswith('.git') and not f.endswith('.zip')]

if not pdf_files:
    print("Uyarı: Belirtilen dizinde hiç PDF dosyası bulunamadı. Lütfen yolları kontrol edin.")
else:
    print(f"Toplam {len(pdf_files)} PDF dosyası bulundu: {pdf_files}")

# PDF yükleyicisi
for file_path in pdf_files:
    try:
        # unstructured[pdf] kurulumu sayesinde UnstructuredFileLoader daha robusttur
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        for doc in docs:
            # Kaynak bilgisini daha açıklayıcı yapalım
            doc.metadata['source'] = f"Yerel PDF: {os.path.basename(file_path)}"
            all_documents.append(doc)
        print(f"Yüklendi: {file_path}")
    except Exception as e:
        print(f"PDF yükleme hatası {file_path}: {e}")

# Eğer Hugging Face'den veya yerel dosyalardan hiçbir belge yüklenememişse, manuel örnekleri ekle
if not all_documents:
    print("Önemli Uyarı: Hiçbir belge yüklenemedi. Chatbot boş bir bilgi tabanı ile çalışacaktır.")
    print("Lütfen Hugging Face veri setinizi ve yerel dosyalarınızı kontrol edin.")
    print("Test amacıyla manuel olarak örnek tedarik zinciri belgeleri ekleniyor...")
    all_documents.append(Document(page_content="Tedarik zinciri, bir ürünün veya hizmetin ham maddeden son tüketiciye ulaşana kadar geçen tüm süreçlerini kapsar.", metadata={"source": "Manuel Eklenen Belge", "term": "Tedarik Zinciri"}))
    all_documents.append(Document(page_content="Lojistik, ürünlerin ve hizmetlerin tedarik zinciri boyunca etkin ve verimli bir şekilde hareket etmesini yöneten süreçtir.", metadata={"source": "Manuel Eklenen Belge", "term": "Lojistik"}))
    all_documents.append(Document(page_content="Envanter yönetimi, bir işletmenin stok seviyelerini optimize etme ve kontrol etme uygulamasıdır.", metadata={"source": "Manuel Eklenen Belge", "term": "Envanter Yönetimi"}))
    all_documents.append(Document(page_content="Tedarik zinciri yönetiminin temel amacı, verimliliği artırmak ve maliyetleri düşürmektir.", metadata={"source": "Manuel Eklenen Belge", "term": "Tedarik Zinciri Amacı"}))
    all_documents.append(Document(page_content="Yeşil tedarik zinciri yönetimi, çevresel sürdürülebilirliği tedarik zinciri süreçlerine entegre etmeyi hedefler.", metadata={"source": "Manuel Eklenen Belge", "term": "Yeşil Tedarik Zinciri"}))


print(f"Toplam {len(all_documents)} belge yüklendi.")

# ==============================================================================
# BÖLÜM 3: METİNLERİ PARÇALARA BÖL (CHUNK ET)
# Büyük metinleri daha küçük, yönetilebilir parçalara ayırma.
# ==============================================================================
print("\n3. Metinler parçalara bölünüyor...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(all_documents)
print(f"Toplam {len(chunks)} parça oluşturuldu.")
print("İlk parça örneği:")
if chunks:
    print(chunks[0].page_content[:200] + "...")
else:
    print("Hiç parça oluşturulamadı. Lütfen belgelerinizi kontrol edin.")

# ==============================================================================
# BÖLÜM 4: EMBEDDINGS OLUŞTUR VE VEKTÖR VERİTABANINA KAYDET (CHROMA)
# Metin parçalarını vektörlere dönüştürme ve veritabanına indeksleme.
# ==============================================================================
print("\n4. Embeddings oluşturuluyor ve ChromaDB'ye kaydediliyor...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

if os.path.exists(PERSIST_DIRECTORY):
    print(f"Mevcut ChromaDB dizini '{PERSIST_DIRECTORY}' temizleniyor.")
    shutil.rmtree(PERSIST_DIRECTORY)
    
if chunks:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    print(f"ChromaDB '{PERSIST_DIRECTORY}' dizinine kaydedildi. Toplam {len(chunks)} parça indekslendi.")
else:
    print("Uyarı: Hiç parça olmadığı için ChromaDB oluşturulamadı.")
    # Hiç parça olmasa bile boş bir vektör veritabanı oluşturmak Gradio'nun çalışması için iyi.
    vectordb = Chroma.from_documents(
        documents=[Document(page_content="Genel bilgi için boş vektör veritabanı. Tedarik zinciri dersi asistanı için hazırlanmıştır.", metadata={"source": "Boş Veritabanı", "term": "Genel Bilgi"})],
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    print("Boş bir ChromaDB oluşturuldu. Chatbot performansı etkilenebilir.")


# ==============================================================================
# BÖLÜM 5: CHATBOT ZİNCİRİNİ OLUŞTUR (RAG MİMARİSİ)
# Kullanıcının sorusuna cevap verecek LLM ve Retriever'ı birleştirme.
# ==============================================================================
print("\n5. Chatbot zinciri oluşturuluyor...")

try:
    # Modelin Colab GPU'ya yüklendiğinden emin olmak için biraz daha verbose olalım
    print(f"'{LLM_MODEL_NAME}' modeli ve tokenizer yükleniyor. Bu biraz zaman alabilir...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    
    # Modelin GPU'ya taşınması
    if torch.cuda.is_available(): # PyTorch GPU kontrolü için 'import torch' gerekli
        model.to("cuda")
        print("Model GPU'ya taşındı.")
    else:
        print("GPU bulunamadı veya kullanılamıyor, model CPU'da çalışacak. Bu yavaş olabilir.")


    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=LLM_MAX_NEW_TOKENS, # Maksimum token sayısı
        temperature=LLM_TEMPERATURE,
        # num_beams=5, # Daha iyi sonuçlar için beam search eklenebilir, ama daha yavaş.
        # early_stopping=True,
        # device=0 if torch.cuda.is_available() else -1 # Pipeline'ı GPU'ya yönlendir
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"LLM modeli '{LLM_MODEL_NAME}' (HuggingFacePipeline ile) başarıyla yüklendi.")

    retriever = vectordb.as_retriever(search_kwargs={"k": SEARCH_K})

    # Daha iyi Türkçe kullanım, noktalama ve cümleleri kesmeme için detaylı prompt
    template = """Sen, tedarik zinciri derslerine yardımcı olan, bilgilendirici ve açıklayıcı bir asistansın.
    Aşağıda sana verilen "Bağlam" kısmındaki bilgileri kullanarak kullanıcının sorusunu Türkçe olarak, detaylı, akıcı, dilbilgisi kurallarına ve noktalama işaretlerine uygun bir şekilde cevapla.
    Cevaplarını her zaman tam cümlelerle ve ilgili bilgileri birleştirerek oluştur. Cümleleri asla yarıda kesme.
    Eğer verilen bağlamda doğrudan bir cevap bulamıyorsan, kibarca "Üzgünüm, bu konu hakkında mevcut bilgilerimde yeterli detay bulunmuyor." şeklinde cevap ver. Bağlam dışından bilgi uydurma.

    Bağlam:
    {context}

    Soru: {question}
    Cevap:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' yeterli gelmezse 'refine' veya 'map_reduce' deneyin.
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("RetrievalQA zinciri başarıyla oluşturuldu.")

except Exception as e:
    print(f"LLM veya RetrievalQA zinciri oluşturulurken bir hata oluştu: {e}")
    print("Lütfen Hugging Face API anahtarınızın doğru olduğundan ve seçtiğiniz modelin erişilebilir olduğundan emin olun.")
    print("Modelin Colab ortamında yeterli belleğe sahip olduğundan emin olun (GPU kullanımı).")
    print("PyTorch'un (torch kütüphanesi) yüklü olduğundan emin olun.")
    print("Devam etmek için dummy bir cevap fonksiyonu kullanılacaktır.")
    qa_chain = None

def get_chatbot_response(question):
    if qa_chain:
        try:
            result = qa_chain({"query": question})
            response_text = result["result"].strip()
            source_docs = result.get("source_documents", [])
            
            sources_info = ""
            if source_docs:
                sources_info = "\n\n--- Kaynak Bilgileri ---\n"
                for i, doc in enumerate(source_docs):
                    source_name = doc.metadata.get('source', 'Bilinmiyor')
                    term_name = doc.metadata.get('term', None) # Metadata'daki terimi alıyoruz
                    
                    # Kaynağı daha iyi temsil etmek için başlık/terim kullanıyoruz
                    source_display = f"{source_name}"
                    if term_name: # Eğer metadata'da terim varsa, onu da ekleyelim
                        source_display = f"{term_name} ({source_name})"
                    
                    sources_info += f"- {source_display}\n"
                    # İlgili içeriğin tamamını değil, ilk 300 karakterini gösteriyoruz
                    # Kullanıcıya bir fikir vermesi için yeterli.
                    sources_info += f"  İlgili İçerik: {doc.page_content[:300]}...\n\n"
            
            # Eğer model 'Üzgünüm' diye cevap verdiyse, kaynak göstermeyelim
            if "üzgünüm" in response_text.lower() or "yeterli detay bulunmuyor" in response_text.lower():
                 return response_text
            else:
                 return response_text + sources_info
        except Exception as e:
            return f"Cevap üretilirken bir hata oluştu: {e}"
    else:
        return "Chatbot başlatılamadı. Lütfen yapılandırma adımlarını kontrol edin ve tekrar deneyin."

print("\nChatbot hazır.")

# ==============================================================================
# BÖLÜM 6: BASİT BİR WEB ARAYÜZÜ OLUŞTUR (GRADIO)
# Chatbot'u kullanıcı etkileşimine açmak için web arayüzü.
# ==============================================================================
print("\n6. Gradio web arayüzü başlatılıyor...")

iface = gr.Interface(
    fn=get_chatbot_response,
    inputs=gr.Textbox(lines=3, placeholder="Tedarik zinciri dersi hakkında bir soru sorun...", label="Sorunuz"),
    outputs=gr.Textbox(lines=12, label="Chatbot Cevabı"), # Çıkış kutusunu biraz daha büyüttük
    title="Tedarik Zinciri Dersi Asistanı Chatbot (RAG)",
    description="Hugging Face veri setinden ve yerel PDF'lerden bilgi çeken bir RAG tabanlı tedarik zinciri dersi asistanı. Lütfen sorularınızı net ve Türkçe olarak sorun!"
)

iface.launch(share=True)
print("\nGradio arayüzü başlatıldı. Lütfen yukarıdaki Public URL'yi kontrol edin.")
print("\nColab ortamında Gradio arayüzünü kapatmak için bu hücrenin çalışmasını durdurmanız yeterlidir.")

         
          
