

import os
from dotenv import load_dotenv

load_dotenv()

# API KEY'LER
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# TUTKU'OZDENIZ SUPPLY CHAIN DATASET - ÖZEL
HF_DATASET_NAME = "tutkuozdeniz/supply-chain-management"
HF_SPLIT = "train"

# PDF İŞLEME AYARLARI
CHUNK_SIZE = 800    # PDF'ler için optimize edildi
CHUNK_OVERLAP = 150
MODEL_NAME = "gemini-pro"
VECTOR_DB_PATH = "vector_store"

# DOSYA TİPLERİ
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.zip']
