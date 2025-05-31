import os
import json
import faiss
import pickle
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer

# --- 設定區 ---
KNOWLEDGE_DIR = "knowledge_files"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
INDEX_PATH = "faiss_index.index"
SOURCE_PATH = "doc_sources.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# --- 初始化模型 ---
embedder = SentenceTransformer(EMBEDDING_MODEL)

# --- 資料儲存區 ---
docs = []
sources = []

# --- 各類型讀取函式 ---
def read_pdf(path):
    with fitz.open(path) as pdf:
        return [line for page in pdf for line in page.get_text().split("\n") if line.strip()]

def read_docx(path):
    return [para.text for para in Document(path).paragraphs if para.text.strip()]

# def read_csv(path):
#     return pd.read_csv(path).astype(str).values.flatten().tolist()

def read_csv(path):
    df = pd.read_csv(path)
    lines = []
    for i, row in df.iterrows():
        description = ", ".join(f"{col.strip()}：{str(val).strip()}" for col, val in row.items())
        lines.append(f"第 {i+1} 筆資料：{description}")
    return lines

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_json(path):
    def extract_text(obj):
        if isinstance(obj, dict):
            return sum([extract_text(v) for v in obj.values()], [])
        elif isinstance(obj, list):
            return sum([extract_text(i) for i in obj], [])
        elif isinstance(obj, str):
            return [obj]
        return []
    with open(path, "r", encoding="utf-8") as f:
        return extract_text(json.load(f))

def read_md(path):
    return read_txt(path)

READER_FUNCS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".csv": read_csv,
    ".txt": read_txt,
    ".json": read_json,
    ".md": read_md,
}

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

# --- 讀取並處理資料 ---
for filename in os.listdir(KNOWLEDGE_DIR):
    path = os.path.join(KNOWLEDGE_DIR, filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in READER_FUNCS:
        print(f"! 不支援的檔案格式：{filename}")
        continue

    lines = READER_FUNCS[ext](path)

    for line in lines:
        for chunk in chunk_text(line):
            if chunk:
                docs.append(chunk)
                sources.append(filename)

# --- 向量化並建立索引 ---
print("📦 正在編碼向量...")
doc_embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

d = doc_embeddings.shape[1]  # 向量維度
nlist = 500  # 聚類中心數量，可依資料量調整，資料越多nlist越大

quantizer = faiss.IndexFlatL2(d)  # 用作聚類中心的精確索引
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# 訓練索引 (必須先呼叫train，並且資料筆數需大於nlist)
assert doc_embeddings.shape[0] > nlist, "資料筆數必須大於聚類數 nlist"

print("⚙️ 正在訓練索引...")
index.train(doc_embeddings)

print("➕ 正在加入向量...")
index.add(doc_embeddings)

# --- 儲存索引與對應來源 ---
faiss.write_index(index, INDEX_PATH)
with open(SOURCE_PATH, "wb") as f:
    pickle.dump({"docs": docs, "sources": sources}, f)

print("✅ 向量索引建立與儲存完成！")

