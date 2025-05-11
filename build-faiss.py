import os
import json
import faiss
import pickle
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
sources = []

knowledge_dir = "knowledge_files"

# --- 各類型讀取函式 ---

def read_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text.split("\n")

def read_docx(path):
    doc = Document(path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def read_csv(path):
    df = pd.read_csv(path)
    return df.astype(str).values.flatten().tolist()

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def read_json(path):
    def extract_text(obj):
        if isinstance(obj, dict):
            return sum([extract_text(v) for v in obj.values()], [])
        elif isinstance(obj, list):
            return sum([extract_text(i) for i in obj], [])
        elif isinstance(obj, str):
            return [obj]
        else:
            return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return extract_text(data)

def read_md(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 可略過 markdown 標記語法（如標題、連結等）或保留
    return [line.strip() for line in lines if line.strip()]

def chunk_text(text, chunk_size=200, overlap=50):
    """將長段文字切成重疊的小段落"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# --- 主處理迴圈 ---

for filename in os.listdir(knowledge_dir):
    path = os.path.join(knowledge_dir, filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        lines = read_pdf(path)
    elif ext == ".docx":
        lines = read_docx(path)
    elif ext == ".csv":
        lines = read_csv(path)
    elif ext == ".txt":
        lines = read_txt(path)
    elif ext == ".json":
        lines = read_json(path)
    elif ext == ".md":
        lines = read_md(path)
    else:
        print(f"⚠️ 不支援的檔案格式：{filename}")
        continue
    
    for line in lines:
        if line.strip():
            # 這裡進行 chunking（每 200 字一段，重疊 50 字）
            for chunk in chunk_text(line.strip(), chunk_size=200, overlap=50):
                if chunk:
                    docs.append(chunk)
                    sources.append(filename)

# --- 建立向量索引 ---

doc_embeddings = embedder.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# 儲存
faiss.write_index(index, "faiss_index.index")
with open("doc_sources.pkl", "wb") as f:
    pickle.dump({"docs": docs, "sources": sources}, f)

print("✅ 多格式知識庫已完成！")

