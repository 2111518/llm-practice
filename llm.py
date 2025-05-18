import os
import pickle
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- 設定區 ---
USE_FAISS = True
API_KEY_FILE = "api-key.txt"
INDEX_FILE = "faiss_index.index"
SOURCE_FILE = "doc_sources.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 10
L2_THRESHOLD = 0.75

# --- 初始化 Gemini ---
if not os.path.exists(API_KEY_FILE):
    raise FileNotFoundError(f"找不到 API 金鑰檔案：{API_KEY_FILE}")

with open(API_KEY_FILE, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# --- 初始化 FAISS 與嵌入模型 ---
if USE_FAISS:
    index = faiss.read_index(INDEX_FILE)
    with open(SOURCE_FILE, "rb") as f:
        data = pickle.load(f)
    docs = data["docs"]
    sources = data["sources"]
    embedder = SentenceTransformer(EMBEDDING_MODEL)

# --- 儲存歷史檔案 ---
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
history_filename = f"chat_history_{start_time}.txt"

# --- 對話處理函式 ---
def chat_with_gemini(user_input):
    if USE_FAISS:
        query_vector = embedder.encode([user_input], convert_to_numpy=True)
        query_vector = query_vector.reshape(1, -1)

        assert query_vector.shape[1] == index.d, f"❌ 維度錯誤：查詢向量為 {query_vector.shape[1]}，索引為 {index.d}"

        D, I = index.search(query_vector, TOP_K)
        valid_results = [(docs[i], sources[i], d) for i, d in zip(I[0], D[0]) if d < L2_THRESHOLD]

        if valid_results:
            context = "\n".join(f"[{src}] {chunk}" for chunk, src, _ in valid_results)
            prompt = f"你是一個聰明的 AI 助理，請參考以下資料和你的知識回答問題：\n\n{context}\n\n問題：{user_input}"
        else:
            prompt = f"找不到相關資料。請依你自己的知識回答以下問題：\n問題：{user_input}"
    else:
        prompt = user_input

    response = chat.send_message(prompt)
    ai_reply = response.text

    with open(history_filename, "a", encoding="utf-8") as f:
        f.write(f"你：{user_input}\n\nGemini：{ai_reply}\n\n")

    return ai_reply

# --- 主互動介面 ---
if __name__ == "__main__":
    print("🤖 Gemini Chat CLI 已啟動（輸入 'exit' 或 'quit' 離開）")
    print("📚 已啟用知識庫查詢模式\n" if USE_FAISS else "💬 使用純 LLM 模式（未啟用知識庫）\n")

    while True:
        user_input = input("你：")
        if user_input.strip().lower() in {"exit", "quit"}:
            print(f"📄 對話已儲存為：{history_filename}")
            print("👋 再見！")
            break
        reply = chat_with_gemini(user_input)
        print("Gemini：", reply)
