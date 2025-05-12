import google.generativeai as genai
import os
from datetime import datetime
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# --- 參數區 ---
USE_FAISS = True  # ✅ 開關：是否使用向量知識庫輔助回答

# --- 初始化 Gemini ---
api_key_path = "api-key.txt"
if not os.path.exists(api_key_path):
    raise FileNotFoundError(f"找不到 API 金鑰檔案：{api_key_path}")
with open(api_key_path, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# --- 讀取向量庫 ---
if USE_FAISS:
    index = faiss.read_index("faiss_index.index")
    with open("doc_sources.pkl", "rb") as f:
        data = pickle.load(f)
    docs = data["docs"]
    sources = data["sources"]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- 檔案儲存設定 ---
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
history_filename = f"chat_history_{start_time}.txt"

def chat_with_gemini(user_input):
    if USE_FAISS:
        query_vector = embedder.encode([user_input])
        top_k = 3
        D, I = index.search(query_vector, top_k)

        # 判斷是否找到足夠相關的資料（以距離 D 判斷）
        threshold = 0.75  # 距離閾值越小表示越相似
        found = any(d < threshold for d in D[0])

        if found:
            retrieved_chunks = [docs[i] for i, d in zip(I[0], D[0]) if d < threshold]
            context = "\n".join(retrieved_chunks)
            prompt = f"你是一個聰明的 AI 助理，請根據以下資料回答問題：\n\n{context}\n\n問題：{user_input}"
        else:
            # 找不到相似資料，使用 fallback prompt
            prompt = f"""找不到相關資料。請依你自己的知識回答以下問題：
問題：{user_input}"""
    else:
        prompt = user_input

    response = chat.send_message(prompt)
    ai_reply = response.text

    # 儲存對話
    with open(history_filename, "a", encoding="utf-8") as f:
        f.write(f"你：{user_input}\n\n")
        f.write(f"Gemini：{ai_reply}\n\n")

    return ai_reply

if __name__ == "__main__":
    print("🤖 Gemini Chat CLI 已啟動（輸入 'exit' 離開）")
    if USE_FAISS:
        print("📚 已啟用知識庫查詢（FAISS + RAG）模式\n")
    else:
        print("💬 使用純 LLM 模式（未啟用知識庫）\n")

    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print(f"📄 對話已儲存為：{history_filename}")
            print("👋 再見！")
            break
        reply = chat_with_gemini(user_input)
        print("Gemini：", reply)

