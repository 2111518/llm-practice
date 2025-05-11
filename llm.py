import google.generativeai as genai
import os
import faiss
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 初始化 Gemini
with open("api-key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# 載入 FAISS 索引與檔案來源
index = faiss.read_index("faiss_index.index")
with open("doc_sources.pkl", "rb") as f:
    doc_data = pickle.load(f)
docs = doc_data["docs"]
sources = doc_data["sources"]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
history_filename = f"chat_history_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

def search_knowledge(query, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    results = [docs[i] for i in I[0]]
    return "\n".join(results)

def chat_with_gemini(user_input):
    retrieved = search_knowledge(user_input)
    prompt = (
        f"以下是與使用者問題相關的資料片段：\n{retrieved}\n\n"
        f"根據這些資料，請回答使用者的問題：\n{user_input}"
    )
    response = chat.send_message(prompt)
    reply = response.text

    with open(history_filename, "a", encoding="utf-8") as f:
        f.write(f"你：{user_input}\n\n")
        f.write(f"Gemini：{reply}\n\n")
    return reply

if __name__ == "__main__":
    print("🤖 Gemini RAG Chat with FAISS 已啟動（輸入 'exit' 離開）\n")

    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print(f"📄 對話已儲存為：{history_filename}")
            print("👋 再見！")
            break
        reply = chat_with_gemini(user_input)
        print("Gemini：", reply)

