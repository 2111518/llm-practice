import os
import pickle
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
import io

# --- 參數設定區 ---
USE_FAISS = False  # 是否啟用知識庫搜尋(RAG)
USE_IMAGE = False  # 是否啟用圖片理解
API_KEY_FILE = "api-key.txt"
INDEX_FILE = "faiss_index.index"
SOURCE_FILE = "doc_sources.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 50
NPROBE = 50  # 用於 IVFFlat 的查詢參數

# --- 初始化 API Key 與 Gemini 模型 ---
if not os.path.exists(API_KEY_FILE):
    raise FileNotFoundError(f"找不到 API 金鑰檔案: {API_KEY_FILE}")

with open(API_KEY_FILE, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

# --- 啟用圖片辨識模型 ---
if USE_IMAGE:
    multimodal_model = genai.GenerativeModel("gemini-1.5-flash")
    image_chat = multimodal_model.start_chat()

def read_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def chat_with_image(image_path, user_prompt):
    image_bytes = read_image_bytes(image_path)
    image_pil = Image.open(io.BytesIO(image_bytes))
    response = image_chat.send_message([user_prompt, image_pil])
    return response.text

# --- FAISS 初始化 ---
if USE_FAISS:
    index = faiss.read_index(INDEX_FILE)
    index.nprobe = NPROBE  # V 設定查詢範圍以提升命中率
    with open(SOURCE_FILE, "rb") as f:
        data = pickle.load(f)
    docs = data["docs"]
    sources = data["sources"]
    embedder = SentenceTransformer(EMBEDDING_MODEL)

# --- 對話歷史儲存 ---
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
history_filename = f"chat_history_{start_time}.txt"

# --- Gemini 主對話邏輯 ---
def chat_with_gemini(user_input):
    if USE_FAISS:
        query_vector = embedder.encode([user_input], convert_to_numpy=True)
        query_vector = query_vector.reshape(1, -1)

        assert query_vector.shape[1] == index.d, f"維度錯誤: 查詢向量為 {query_vector.shape[1]}，索引為 {index.d}"

        D, L = index.search(query_vector, TOP_K)

        # V 保留所有有效結果，不使用距離閾值篩選
        valid_results = [(docs[i], sources[i], d) for i, d in zip(L[0], D[0]) if i != -1]

        if valid_results:
            # match_count = len(valid_results)
            # print(f"找到 {match_count} 筆相似資料（Top {TOP_K}）")
            # context = "\n".join(f"[{src}] {chunk}（距離: {dist:.2f}）" for chunk, src, dist in valid_results)
            context = "\n".join(f"[{src}] {chunk}" for chunk, src, _ in valid_results)
            prompt = f"請參考以下資料和你的知識回答問題: \n{context}\n問題: {user_input}"
        else:
            print("! 找不到相似資料，改以 LLM 模型知識回答")
            prompt = f"{user_input}"
    else:
        prompt = user_input

    response = chat.send_message(prompt)
    return response.text

# --- 主互動介面 ---
if __name__ == "__main__":
    print("🤖 Gemini Chat CLI 已啟動(輸入 'exit' 或 'quit' 離開)")
    if USE_FAISS:
        print("📚 已啟用知識庫查詢模式")
    else:
        print("💬 使用純 LLM 模式")

    if USE_IMAGE:
        print("🖼️ 圖片理解功能已啟用(使用格式：img: ./example.jpg 您的問題)")
    else:
        print("未啟用圖片理解功能")

    while True:
        user_input = input("您：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print(f"📄 對話已儲存為: {history_filename}")
            # print("再見！")
            break

        if USE_IMAGE and user_input.startswith("img:"):
            try:
                parts = user_input[4:].strip().split(" ", 1)
                image_path = parts[0]
                prompt = parts[1] if len(parts) > 1 else "請說明這張圖的內容"
                reply = chat_with_image(image_path, prompt)
            except Exception as e:
                reply = f"圖片處理錯誤：{str(e)}"
        else:
            reply = chat_with_gemini(user_input)

        print("Gemini：", reply)

        try:
            with open(history_filename, "a", encoding="utf-8") as f:
                f.write(f"你：{user_input}\n\nGemini：{reply}\n\n")
        except Exception as e:
            print(f"無法儲存對話紀錄: {str(e)}")

