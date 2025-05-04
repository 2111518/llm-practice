import google.generativeai as genai
import os
from datetime import datetime

api_key_path = "api-key.txt"
if not os.path.exists(api_key_path):
    raise FileNotFoundError(f"找不到 API 金鑰檔案：{api_key_path}")
with open(api_key_path, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
history_filename = f"chat_history_{start_time}.txt"

def chat_with_gemini(user_input):
    response = chat.send_message(user_input)
    ai_reply = response.text

    # 儲存對話到唯一檔案
    with open(history_filename, "a", encoding="utf-8") as f:
        f.write(f"你：{user_input}\n\n")
        f.write(f"Gemini：{ai_reply}\n\n")

    return ai_reply

if __name__ == "__main__":
    print("🤖 Gemini Chat CLI 已啟動（輸入 'exit' 離開）\n")

    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print(f"📄 對話已儲存為：{history_filename}")
            print("👋 再見！")
            break
        reply = chat_with_gemini(user_input)
        print("Gemini：", reply)

