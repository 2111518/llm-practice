### 如何使用
1. 使用 `git clone 本專案連結` 複製專案  
切換到專案的檔案夾  
```bash
cd ~/llm-practice
```

2. 使用 pip/uv 安裝依賴庫  
以下方法二選一
```bash
1. pip install <依賴庫> 或 pip3 install <依賴庫>  

2. uv sync
```

3. 申請 [Gemini](https://ai.google.dev/) 的金鑰
```bash
touch api-key.txt
```
把金鑰貼到 `api-key.txt` 上  

4. (1) 使用RAG
創建資料庫  
```bash
mkdir knowledge_files
```
將需要使用的資料放入 `knowledge_files`  
先執行 `build-faiss.py`  
將 `llm.py` 中 USE_FAISS 改成 **True**
接著執行`llm.py`

4. (2) 不使用RAG  
直接執行`llm.py`

5. (1) 使用圖片辨識  
把圖片放到專案資料夾下，將 `llm.py` 中的變數 USE_IMAGE 改成 **True** ，  
輸入要辨識圖片，格式為 `img: ./example.jpg 您的問題`  
接著執行`llm.py`

5. (2) 不使用圖片辨識  
直接執行 `llm.py`

### 依賴庫
- faiss-cpu
- google-generativeai
- pandas
- pymupdf
- python-docx
- sentence-transformers
- pillow
