#### 如何使用
使用 `git clone 本專案連結` 複製專案  
```bash
cd llm-practice
```

使用 pip/uv 安裝依賴庫  
以下方法二選一
```bash
1. pip install <依賴庫>  

2. uv sync
```

申請[Gemini](https://ai.google.dev/) 的api
```bash
touch api-key.txt
```
把金要貼到api-key.txt上
#### 依賴庫
- faiss-cpu
- google-generativeai
- pandas
- pymupdf
- python-docx
- sentence-transformers
