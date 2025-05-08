import zipfile
import os

def compress_chat_history_txt_files(output_zip):
    """
    搜尋所有包含 'chat_history' 的 .txt 檔並壓縮成一個 zip 檔。
    
    :param output_zip: 輸出的 zip 檔案路徑
    """
    # 找出所有包含 'chat_history' 的 txt 檔案
    txt_files = [
        f for f in os.listdir('.') 
        if f.endswith('.txt') and 'chat_history' in f
    ]

    if not txt_files:
        print("找不到符合條件的 .txt 檔案。")
        return

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in txt_files:
            zipf.write(file, arcname=file)
            # print(f"已壓縮：{file}")
    print("已完成壓縮")
# 使用範例
output_zip_name = 'chat_history_archive.zip'
compress_chat_history_txt_files(output_zip_name)

