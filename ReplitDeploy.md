# Replit 部署指南

Replit 是一個優秀的線上 IDE，可以輕鬆部署 Streamlit 應用。以下是在 Replit 上部署此專案的步驟。

## 步驟

1.  **註冊/登入 Replit:**
    前往 [replit.com](https://replit.com) 並建立一個帳戶。

2.  **建立新的 Repl:**
    - 點擊 `+ Create Repl` 按鈕。
    - 在 "Template" 搜尋框中，選擇 `Python`。
    - 為您的 Repl 命名，然後點擊 `Create Repl`。

3.  **上傳專案文件:**
    - 在左側的 "Files" 面板中，點擊三點選單 (`⋮`)，然後選擇 `Upload file` 或 `Upload folder`。
    - 將本地的所有專案文件 (`app.py`, `requirements.txt`, `.md` 文件等) 全部上傳。

4.  **執行與安裝:**
    - Replit 通常會自動偵測 `requirements.txt` 並安裝所有套件。您可以在 `Shell` 分頁中觀察進度。
    - 如果沒有自動安裝，可以在 `Shell` 中手動執行 `pip install -r requirements.txt`。

5.  **設定執行命令:**
    - 在 Replit 的 `.replit` 配置文件中，需要指定啟動命令。
    - 找到並打開 `.replit` 文件 (如果不存在，請創建它)。
    - 將 `run` 的值修改為 `"streamlit run app.py"`。
    - 完整的 `.replit` 文件內容可能如下：
      ```toml
      # .replit
      language = "python3"
      run = "streamlit run app.py"
      
      [packager]
      [packager.features]
      guessImports = true
      ```

6.  **啟動應用:**
    - 點擊頂部綠色的 `▶ Run` 按鈕。
    - Replit 會在右側的 "Webview" 視窗中啟動 Streamlit 伺服器並顯示您的應用程式。

現在您的應用程式已經成功部署在 Replit 上，您可以分享其公開 URL 給任何人存取。
