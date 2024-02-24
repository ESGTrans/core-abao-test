# core-abao-test


### 安裝套件管理工具 Poetry

- 安裝 poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

- 將以下加入 .bashrc 重啟 terminal 以使環境變數設定生效
    
```bash
export PATH=$PATH:$HOME/.local/bin
```

### 重現環境

```bash
poetry install
```

### 使用說明

(暫時)

運行 app.py 可從 Qdrant DB 中取得碳盤資料
