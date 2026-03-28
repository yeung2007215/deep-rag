# deep-rag 設計文件

> 桌遊規則問答系統：從實驗腳本到模組化 RAG 系統的演進紀錄

---

## 一、原始問題（舊版腳本的缺陷）

### 1. 架構層面
| 問題 | 說明 |
|------|------|
| **腳本式設計** | 所有邏輯混在單一 `.py` 檔，無法重用、難以維護 |
| **硬編碼問題** | `user_question = "星杯傳說, 起始手牌幾張"` 寫死在程式碼中，無法互動 |
| **參數散落各處** | `k=3`、`k=8`、`k=20`、`max_rounds=2/20` 散落在不同檔案，沒有統一管理 |
| **無 requirements.txt** | 依賴套件版本不明，換環境就壞 |

### 2. 資料索引層面
| 問題 | 說明 |
|------|------|
| **只索引一個文件** | `ChromaDB.py` 寫死 `docs/星杯傳說_規則說明書.md`，三國殺規則書完全未被索引 |
| **重複插入** | 每次執行 `ChromaDB.py` 都會重複 add 相同內容，資料庫會累積重複 chunks |
| **無 metadata** | Chunk 沒有 `game_name`、`source_file` 等標記，無法判斷來源 |

### 3. 檢索層面
| 問題 | 說明 |
|------|------|
| **純向量搜尋** | 只用 Similarity Search，對「手牌」「幾張」這類關鍵字匹配較弱 |
| **充足性判斷太粗糙** | `_is_context_sufficient()` 只檢查字數 ≥ 120，語意完全不考慮 |
| **Rerank 無 metadata 感知** | 舊版 Rerank 只看文字，不知道這段來自哪個遊戲，容易跨遊戲混淆 |
| **無信心門檻** | Rerank 結果不管相關性多低都納入，低品質段落影響回答 |

### 4. 相容性問題
| 問題 | 說明 |
|------|------|
| **`CrossEncoder` 未 import** | `DeepRAG-rewrite.py` 使用了 `CrossEncoder` 但沒有 import，直接 NameError |
| **無異常處理** | API 呼叫（Ollama / DeepSeek）失敗時整個程式 crash，沒有 fallback |

---

## 二、改善方法

### 1. 架構重構：職責分離（Separation of Concerns）

```
舊版：所有邏輯 → 一個腳本
新版：
  config.py     → 配置管理
  ingestion.py  → 文件索引
  retriever.py  → 檢索邏輯
  main.py       → 入口 + UI
```

每個模組只做一件事，可以獨立測試、獨立替換。

### 2. 統一配置（config.py）

所有參數集中在一個地方，並支援 `.env` 覆蓋：

```python
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "15"))
DEEPRAG_MAX_ROUNDS  = int(os.getenv("DEEPRAG_MAX_ROUNDS", "3"))
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.3"))
```

不用改程式碼，改 `.env` 就可以調整行為。

### 3. 多文件支援 + SHA-256 去重（ingestion.py）

```python
# 自動掃描所有 .md 文件
files = glob.glob("docs/**/*.md", recursive=True)

# 每個 chunk 計算 hash，只插入新的
content_hash = hashlib.sha256(text.encode()).hexdigest()
if content_hash not in existing_hashes:
    vector_store.add_documents([chunk])
```

重複執行不會重複插入，資料庫保持乾淨。

### 4. 混合檢索 Hybrid Search（retriever.py）

```
BM25（關鍵字匹配）  ┐
                     ├──→ 去重合併 → 最終候選集
Vector（語意相似）  ┘
```

- **向量搜尋**：擅長語意理解，例如「起手牌」對應「初始手牌」
- **BM25**：擅長精確關鍵字，例如「殺」「閃」「桃」這種專有名詞
- **互補**：兩者各取所長，合併去重後品質更高

### 5. LLM 充足性判斷（is_context_sufficient）

```
舊版：len(context) >= 120  ← 只看字數

新版：讓 LLM 分析
  → 找出問題中的關鍵實體（遊戲名、規則項目、數值）
  → 檢查檢索內容是否涵蓋這些實體
  → 輸出 SUFFICIENT / INSUFFICIENT:缺少XXX
  → 失敗時降級回字數判斷（容錯）
```

### 6. Metadata 感知 + 信心門檻 Reranker（llm_rerank）

```python
# 傳給 LLM 的候選段落包含遊戲名稱
"[3] (遊戲:星杯傳說) 遊戲開始時，每位玩家抽取5張手牌..."
"[7] (遊戲:三國殺)   每位玩家的初始手牌數量..."

# LLM 回傳編號 + 信心分數
3,95
7,40   ← 40 < 門檻 30% → 排除

# 只保留高信心段落
```

防止不同遊戲的規則混入回答。

### 7. 交互式問答循環（main.py）

```python
while True:
    question = input("❓ 請輸入問題: ").strip()
    if question.lower() in ("quit", "exit"):
        break
    answer = ask(question, vector_store, bm25_docs)
    print(answer)
```

支援連續提問，不用每次重新啟動程式。

---

## 三、整體設計邏輯

### 3.1 交互流程（兩層式選單）

```
程式啟動 (main.py)
       │
       ▼
┌──────────────────────────────┐
│  第一層：遊戲選擇選單         │
│                              │
│  1. 星杯傳說                  │
│  2. 三國殺                    │
│  輸入 quit 退出               │
└────────────┬─────────────────┘
             │ 使用者選擇編號
             ▼
┌──────────────────────────────┐
│  連接對應 Collection          │
│  → get_vector_store(col)     │
│  → 建立 BM25 索引            │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  第二層：問題諮詢循環         │
│                              │
│  ❓ [星杯傳說] 請輸入問題:    │
│                              │
│  → 輸入問題 → DeepRAG 檢索   │
│  → 輸入 'back' → 返回第一層  │
│  → 輸入 'quit' → 結束程式    │
└──────────────────────────────┘
```

### 3.2 Multi-collection 架構

```
config.py 定義：

  GAME_COLLECTIONS = {
    "asteriated_grail": {
        collection: "asteriated_grail_collection"
        file_keywords: ["星杯傳說"]
    },
    "war_of_three_kingdoms": {
        collection: "war_of_the_three_kingdom_collection"
        file_keywords: ["三國殺"]
    }
  }

ingestion 時：
  星杯傳說_規則說明書.md  →  asteriated_grail_collection
  三國殺_規則說明書.md    →  war_of_the_three_kingdom_collection

檢索時：
  使用者選遊戲 → 只連接對應 collection → 不會跨遊戲混淆
```

### 3.3 DeepRAG 檢索流程

```
使用者輸入問題
       │
       ▼
┌─────────────────────────────┐
│  第一輪：混合檢索              │
│  BM25 + Vector → 去重合併    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  LLM 充足性判斷              │
│  分析關鍵實體是否被涵蓋       │
└────────────┬────────────────┘
             │
      充足？ No
             │
             ▼
┌─────────────────────────────┐
│  LLM 生成 Follow-up Queries  │
│  針對缺失資訊生成新查詢       │
│  （貼近規則書用語、同義詞）   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  第 N 輪：再次混合檢索        │
│  去重、合併至 context 池      │
└────────────┬────────────────┘
             │
    最多 3 輪後停止
             │
             ▼
┌─────────────────────────────┐
│  收集所有 queries 的候選文件  │
│  LLM Reranker 重排           │
│  - 帶入 game_name metadata   │
│  - 信心分數 < 30% 排除       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  最終 Context → LLM 回答    │
│  要求引用規則原文            │
│  找不到則說明缺少什麼        │
└─────────────────────────────┘
```

---

## 四、新舊版本對比

| 項目 | 舊版 | 新版 |
|------|------|------|
| **架構** | 單一腳本 | 4 個職責分離模組 |
| **配置管理** | 硬編碼 | config.py + .env 支援 |
| **支援遊戲** | 只有星杯傳說 | 星杯傳說 + 三國殺（自動掃描）|
| **向量集合** | 單一 collection | 每個遊戲獨立 collection |
| **重複索引** | 每次重複插入 | SHA-256 去重 |
| **檢索方式** | 純向量搜尋 | BM25 + Vector 混合 |
| **充足性判斷** | 只看字數 | LLM 分析關鍵實體 |
| **Reranker** | 無 metadata / 無門檻 | 帶 game_name + 信心門檻 |
| **異常處理** | 無，直接 crash | 每個 API 呼叫有 try/except + fallback |
| **使用方式** | 改程式碼重跑 | 兩層式交互選單 / CLI 參數 |
| **導航** | 無 | back 返回上層 / quit 退出 |

---

## 五、使用方式

```bash
# 安裝依賴
pip install -r requirements.txt

# 首次索引（自動歸類到各遊戲 collection）
python main.py --ingest

# 啟動交互式問答（兩層式選單）
python main.py

# 單次查詢（指定遊戲）
python main.py --query "起始手牌幾張" --game asteriated_grail
python main.py --query "殺的使用規則" --game war_of_three_kingdoms

# 強制重新索引
python main.py --ingest --force
```

---

## 六、檔案一覽

| 檔案 | 角色 | 狀態 |
|------|------|------|
| `config.py` | 配置中心（含多遊戲集合定義）| ✅ 使用中 |
| `ingestion.py` | 文件索引（按遊戲歸類 collection）| ✅ 使用中 |
| `retriever.py` | 檢索核心（支援動態 collection）| ✅ 使用中 |
| `main.py` | 程式入口（兩層式選單）| ✅ 使用中 |
| `requirements.txt` | 依賴清單 | ✅ 使用中 |
| `ChromaDB.py` | 舊版索引腳本 | 🗑️ 可刪除 |
| `Standard-RAG.py` | 舊版 RAG | 🗑️ 可刪除 |
| `BM25-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `Similarity-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `DeepRAG.py` | 舊版最終腳本 | 🗑️ 可刪除 |
| `DeepRAG-rewrite.py` | 舊版中間版 | 🗑️ 可刪除 |
| `DeepRAG-backup.py` | 舊版早期版 | 🗑️ 可刪除 |

