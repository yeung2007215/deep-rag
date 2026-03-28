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
| **BM25 每次重建** | 每次 `hybrid_search` 都重新建立 BM25 索引，嚴重浪費效能 |
| **步驟 3 冗餘搜尋** | DeepRAG 迭代完後又用裸 `similarity_search` 重搜一遍，忽略 BM25 命中 |
| **無對話記憶** | 每次問答獨立（Stateless），無法處理「那三國殺呢？」這類指代問題 |
| **無粵語支援** | 廣東話口語（「點樣先贏？」）直接送入檢索，命中率極低 |

### 4. 相容性問題
| 問題 | 說明 |
|------|------|
| **`CrossEncoder` 未 import** | `DeepRAG-rewrite.py` 使用了 `CrossEncoder` 但沒有 import，直接 NameError |
| **無異常處理** | API 呼叫（Ollama / DeepSeek）失敗時整個程式 crash，沒有 fallback |
| **無 Token 防護** | 對話歷史和 Context 無截斷保護，可能超出 LLM context window |

---

## 二、改善方法

### 1. 架構重構：職責分離（Separation of Concerns）

```
舊版：所有邏輯 → 一個腳本
新版：
  config.py     → 配置管理（含術語對照表）
  ingestion.py  → 文件索引（多集合歸類）
  retriever.py  → 檢索邏輯（對話記憶 + 粵語處理）
  main.py       → 入口 + UI（兩層式選單）
```

### 2. 統一配置（config.py）

所有參數集中管理，支援 `.env` 覆蓋：

```python
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "15"))
DEEPRAG_MAX_ROUNDS  = int(os.getenv("DEEPRAG_MAX_ROUNDS", "3"))
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.3"))
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
```

### 3. 多文件支援 + SHA-256 去重 + 多集合歸類（ingestion.py）

```python
# 自動掃描 → 按檔名關鍵字歸類到對應 collection
星杯傳說_規則說明書.md  →  asteriated_grail_collection
三國殺_規則說明書.md    →  war_of_the_three_kingdom_collection

# SHA-256 去重：重複執行不重複插入
content_hash = hashlib.sha256(text.encode()).hexdigest()
```

### 4. 混合檢索 + BM25 預建索引（retriever.py）

```
BM25（關鍵字匹配，只建一次）  ┐
                               ├──→ 去重合併 → 候選集
Vector（語意相似）            ┘

舊版：每次 hybrid_search 都重建 BM25 索引（12+ 次/問題）
新版：build_bm25_retriever() 只建一次，整個 session 複用
```

### 5. 對話記憶 + 指代消解（retriever.py）

```
歷史: [("星杯傳說起始手牌？", "遊戲開始每人抽 5 張")]
問題: "那如果是三國殺呢？"

rewrite_query_with_history()
  → LLM 分析指代詞
  → 改寫: "三國殺的起始手牌是幾張？"

保護機制：
  - 滑動窗口：最多保留 CHAT_HISTORY_MAX_TURNS 輪
  - 回答截斷：每輪回答最多 CHAT_HISTORY_ANSWER_MAX_CHARS 字
  - 切換遊戲時自動清除歷史
```

### 6. 廣東話雙層處理（retriever.py + config.py）

```
層級 1：LLM 口語改寫（語意層）
  「點樣先贏？」→ LLM → 「勝利條件是什麼？」

層級 2：術語對照表 BM25 展開（關鍵字層）
  「攞幾多張手牌」→ CANTONESE_TERM_MAP 展開 →
    ["拿幾多張手牌", "取得幾多張手牌", "攞多少張手牌"]
  → 全部送入 BM25 搜尋

兩層互補：即使 LLM 改寫失敗，術語展開仍然有效
```

### 7. Rerank 四級評分標準（retriever.py）

```
舊版 prompt：「選出最相關的段落，給分」
新版 prompt：
  90-100：段落直接回答了問題核心
  60-89：包含相關規則但需推理
  30-59：僅間接相關
  0-29：無關（不輸出）
  + 跨遊戲降分指令
```

### 8. Token 溢出防護（三層保護）

```
① 歷史截斷: CHAT_HISTORY_ANSWER_MAX_CHARS = 300（每輪回答）
② 歷史窗口: CHAT_HISTORY_MAX_TURNS = 5（最多輪數）
③ Context 截斷: MAX_CONTEXT_CHARS = 8000（傳入 LLM 上限）

最壞估算：5×(50+300) + 8000 = 9750 字 ≈ 19500 tokens
DeepSeek context window = 64K tokens → 安全 ✅
```

---

## 三、整體設計邏輯

### 3.1 交互流程（兩層式選單 + 對話記憶）

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
│  → build_bm25_retriever()    │
│  → chat_history = []         │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  第二層：問題諮詢循環         │
│                              │
│  ❓ [星杯傳說] 請輸入問題:    │
│                              │
│  → 輸入問題 → DeepRAG 檢索   │
│       (帶 chat_history)      │
│  → 回答後追加到 history      │
│  → 輸入 'history' 查看記憶   │
│  → 輸入 'back' → 清除記憶    │
│       → 返回第一層           │
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

### 3.3 DeepRAG 檢索流程（完整版）

```
使用者輸入問題（可能含粵語/指代）
       │
       ▼
┌─────────────────────────────────┐
│  步驟 0：Query 改寫             │
│  • 指代消解（用 chat_history）   │
│  • 廣東話 → 書面語              │
│  「那三國殺呢？」               │
│    → 「三國殺的起始手牌幾張？」  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  步驟 1：初始混合檢索           │
│  BM25 + Vector → 去重合併       │
│  + 粵語術語展開搜尋             │
│  → 候選文件即時收集             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  步驟 2：多輪迭代               │
│  ┌─ LLM 充足性判斷             │
│  │  分析關鍵實體是否被涵蓋      │
│  │                              │
│  │  充足？→ 跳到步驟 3          │
│  │  不足？↓                     │
│  │                              │
│  ├─ LLM 生成 Follow-up Queries │
│  │  （含 chat_history 參考）    │
│  │                              │
│  └─ 再次混合檢索 → 收集候選    │
│     最多 3 輪                   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  步驟 3：LLM Reranker          │
│  直接複用已收集的候選文件       │
│  （不再重複搜尋）               │
│  - 四級評分標準                 │
│  - metadata 遊戲名稱感知        │
│  - 信心 < 30% 排除             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  步驟 4：LLM 回答               │
│  Context 截斷至 8000 字         │
│  要求引用規則原文               │
│  找不到則說明缺少什麼           │
└─────────────────────────────────┘
```

### 3.4 API 呼叫量分析

```
單次問答的 API 呼叫數（最壞情況 = 3 輪迭代）：

DeepSeek LLM 呼叫：
  rewrite_query_with_history  × 1
  is_context_sufficient       × 3（每輪 1 次）
  generate_followup_queries   × 3（每輪 1 次）
  llm_rerank                  × 1
  build_answer_agent          × 1
  ─────────────────────────────
  合計最多 9 次

Ollama Embedding 呼叫（hybrid_search）：
  初始搜尋  × 1
  粵語展開  × 0~3（視術語命中數）
  迭代搜尋  × 3~9（3 輪 × 1~3 followup）
  ─────────────────────────────
  合計最多 13 次

BM25：
  build_bm25_retriever × 1（整個 session 只建一次）✅
```

---

## 四、新舊版本對比

| 項目 | 舊版 | 新版 |
|------|------|------|
| **架構** | 單一腳本 | 4 個職責分離模組 |
| **配置管理** | 硬編碼 | config.py + .env（30+ 參數）|
| **支援遊戲** | 只有星杯傳說 | 星杯傳說 + 三國殺（自動掃描）|
| **向量集合** | 單一 collection | 每個遊戲獨立 collection |
| **重複索引** | 每次重複插入 | SHA-256 去重 |
| **檢索方式** | 純向量搜尋 | BM25 + Vector 混合（BM25 預建索引）|
| **BM25 效能** | 每次搜尋重建索引 | 只建一次，session 複用 |
| **Rerank 候選** | 步驟 3 重複搜尋 | 迭代期間即時收集，直接 rerank |
| **Rerank 評分** | 簡單提示 | 四級評分標準 + 跨遊戲降分 |
| **充足性判斷** | 只看字數 | LLM 分析關鍵實體 |
| **對話記憶** | 無（Stateless）| 滑動窗口歷史 + 指代消解 |
| **粵語支援** | 無 | LLM 口語改寫 + 36 組術語對照表 BM25 展開 |
| **Token 防護** | 無 | 三層截斷保護（歷史 / 窗口 / Context）|
| **異常處理** | 無，直接 crash | 每個 API 呼叫有 try/except + fallback |
| **使用方式** | 改程式碼重跑 | 兩層式交互選單 / CLI / history 指令 |

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

# 交互範例
❓ [星杯傳說] 請輸入問題: 起始手牌幾張
🤖 遊戲開始時每人抽 5 張手牌...

❓ [星杯傳說] 請輸入問題: 那三國殺呢？
🔄 指代消解: 「那三國殺呢？」→「三國殺的起始手牌是幾張？」
🤖 三國殺中每位玩家起始手牌為 4 張...

❓ [星杯傳說] 請輸入問題: 點樣先贏？
🔄 粵語改寫: 「點樣先贏？」→「勝利條件是什麼？」
🤖 ...

❓ [星杯傳說] 請輸入問題: history
💬 對話記憶（共 3 輪）：...

❓ [星杯傳說] 請輸入問題: back
↩️  返回遊戲選擇... (已清除 3 輪對話記憶)
```

---

## 六、Embedding 模型升級建議

| 模型 | 中文效果 | 資源需求 | 適用場景 |
|------|---------|---------|---------|
| `nomic-embed-text` | ⭐⭐ | 低 | 開發測試 |
| `bge-m3` | ⭐⭐⭐⭐ | 中 | **推薦：中英混合最佳** |
| `bge-large-zh` | ⭐⭐⭐⭐⭐ | 高 | 純中文語義最強 |

切換方式：修改 `.env` 中的 `EMBEDDING_MODEL` 後重新 `--ingest --force`。

---

## 七、檔案一覽

| 檔案 | 角色 | 狀態 |
|------|------|------|
| `config.py` | 配置中心（多集合 + 術語對照表 + Token 防護參數）| ✅ 使用中 |
| `ingestion.py` | 文件索引（按遊戲歸類 collection + SHA-256 去重）| ✅ 使用中 |
| `retriever.py` | 檢索核心（對話記憶 + 粵語處理 + 預建 BM25 + 四級 Rerank）| ✅ 使用中 |
| `main.py` | 程式入口（兩層式選單 + 歷史管理 + Token 截斷）| ✅ 使用中 |
| `requirements.txt` | 依賴清單 | ✅ 使用中 |
| `DESIGN.md` | 本設計文件 | ✅ 使用中 |
| `ChromaDB.py` | 舊版索引腳本 | 🗑️ 可刪除 |
| `Standard-RAG.py` | 舊版 RAG | 🗑️ 可刪除 |
| `BM25-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `Similarity-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `DeepRAG.py` | 舊版最終腳本 | 🗑️ 可刪除 |
| `DeepRAG-rewrite.py` | 舊版中間版 | 🗑️ 可刪除 |
| `DeepRAG-backup.py` | 舊版早期版 | 🗑️ 可刪除 |

