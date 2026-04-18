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
| **BM25 中文分詞壞掉** | BM25 預設用 `.split()`（按空格），中文沒空格 → 整段變 1 個 token → **完全無法匹配** |
| **BM25 每次重建** | 每次 `hybrid_search` 都重新建立 BM25 索引，嚴重浪費效能 |
| **混合檢索合併粗糙** | 只做拼接去重（vector 在前），BM25 排第一的結果也會被排到後面 |
| **充足性判斷太粗糙** | `_is_context_sufficient()` 只檢查字數 ≥ 120，語意完全不考慮 |
| **Rerank 無 metadata 感知** | 舊版 Rerank 只看文字，不知道這段來自哪個遊戲，容易跨遊戲混淆 |
| **無信心門檻** | Rerank 結果不管相關性多低都納入，低品質段落影響回答 |
| **步驟 3 冗餘搜尋** | DeepRAG 迭代完後又用裸 `similarity_search` 重搜一遍，忽略 BM25 命中 |
| **一視同仁無路由** | 簡單問題「起始手牌幾張」也走完整 DeepRAG 3 輪迭代，每輪引入噪音反而稀釋正確答案 |
| **無對話記憶** | 每次問答獨立（Stateless），無法處理「那三國殺呢？」這類指代問題 |
| **無粵語支援** | 廣東話口語（「點樣先贏？」）直接送入檢索，命中率極低 |
| **玩家俗語斷裂** | 使用者說「反彈」但規則書寫「應戰」，BM25 和向量搜尋都無法匹配 |
| **HTML 註解浪費 chunk** | 規則書嵌入大量 `<!--section_id:...-->` 佔用 chunk 空間，稀釋語義 |

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
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.5"))
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
QUERY_CLASSIFY_ENABLED = true  # 問題複雜度路由開關
```

### 3. 多文件支援 + SHA-256 去重 + 多集合歸類（ingestion.py）

```python
# 自動掃描 → 按檔名關鍵字歸類到對應 collection
星杯傳說_規則說明書.md  →  asteriated_grail_collection
三國殺_規則說明書.md    →  war_of_the_three_kingdom_collection

# SHA-256 去重：重複執行不重複插入
content_hash = hashlib.sha256(text.encode()).hexdigest()
```

### 4. 混合檢索 + BM25 中文分詞修復（retriever.py）

```
舊版 BM25 分詞（完全失效）：
  "每位玩家依次從中摸取4張作為起始手牌"
  → .split() → ['每位玩家依次從中摸取4張作為起始手牌']  ← 1 個 token
  → 查詢 "起始手牌是多少" → 0 匹配 ❌

新版 BM25 分詞（逐字拆分）：
  → _chinese_tokenize() → ['每','位','玩','家','摸','取','4','張','起','始','手','牌']
  → 查詢 ['起','始','手','牌','是','多','少'] → 4 個字匹配 ✅

混合合併：舊版拼接去重 → 新版 RRF (Reciprocal Rank Fusion)
  score(d) = Σ 1/(k + rank_i(d))，雙源命中的文件分數疊加排更前

BM25 建索引：build_bm25_retriever() 只建一次，整個 session 複用
```

### 5. 語義標準化引擎（Semantic Normalization Engine）

```
rewrite_query_with_history() 從「指代消解工具」升級為「語義標準化引擎」
三層職責 + 雙軌並行架構：

═══ 軌道 1：LLM 語義理解（rewrite_query_with_history）═══

  「暗滅對暗滅，可唔可以反彈？」
    │
    ├─ A. 粵語口語 → 書面語:   「可唔可以」→「可不可以」
    ├─ B. 玩家俗語 → 規則術語: 「反彈」→「應戰」  ← 最重要！
    └─ C. 指代消解:             「那三國殺呢？」→「三國殺的起始手牌幾張？」
    │
    → 「暗滅對暗滅，可不可以應戰」（resolved_question，主查詢）

═══ 軌道 2：映射表精確匹配（expand_query_terms）═══

  同時查四張表：
  ① PLAYER_SLANG_MAP    (56 組) ← 「反彈→應戰」「扣血→降低士氣」
  ② CANTONESE_TERM_MAP  (56 組) ← 「唔→不」「嘅→的」
  ③ CANTONESE_FUZZY_MAP (13 組) ← 「錦朗→錦囊」拼音校正
  ④ 組合展開 — 俗語結果再套粵語，粵語結果再套俗語

  → 多條展開 queries 全部送入 BM25 搜尋

兩軌互為安全網：
  - LLM 改寫失敗 → 映射表仍能精確匹配
  - 映射表沒收錄 → LLM 能從語境推論

觸發條件（5 種）：
  ① 偵測到粵語 (_has_cantonese)        → 強制觸發
  ② 偵測到玩家俗語 (_has_player_slang) → 強制觸發
  ③ 有歷史 + 有指代詞
  ④ 有歷史 + 問題太短 (≤20 字)
  ⑤ 以上任一命中即觸發，無需等歷史存在

防語義漂移驗證：
  - 改寫結果 < 原問題 30% → 回退
  - 實體丟失 > 50% → 合併為「改寫（原問題）」
  - 已知俗語被改掉不視為「丟失」（正確行為）
```

### 6. 問題複雜度智慧路由（Query Complexity Routing）

```
classify_query_complexity() — LLM 判斷問題類型，動態選擇檢索策略

  FACTOID（事實查詢）
    「起始手牌幾張」「最多幾人」
    → _standard_retrieve(): hybrid_search → Rerank → 完成
    → 不迭代，不生成 follow-up，不引入噪音
    → LLM 呼叫 3 次

  PROCEDURAL（流程步驟）
    「一個回合怎麼進行」
    → _deep_retrieve(max_rounds=1): 輕量迭代
    → LLM 呼叫 5 次

  REASONING（推理分析）
    「如果主公死了忠臣會怎樣」
    → _deep_retrieve(max_rounds=3): 完整 DeepRAG
    → LLM 呼叫最多 10 次

  COMPARISON（跨規則比較）
    「星杯和三國殺哪個更複雜」
    → _deep_retrieve(max_rounds=3): 完整 DeepRAG
    → LLM 呼叫最多 10 次

  ⚙️ 可透過 QUERY_CLASSIFY_ENABLED=false 關閉分類（全部走 DeepRAG）
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
① 歷史截斷: CHAT_HISTORY_ANSWER_MAX_CHARS = 500（每輪回答）
② 歷史窗口: CHAT_HISTORY_MAX_TURNS = 5（最多輪數）
③ Context 截斷: MAX_CONTEXT_CHARS = 8000（傳入 LLM 上限）

最壞估算：5×(50+500) + 8000 = 10750 字 ≈ 21500 tokens
DeepSeek context window = 64K tokens → 安全 ✅
```

### 9. Chunk 預處理優化（ingestion.py）

```
問題：規則書嵌入大量 <!--section_id:...--> HTML 註解，佔用 ~200 字/chunk
修復：_strip_html_comments() 在切分前移除，每 chunk 節省 ~200 字

CHUNK_SIZE: 500 → 300（規則段落通常 100-300 字，避免跨段噪音）
CHUNK_OVERLAP: 100 → 80
CHUNK_SEPARATORS: 新增 "\n## "、"\n### "、"，" 讓切分更精確
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

### 3.3 檢索流程（智慧路由版）

```
使用者輸入問題（可能含粵語/俗語/指代）
       │
       ▼
┌──────────────────────────────────────────┐
│  步驟 0：語義標準化                       │
│  rewrite_query_with_history()            │
│                                          │
│  觸發: 粵語 / 俗語 / 指代 / 太短         │
│                                          │
│  「暗滅對暗滅，可唔可以反彈？」           │
│    ├─ 粵語:「可唔可以」→「可不可以」      │
│    ├─ 俗語:「反彈」→「應戰」             │
│    └→ 「暗滅對暗滅，可不可以應戰」        │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  步驟 1：問題複雜度分類                   │
│  classify_query_complexity()             │
│                                          │
│  FACTOID ─────→ Standard 路徑            │
│  PROCEDURAL ──→ DeepRAG 輕量（1 輪）     │
│  REASONING ───→ DeepRAG 完整（3 輪）     │
│  COMPARISON ──→ DeepRAG 完整（3 輪）     │
└──┬─────────────────┬────────────────────┘
   │ FACTOID          │ REASONING/COMPARISON/PROCEDURAL
   ▼                  ▼
┌──────────┐   ┌──────────────────────────┐
│ Standard │   │  DeepRAG                 │
│          │   │                          │
│ hybrid   │   │  hybrid_search           │
│ search   │   │    ↓                     │
│    ↓     │   │  is_context_sufficient?  │
│ Rerank   │   │    ↓ 不足                 │
│    ↓     │   │  follow-up queries       │
│ 完成     │   │    ↓                     │
│          │   │  再次 hybrid_search      │
│ 3 次     │   │  （最多 1~3 輪）           │
│ LLM 呼叫 │   │    ↓                     │
│          │   │  Rerank → 完成           │
│          │   │                          │
│          │   │  5~10 次 LLM 呼叫         │
└──┬───────┘   └────────┬─────────────────┘
   │                    │
   └──────┬─────────────┘
          │
          ▼ （兩條路徑都會執行）
┌──────────────────────────────────────────┐
│  術語展開搜尋 expand_query_terms()         │
│                                          │
│  同時查四張表:                             │
│  ① PLAYER_SLANG_MAP → 「反彈」→「應戰」    │
│  ② CANTONESE_TERM_MAP → 「唔」→「不」     │
│  ③ CANTONESE_FUZZY_MAP → 「錦朗」→「錦囊」 │
│  ④ 組合展開（俗語+粵語交叉替換）             │
│                                          │
│  展開結果全部送入 BM25 補充搜尋              │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  LLM Reranker（四級評分）                 │
│  90-100：直接回答問題核心                  │
│  60-89：相關但需推理                      │
│  50 以下：排除（信心門檻 0.5）              │
│  + 跨遊戲降分指令                           │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  LLM 回答                                │
│  Context 截斷至 MAX_CONTEXT_CHARS (8000) │
│  要求引用規則原文                         │
└──────────────────────────────────────────┘
```

### 3.4 API 呼叫量分析（按問題類型）

```
┌─────────────┬──────────────┬────────────────┬───────────────┐
│             │ FACTOID      │ PROCEDURAL     │ REASONING /   │
│             │ (Standard)   │ (輕量 DeepRAG)  │ COMPARISON    │
├─────────────┼──────────────┼────────────────┼───────────────┤
│ 語義標準化   │ 0~1 次       │ 0~1 次         │ 0~1 次        │
│ 問題分類     │ 1 次         │ 1 次           │ 1 次          │
│ 充足性判斷   │ 0 次         │ 1 次           │ 最多 3 次     │
│ Follow-up   │ 0 次         │ 1 次           │ 最多 3 次     │
│ Rerank      │ 1 次         │ 1 次           │ 1 次          │
│ 回答        │ 1 次         │ 1 次           │ 1 次          │
├─────────────┼──────────────┼────────────────┼───────────────┤
│ LLM 總計    │ 3~4 次 ⚡    │ 5~6 次         │ 7~10 次       │
│ Embedding   │ 1~5 次       │ 3~8 次         │ 5~13 次       │
└─────────────┴──────────────┴────────────────┴───────────────┘

BM25: build_bm25_retriever × 1（整個 session 只建一次）✅

對比舊版：所有問題一律 9 次 LLM 呼叫
新版 FACTOID：只需 3 次，回應速度提升 ~3 倍，準確度更高
```

---

## 四、新舊版本對比

| 項目 | 舊版 | 新版 |
|------|------|------|
| **架構** | 單一腳本 | 4 個職責分離模組 |
| **配置管理** | 硬編碼 | config.py + .env（40+ 參數 + 3 張映射表）|
| **支援遊戲** | 只有星杯傳說 | 星杯傳說 + 三國殺（自動掃描）|
| **向量集合** | 單一 collection | 每個遊戲獨立 collection |
| **重複索引** | 每次重複插入 | SHA-256 去重 |
| **Chunk 預處理** | 無 | HTML 註解清除 + CHUNK_SIZE 300 + 章節分隔符 |
| **BM25 分詞** | `.split()`（中文完全失效）| `_chinese_tokenize()` 逐字拆分 |
| **BM25 效能** | 每次搜尋重建索引 | 只建一次，session 複用 |
| **混合檢索合併** | 拼接去重 | RRF 加權合併（雙源命中分數疊加）|
| **檢索路由** | 一視同仁全走 DeepRAG | 問題分類 → Standard / DeepRAG 動態選擇 |
| **Rerank 候選** | 步驟 3 重複搜尋 | 迭代期間即時收集，直接 rerank |
| **Rerank 評分** | 簡單提示，門檻 0.3 | 四級評分標準 + 跨遊戲降分，門檻 0.5 |
| **充足性判斷** | 只看字數 | LLM 分析關鍵實體 |
| **對話記憶** | 無（Stateless）| 滑動窗口歷史 + 指代消解 |
| **Query 標準化** | 無 | 語義標準化引擎（粵語+俗語+指代 三合一）|
| **術語映射** | 無 | 四張表 125+ 組（俗語56+粵語56+拼音13+組合展開）|
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

# 交互範例 — 簡單事實查詢（走 Standard 路徑）
❓ [星杯傳說] 請輸入問題: 起始手牌幾張
🏷️  問題分類: ⚡ Standard（事實查詢，不迭代）
📋 使用的查詢 (1 條):
  [1] 起始手牌幾張
📊 Rerank 結果: 3 段
🤖 每位玩家依次從中摸取 4 張作為起始手牌。

# 交互範例 — 指代消解
❓ [星杯傳說] 請輸入問題: 那三國殺呢？
🏷️  問題分類: ⚡ Standard（事實查詢，不迭代）
🔄 指代消解: 「那三國殺呢？」→「三國殺的起始手牌是幾張？」
💬 對話歷史: 已記憶 1 輪 (最多保留 5 輪)
🤖 三國殺中每位玩家起始手牌為 4 張...

# 交互範例 — 粵語+俗語（語義標準化 + 術語展開雙軌）
❓ [星杯傳說] 請輸入問題: 暗滅對暗滅，可唔可以反彈？
🏷️  問題分類: ⚡ Standard（事實查詢，不迭代）
🔄 語義標準化: 「暗滅對暗滅，可唔可以反彈？」
             → 「暗滅對暗滅，可不可以應戰」
📋 使用的查詢 (5 條):
  [1] 暗滅對暗滅，可不可以應戰      ← LLM 改寫（主查詢）
  [2] 暗滅對暗滅，可唔可以應戰？    ← 俗語展開
  [3] 暗滅對暗滅，可不可以反彈？    ← 粵語展開
  [4] 暗滅對暗滅，可不可以應戰？    ← 組合展開
  ...
🤖 暗滅不可以被應戰。暗滅只能用聖盾或聖光來抵擋。

# 交互範例 — 複雜推理（走 DeepRAG 路徑）
❓ [星杯傳說] 請輸入問題: 如果主公死了忠臣會怎樣
🏷️  問題分類: 🧠 DeepRAG 完整（推理分析，最多 3 輪）
📋 使用的查詢 (7 條):
  [1] 如果主公死了忠臣會怎樣
  [2] 主公陣亡判定規則
  ...
🤖 ...

❓ [星杯傳說] 請輸入問題: history
💬 對話記憶（共 4 輪）：...

❓ [星杯傳說] 請輸入問題: back
↩️  返回遊戲選擇... (已清除 4 輪對話記憶)
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
| `config.py` | 配置中心（40+ 參數 + PLAYER_SLANG_MAP + CANTONESE 映射表 + Token 防護）| ✅ 使用中 |
| `ingestion.py` | 文件索引（HTML 清除 + 章節分隔符切分 + SHA-256 去重 + 多集合歸類）| ✅ 使用中 |
| `retriever.py` | 檢索核心（語義標準化 + 問題路由 + 中文 BM25 + RRF 合併 + 四級 Rerank）| ✅ 使用中 |
| `main.py` | 程式入口（兩層式選單 + 歷史管理 + Token 截斷 + 路徑標籤顯示）| ✅ 使用中 |
| `requirements.txt` | 依賴清單 | ✅ 使用中 |
| `DESIGN.md` | 本設計文件 | ✅ 使用中 |
| `ChromaDB.py` | 舊版索引腳本 | 🗑️ 可刪除 |
| `Standard-RAG.py` | 舊版 RAG | 🗑️ 可刪除 |
| `BM25-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `Similarity-search.py` | 舊版實驗 | 🗑️ 可刪除 |
| `DeepRAG.py` | 舊版最終腳本 | 🗑️ 可刪除 |
| `DeepRAG-rewrite.py` | 舊版中間版 | 🗑️ 可刪除 |
| `DeepRAG-backup.py` | 舊版早期版 | 🗑️ 可刪除 |

