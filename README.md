# DeepAlpha-Transformer 🚀

`DeepAlpha-Transformer` 是一個專為量化交易（Quantitative Trading）設計的端到端深度學習選股與預測系統。本專案旨在解決金融時序數據中極低的訊噪比（SNR）以及市場風格轉換（Market Regime Shift）帶來的預測瓶頸。

本系統的核心創舉在於融合了 **GRU 的時間序列特徵提取能力** 與 **Transformer 的全市場跨截面（Cross-Sectional）多頭注意力機制**，並開創性地引入**可訓練實體嵌入（Entity Embeddings）**，賦予模型對個股屬性的長效記憶，最終透過極其嚴謹的 **滾動式驗證（Walk-Forward Validation）** 進行盲測評估，並具備完整的實盤推論與績效驗證管線。

---

## 🌟 核心技術亮點 (Key Features)

### 1. 時序與截面雙層混合網路 (Hybrid Temporal-Cross-Sectional Network)

系統採用雙層深度學習網路架構：

* **時序特徵壓縮：** 利用 GRU 網路壓縮過去 $S = 60$ 個交易日的量價因子動態，提取精煉的時間序列隱含特徵（Hidden States）。
* **截面輪動注意力：** 將所有股票在同一交易日的時序特徵同時送入 Transformer Encoder。透過多頭注意力機制（Multi-Head Attention），模型能自發性地學習到個股在全市場中的相對強弱關係與資金在板塊間的輪動規律。

### 2. 股票實體嵌入 (Entity Embeddings)

針對金融數據中常見的「同線型、異走勢」（One-to-Many Mapping）難題，本專案在時序特徵後方拼接（Concat）了一個可訓練的 `nn.Embedding` 身份編碼層。透過線性投影層將動態時序資訊與全局個股身份融合，大幅降低模型的盲測負擔，使其能自主區分「高貝塔科技成長股」與「低波動傳統價值股」在相同形態下的本質差異。

### 3. 不平衡樣本優化 (Positive Weighting)

系統將預測目標定義為「未來 20 個交易日超額累積報酬率位居全市場前 20% 的強勢股」。面對天生類別不平衡，系統深度整合了 **BCEWithLogitsLoss 與正樣本加權機制 (Positive Weighting)**，透過動態調降易分類負樣本的權重，強迫模型將注意力聚焦在真正具有 Alpha 爆發潛力的非線性波動標的。

### 4. 嚴謹的滾動式驗證 (Walk-Forward Validation)

拒絕傳統機器學習在金融時序數據上不切實際的靜態切分（Static Split）。本專案完全實作了滾動時間窗口（Sliding Window）機制。每個 Fold 的模型皆會**徹底重置權重**，並依據設定的窗口大小進行訓練、驗證，以及最後嚴格的 Out-of-Sample (OOS) 盲測測試。透過多輪跨越不同年份的市場考驗，徹底杜絕潛在的看前預知偏差（Look-ahead Bias）。

---

## 📂 專案目錄結構 (Directory Structure)

```text
DeepAlpha-Transformer/
│
├── dl.py                 # 數據模組：yfinance 批次並行下載器
├── add.py                # 特徵模組：流動性篩選與 Expanding 因子工程
├── v1.py                 # 訓練模組：滾動式驗證與模型核心架構
├── infer.py              # 推論模組：實盤 Top-K 強勢股預測與推薦清單生成
├── verify.py             # 驗證模組：自動化回測與大盤基準績效比對 (🆕 新增)
│
├── ticker_list.txt       # 股票清單：每行一個代碼（如 2330.TW）
├── download.log          # 數據下載日誌
│
├── stock_data_batches/   # 暫存區：原始 CSV 批次數據夾
├── processed_data/       # 特徵區：封裝完成的標準化特徵矩陣 (features.npz)
└── checkpoints/          # 模型區：存放各個 Fold 盲測表現最優的權重

```

---

## ⚙️ 核心因子工程 (Feature Engineering)

在 `add.py` 中，系統會自發性篩選全市場平均成交量最大的 **500 檔高流動性標的**，並建構 **15 項核心量價與結構特徵**：

| 特徵名稱 | 描述 | 計算邏輯簡述 |
| --- | --- | --- |
| `ret` | 當日收益率 | 收盤價的當日百分比變動（Pct Change） |
| `amp` | 當日振幅 | (最高價 - 最低價) / 昨日收盤價 |
| `gap` | 跳空缺口 | (今日開盤價 - 昨日收盤價) / 昨日收盤價 |
| `vol_chg` | 成交量變動率 | 當日成交量的百分比變動 |
| `rel_vol` | 相對成交量 | 當日成交量 / 20日移動平均成交量 - 1 |
| `bias5` | 5日乖離率 | 今日收盤價 / 5日移動平均收盤價 - 1 |
| `bias20` | 20日乖離率 | 今日收盤價 / 20日移動平均收盤價 - 1 |
| `volatility5` | 5日波動率 | 過去 5 日收益率的滾動標準差（Rolling Std） |
| `mom20` | 20日動量 | 過去 20 個交易日的累積收益率 |
| `turnover` | 週轉率代理 | 當日成交量 / 20日移動平均成交量 |
| `price_pos` | 價格相對位置 | (今日收盤 - 20日最低) / (20日最高 - 20日最低) |
| `excess_20` | 20日累積超額報酬 | 個股累積收益率 - 全市場等權重平均收益率 |
| `dollar_vol` | 資金動能 | 收盤價 × 當日成交量 |
| `dist_high60` | 創高動能 | 今日收盤價 / 60日最高價 - 1 |
| `dist_low60` | 底部距離 | 今日收盤價 / 60日最低價 - 1 |

> 💡 **無洩漏標準化：** 所有特徵在成形後，一律通過擴展窗口（Expanding Window）計算均值與標準差，並向前平移一單位（`.shift(1)`），確保在標準化過程中**絕不引入未來數據的任何統計量**。

$$x'_{t} = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}}$$

---

## 🛠️ 快速開始與實盤管線 (Quick Start)

請確保你的 Python 環境已安裝相關依賴項（`torch`, `pandas`, `numpy`, `yfinance`, `tqdm`）。

### Step 1. 準備股票清單

在專案根目錄建立 `ticker_list.txt`，並填入標的代碼：

```text
2330.TW
2454.TW
2317.TW

```

### Step 2. 歷史數據下載

執行數據下載模組，系統會自動切分 Batch 並啟動隨機延遲（Anti-429 速率限制）抓取最新日 K 線：

```bash
python dl.py

```

### Step 3. 特徵工程與標籤封裝

融合批次 CSV 檔案，進行流動性前 500 檔篩選、時序標準化與未來 20 日 Alpha 標籤計算：

```bash
python add.py

```

### Step 4. 啟動滾動式訓練 (Windows 環境命令)

執行以下指令，啟動具備 Entity Embeddings 與混合精度加速 (AMP) 的滾動驗證流程：

```cmd
python v1.py ^
  --data processed_data/features.npz ^
  --max_stocks 500 ^
  --batch_size 4 ^
  --gru_hidden 192 ^
  --num_layers 4 ^
  --embed_dim 32 ^
  --pos_weight 5.0 ^
  --epochs 20 ^
  --lr 5e-5 ^
  --train_window 2000 ^
  --step_size 250 ^
  --device cuda

```

*(如果是 PowerShell 環境，請將行尾的 `^` 替換為 ```；或直接將整行串接成單一指令執行。)*

### Step 5. 實盤強勢股推論 (每日盤後作業)

載入最新一期（當前天數前 60 天）的特徵快照，配合受訓完成的最新 Fold 權重進行截面勝率預測：

```bash
python infer.py --model checkpoints/fold_3_best.pth --gru_hidden 192 --embed_dim 32 --top_k 50 --device cuda

```

預測完成後，系統會自動在根目錄生成帶有日期戳記的 CSV 檔案（例如 `recommendations_2026-06-09.csv`），依據超額報酬機率由大到小排序。

### Step 6. 策略績效驗證 (20天後回測) 🆕

當實盤推論經過一段時間後（例如 20 個交易日），可使用驗證模組自動抓取最新股價，結算該份推薦清單的真實勝率與投組報酬，並與大盤 (^TWII) 進行直觀對比：

```bash
python verify.py --file recommendations_2026-06-09.csv

```

系統將輸出完整的勝率報告，並導出 `verify_report_...csv` 供後續分析，形成完整的量化迭代閉環。
