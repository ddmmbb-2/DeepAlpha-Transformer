import yfinance as yf
import pandas as pd
import time
import random
import logging
import os
from datetime import datetime

# ==================== 設定區 ====================
# 1. 基礎目錄與檔案設定
TICKER_FILE = 'ticker_list.txt'
OUTPUT_DIR = 'stock_data_batches'
MACRO_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'macro_global.csv')

# 2. 海外宏觀與美股巨頭清單
MACRO_TICKERS = ['^NDX', '^GSPC', '^VIX', '^SOX', 'NVDA', 'AAPL', 'AMD']

# 3. 下載日期範圍
START_DATE = '2015-01-01'
END_DATE   = '2023-12-31'

# 4. 每批股票數量與休息時間
BATCH_SIZE = 10
MIN_SLEEP = 30   # 0.5 分鐘
MAX_SLEEP = 120   # 2 分鐘

# 5. 最大重試次數（下載失敗時）
MAX_RETRIES = 3

# ==================== 股票清單載入函數 ====================
def load_tickers(filepath):
    """從檔案讀取股票代碼，自動過濾空行與前後空白"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到 {filepath}，請先產生股票清單檔")
    with open(filepath, 'r', encoding='utf-8') as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError(f"{filepath} 沒有讀取到任何代碼")
    return tickers

# 實質載入台股清單
TICKERS = load_tickers(TICKER_FILE)

# ==================== 日誌設定 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==================== 核心下載函數 ====================
def create_session():
    """建立帶有瀏覽器偽裝的 requests Session"""
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    return session

def download_batch(tickers, start, end, session):
    """
    下載一批股票的歷史資料，失敗時會重試。
    回傳 DataFrame（若成功）或 None（若最終失敗）。
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"嘗試下載 {tickers} (第 {attempt}/{MAX_RETRIES} 次)")
            data = yf.download(
                tickers,
                start=start,
                end=end,
                session=session,
                progress=False,   # 關閉 yfinance 的進度條，避免干擾日誌
                auto_adjust=True  # 自動調整股價（還原權息）
            )
            if data.empty:
                logging.warning(f"下載完成但回傳空資料，可能代碼錯誤或日期無交易：{tickers}")
                return data
            logging.info(f"成功下載 {len(tickers)} 檔股票，資料筆數：{len(data)}")
            return data
        except Exception as e:
            logging.error(f"下載失敗：{e}")
            if "429" in str(e):
                wait_time = 60 * attempt
                logging.warning(f"偵測到 429 速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                wait_time = random.uniform(5, 15)
                logging.info(f"其他錯誤，等待 {wait_time:.1f} 秒後重試...")
                time.sleep(wait_time)
    logging.error(f"最終下載失敗，跳過此批次：{tickers}")
    return None

def save_batch(data, batch_idx, tickers):
    """將一個批次的 DataFrame 存成 CSV"""
    filename = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}.csv")
    data.to_csv(filename, encoding='utf-8-sig')
    logging.info(f"已儲存 {filename}，包含股票：{tickers}")

# ==================== 主程式 ====================
def main():
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 準備 session
    session = create_session()
    logging.info("Session 初始化完成（瀏覽器偽裝）")

    # 將股票清單分批
    batches = [TICKERS[i:i+BATCH_SIZE] for i in range(0, len(TICKERS), BATCH_SIZE)]
    total_batches = len(batches)
    logging.info(f"共 {len(TICKERS)} 檔股票，分為 {total_batches} 批，每批 {BATCH_SIZE} 檔")
    logging.info(f"日期範圍：{START_DATE} 至 {END_DATE}")
    logging.info(f"批次間休息 {MIN_SLEEP}～{MAX_SLEEP} 秒")

    failed_batches = []

    # 1. 循環下載台股批次
    for idx, batch_tickers in enumerate(batches, start=1):
        logging.info(f"========== 批次 {idx}/{total_batches}：{batch_tickers} ==========")
        data = download_batch(batch_tickers, START_DATE, END_DATE, session)

        if data is not None and not data.empty:
            save_batch(data, idx, batch_tickers)
        else:
            failed_batches.append((idx, batch_tickers))
            logging.warning(f"批次 {idx} 下載失敗或無資料，記錄下來待稍後處理")

        # 最後一批執行完畢後不需要休息
        if idx < total_batches:
            sleep_sec = random.uniform(MIN_SLEEP, MAX_SLEEP)
            logging.info(f"休息 {sleep_sec/60:.1f} 分鐘...")
            time.sleep(sleep_sec)

    # 2. 下載海外宏觀數據 (放在台股循環結束後)
    logging.info("========== 開始下載海外宏觀與美股巨頭數據 ==========")
    logging.info(f"海外標的：{MACRO_TICKERS}")
    
    macro_data = download_batch(MACRO_TICKERS, START_DATE, END_DATE, session)
    
    if macro_data is not None and not macro_data.empty:
        macro_data.to_csv(MACRO_OUTPUT_FILE, encoding='utf-8-sig')
        logging.info(f"✅ 海外宏觀數據下載成功！已儲存至：{MACRO_OUTPUT_FILE}")
    else:
        logging.error("❌ 海外宏觀數據下載失敗，請檢查網路或代碼。")

    # 總結
    logging.info("==================== 全部批次執行完畢 ====================")
    if failed_batches:
        logging.warning(f"以下批次失敗或無資料，請檢查股票代碼或手動重試：")
        for i, tickers in failed_batches:
            logging.warning(f"批次 {i}: {tickers}")
    else:
        logging.info("所有批次皆成功下載！")

    logging.info(f"所有資料已存放於資料夾：{os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()