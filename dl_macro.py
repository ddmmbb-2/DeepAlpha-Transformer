import yfinance as yf
import os

OUTPUT_DIR = 'stock_data_batches'
MACRO_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'macro_global.csv')
MACRO_TICKERS = ['^NDX', '^GSPC', '^VIX', '^SOX', 'NVDA', 'AAPL', 'AMD']

print(f"正在單獨補載海外指標與美股巨頭: {MACRO_TICKERS}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 一次性快速下載
data = yf.download(MACRO_TICKERS, start='2015-01-01', end='2023-12-31', auto_adjust=True)

if not data.empty:
    data.to_csv(MACRO_OUTPUT_FILE, encoding='utf-8-sig')
    print(f"================")
    print(f"✅ 成功補載海外數據！檔案已存至: {MACRO_OUTPUT_FILE}")
    print(f"================")
else:
    print("❌ 下載失敗，請檢查網路連線。")