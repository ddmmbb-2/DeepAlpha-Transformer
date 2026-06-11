import pandas as pd
import yfinance as yf
import argparse
import os
import re
from datetime import datetime, timedelta
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='驗證模型推薦清單的真實績效')
    parser.add_argument('--file', type=str, required=True, help='推薦清單 CSV 的檔案路徑 (例如: recommendations_2026-06-09.csv)')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案：{file_path}")
        return

    # 1. 從檔名解析「推薦日期」(建倉日)
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(file_path))
    if not date_match:
        print("❌ 無法從檔名解析出日期，請確保檔名包含 YYYY-MM-DD 格式。")
        return
    
    start_date = date_match.group(0)
    # yfinance 下載的結束日期預設為今天
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    print(f"🔍 開始驗證績效...")
    print(f"建倉基準日 (推薦日): {start_date}")
    print(f"結算驗證日 (今天): {today_str}")

    # 2. 讀取推薦清單
    df = pd.read_csv(file_path)
    tickers = df['stock'].tolist()
    
    # 加入台灣加權指數作為大盤對照
    benchmark_ticker = '^TWII'
    download_tickers = tickers + [benchmark_ticker]

    # 3. 下載歷史價格資料
    print(f"⬇️ 正在從 Yahoo Finance 獲取 {len(tickers)} 檔股票與大盤資料...")
    # 結束日期加1天確保能抓到今天的資料
    end_date_yf = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(download_tickers, start=start_date, end=end_date_yf, progress=False, auto_adjust=True)

    if data.empty:
        print("❌ 無法獲取價格資料。")
        return

    # 取得收盤價矩陣
    close_prices = data['Close']

    # 確保資料至少有兩天可以比較
    if len(close_prices) < 2:
        print("⚠️ 交易天數不足。可能是今天才剛推薦，還沒有未來的收盤價可以驗證。")
        return

    # 4. 計算績效
    results = []
    actual_start_date = close_prices.index[0].strftime('%Y-%m-%d')
    actual_end_date = close_prices.index[-1].strftime('%Y-%m-%d')

    for stock in tickers:
        if stock not in close_prices.columns:
            continue
            
        stock_series = close_prices[stock].dropna()
        if len(stock_series) < 2:
            continue
            
        buy_price = stock_series.iloc[0]   # 基準日收盤價
        current_price = stock_series.iloc[-1] # 最新收盤價
        ret_pct = ((current_price - buy_price) / buy_price) * 100
        
        results.append({
            'Stock': stock,
            'Buy Price': round(buy_price, 2),
            'Current Price': round(current_price, 2),
            'Return (%)': round(ret_pct, 2)
        })

    # 5. 計算大盤績效
    bench_series = close_prices[benchmark_ticker].dropna()
    bench_ret = 0.0
    if len(bench_series) >= 2:
        bench_ret = ((bench_series.iloc[-1] - bench_series.iloc[0]) / bench_series.iloc[0]) * 100

    # 6. 統整與輸出結果
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("❌ 無法計算有效回報。")
        return

    # 依照報酬率由高到低排序
    res_df = res_df.sort_values(by='Return (%)', ascending=False).reset_index(drop=True)
    
    # 統計數據
    avg_return = res_df['Return (%)'].mean()
    win_rate = (res_df['Return (%)'] > 0).mean() * 100
    beat_market_rate = (res_df['Return (%)'] > bench_ret).mean() * 100

    print(f"\n================ 驗證報告 ================")
    print(f"實際計算區間: {actual_start_date} 至 {actual_end_date} (共 {len(bench_series)} 個交易日)")
    print(f"大盤 (^TWII) 同期報酬率: {bench_ret:.2f}%")
    print(f"------------------------------------------")
    print(f"💰 投組平均報酬率: {avg_return:.2f}%")
    print(f"🎯 投組絕對勝率 (賺錢比例): {win_rate:.2f}%")
    print(f"⚔️ 擊敗大盤勝率 (跑贏大盤比例): {beat_market_rate:.2f}%")
    print(f"==========================================")
    
    print("\n[個股績效排行 Top 5]")
    print(res_df.head(5).to_string(index=False))
    print("\n[個股績效排行 Bottom 5]")
    print(res_df.tail(5).to_string(index=False))

    # 存檔
    out_file = f"verify_report_{start_date}_to_{actual_end_date}.csv"
    res_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n📁 完整驗證明細已存至: {out_file}")

if __name__ == '__main__':
    main()