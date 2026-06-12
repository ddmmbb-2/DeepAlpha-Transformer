import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def main():
    print("🚀 啟動 V5.1 終極特徵與標籤工程 (無未來函數/無生存者偏差版)")
    
    raw_path = 'processed_data/all_stocks_raw.pkl'
    out_path = 'processed_data/features.npz'
    
    if not os.path.exists(raw_path):
        print(f"❌ 找不到原始資料: {raw_path}")
        return

    print("📦 載入原始股價資料...")
    full_df = pd.read_pickle(raw_path)
    close_raw = full_df.xs('Close', axis=1, level=0)
    vol_raw = full_df.xs('Volume', axis=1, level=0)

    # ==========================================
    # 🛡️ 1. 動態流動性遮罩 (絕對不偷看未來)
    # ==========================================
    print("🛡️ 計算動態流動性遮罩...")
    TOP_LIQUID = 500
    
    # 計算過去 20 天的滾動平均成交量
    rolling_vol = vol_raw.rolling(window=20, min_periods=1).mean()
    # 每天橫向排名，成交量越大名次越前
    vol_rank = rolling_vol.rank(axis=1, ascending=False)
    liquid_mask_df = vol_rank <= TOP_LIQUID
    
    # 必須要有收盤價
    price_mask_df = close_raw.notna() & (close_raw > 0)
    final_mask_df = price_mask_df & liquid_mask_df

    # ==========================================
    # 🏷️ 2. 終極標籤計算 (防禦生存者偏差)
    # ==========================================
    print("🏷️ 計算防禦型訓練標籤 (缺失值懲罰 -30%)...")
    labels_df = pd.DataFrame(0, index=close_raw.index, columns=close_raw.columns)
    
    # 未來 20 天的真實報酬
    fwd_cum_ret = close_raw.shift(-20) / close_raw - 1
    PENALTY_RETURN = -0.30  # 針對下市/長期停牌股的嚴厲懲罰

    for date in tqdm(fwd_cum_ret.index, desc="計算標籤"):
        current_eligible_stocks = final_mask_df.loc[date]
        if current_eligible_stocks.sum() == 0:
            continue
            
        row_ret = fwd_cum_ret.loc[date].copy()
        
        # 找出當天合格，但未來 20 天資料消失的「地雷股」，給予 -30% 懲罰
        missing_future = current_eligible_stocks & row_ret.isna()
        row_ret[missing_future] = PENALTY_RETURN
        
        # 計算包含地雷股在內的「真實截面市場平均」
        true_mkt_mean = row_ret[current_eligible_stocks].mean()
        
        # 計算超額報酬
        excess_ret = row_ret - true_mkt_mean
        valid_excess = excess_ret[current_eligible_stocks]
        
        if len(valid_excess) > 0:
            # 嚴格篩選前 20% 強勢股
            threshold = valid_excess.quantile(0.8)
            labels_df.loc[date, current_eligible_stocks] = (valid_excess >= threshold).astype(int)

    # ==========================================
    # 📊 3. 15 項黃金特徵計算 (Expanding Z-score)
    # ==========================================
    print("📊 提取 OHLCV 資料並計算 15 項量價特徵...")
    
    # 提取所需欄位
    open_raw = full_df.xs('Open', axis=1, level=0)
    high_raw = full_df.xs('High', axis=1, level=0)
    low_raw = full_df.xs('Low', axis=1, level=0)
    
    features_dict = {}
    
    # 1-3. 價格動量 (Momentum)
    daily_ret = close_raw / close_raw.shift(1) - 1
    features_dict['mom5'] = close_raw / close_raw.shift(5) - 1
    features_dict['mom20'] = close_raw / close_raw.shift(20) - 1
    features_dict['mom60'] = close_raw / close_raw.shift(60) - 1
    
    # 4-6. 均線乖離率 (Bias)
    features_dict['sma5_bias'] = close_raw / close_raw.rolling(5).mean() - 1
    features_dict['sma20_bias'] = close_raw / close_raw.rolling(20).mean() - 1
    features_dict['sma60_bias'] = close_raw / close_raw.rolling(60).mean() - 1
    
    # 7-9. 成交量比率 (Volume Ratio)
    features_dict['vol_ratio_5'] = vol_raw / vol_raw.rolling(5).mean()
    features_dict['vol_ratio_20'] = vol_raw / vol_raw.rolling(20).mean()
    features_dict['vol_ratio_60'] = vol_raw / vol_raw.rolling(60).mean()
    
    # 10-11. 波動度 (Volatility)
    features_dict['volatility_20'] = daily_ret.rolling(20).std()
    features_dict['volatility_60'] = daily_ret.rolling(60).std()
    
    # 12-14. K線型態與日內特徵
    features_dict['amplitude'] = (high_raw - low_raw) / close_raw.shift(1) # 振幅
    features_dict['gap'] = (open_raw - close_raw.shift(1)) / close_raw.shift(1) # 跳空
    features_dict['close_open_ratio'] = (close_raw - open_raw) / open_raw # 日內實體K線漲跌
    
    # 15. 價格區間位置 (Price Position)
    rolling_min_20 = low_raw.rolling(20).min()
    rolling_max_20 = high_raw.rolling(20).max()
    features_dict['price_pos_20'] = (close_raw - rolling_min_20) / (rolling_max_20 - rolling_min_20 + 1e-8)

    # 整合特徵 (T, N, F) 並進行 Expanding Z-score 標準化
    print("🔄 進行極致嚴格的無未來函數 Z-score 標準化 (啟動 NaN 防護罩)...")
    feat_names = list(features_dict.keys())
    T, N = close_raw.shape
    F = len(feat_names)
    X = np.zeros((T, N, F), dtype=np.float32)
    
    for i, fname in enumerate(tqdm(feat_names, desc="標準化進度")):
        df_f = features_dict[fname].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 計算 expanding 序列並向後平移一天
        exp_mean = df_f.expanding().mean().shift(1)
        exp_std = df_f.expanding().std().shift(1)
        
        # 暴力填補 shift 與 std 造成的前期 NaN
        rolling_mean = exp_mean.bfill().fillna(0).values
        rolling_std = exp_std.bfill().fillna(1e-8).values
        
        rolling_std[rolling_std == 0] = 1e-8
        
        # 計算 Z-score
        z_score = (df_f.values - rolling_mean) / rolling_std
        
        # 強勢歸零殘存的 NaN 或 Inf
        z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 🛡️ 【終極防護：AMP 溢位殺手】將 Z-score 限制在正負 10 之間！
        z_score = np.clip(z_score, -10.0, 10.0)
        
        X[:, :, i] = z_score

    # ==========================================
    # 💾 4. 儲存打包
    # ==========================================
    mask_arr = final_mask_df.values.astype(int)
    y_arr = labels_df.values.astype(int)
    
    print(f"📦 打包資料: 特徵形狀 {X.shape}, 遮罩形狀 {mask_arr.shape}")
    np.savez_compressed(
        out_path,
        features=X,
        labels=y_arr,
        mask=mask_arr,
        stocks=close_raw.columns.values,
        dates=close_raw.index.astype(str).values
    )
    print("✅ 特徵資料與標籤重構完成！")

if __name__ == '__main__':
    main()