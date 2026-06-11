import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# -------------------- 設定 --------------------
batch_dir = 'stock_data_batches'
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

TOP_LIQUID = 500  # 只保留成交量最大的前 N 檔

# -------------------- 1. 合併所有批次 --------------------
print("讀取與合併所有批次...")
all_files = sorted(glob.glob(os.path.join(batch_dir, 'batch_*.csv')))
print(f"找到 {len(all_files)} 個批次檔案")

dfs = []
for f in all_files:
    df = pd.read_csv(f, header=[0, 1], index_col=0, parse_dates=True)
    dfs.append(df)

full_df = pd.concat(dfs, axis=1)
full_df.sort_index(axis=1, inplace=True)
full_df = full_df.loc[:, ~full_df.columns.duplicated()]
print(f"去重後形狀: {full_df.shape}")

# -------------------- 流動性篩選 --------------------
print(f"根據平均成交量篩選前 {TOP_LIQUID} 檔...")
vol_raw = full_df.xs('Volume', axis=1, level=0)
avg_vol = vol_raw.mean()  # 每檔股票的時間平均成交量
top_stocks = avg_vol.nlargest(TOP_LIQUID).index.tolist()

# 只保留這些股票的欄位
full_df = full_df.loc[:, full_df.columns.get_level_values(1).isin(top_stocks)]
print(f"篩選後股票數: {full_df.shape[1] // len(full_df.columns.levels[0])}")

full_df.to_pickle(os.path.join(output_dir, 'all_stocks_raw.pkl'))
print("all_stocks_raw.pkl 已儲存（篩選後）")

# -------------------- 2. 特徵工程 --------------------
print("\n開始特徵工程 (V4 終極完全體版)...")

close = full_df.xs('Close',  axis=1, level=0)
high  = full_df.xs('High',   axis=1, level=0)
low   = full_df.xs('Low',    axis=1, level=0)
open_ = full_df.xs('Open',   axis=1, level=0)
vol   = full_df.xs('Volume', axis=1, level=0)

stocks = close.columns.tolist()
high   = high[stocks]
low    = low[stocks]
open_  = open_[stocks]
vol    = vol[stocks]

n_stocks = len(stocks)
print(f"股票總數: {n_stocks}, 交易日數: {len(close)}")

# 安全除法
def safe_div(a, b):
    return np.where(b != 0, a / b, 0.0)

print("計算 15 項黃金量價與結構因子...")
# ------ 核心 15 項特徵 ------
ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)

amp = pd.DataFrame(safe_div((high - low).values, close.shift(1).values), index=high.index, columns=high.columns).fillna(0)
gap = pd.DataFrame(safe_div((open_ - close.shift(1)).values, close.shift(1).values), index=open_.index, columns=open_.columns).fillna(0)
vol_chg = vol.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)

rolling_mean_vol = vol.rolling(20, min_periods=5).mean()
rel_vol = pd.DataFrame(safe_div(vol.values, rolling_mean_vol.values) - 1, index=vol.index, columns=vol.columns).fillna(0)

ma5 = close.rolling(5, min_periods=2).mean()
bias5 = pd.DataFrame(safe_div(close.values, ma5.values) - 1, index=close.index, columns=close.columns).fillna(0)

ma20 = close.rolling(20, min_periods=5).mean()
bias20 = pd.DataFrame(safe_div(close.values, ma20.values) - 1, index=close.index, columns=close.columns).fillna(0)

volatility5 = ret.rolling(5, min_periods=2).std().fillna(0)
mom20 = close.pct_change(20, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
turnover = pd.DataFrame(safe_div(vol.values, rolling_mean_vol.values), index=vol.index, columns=vol.columns).fillna(0)

high_20 = high.rolling(20, min_periods=5).max()
low_20  = low.rolling(20, min_periods=5).min()
price_pos = pd.DataFrame(safe_div((close - low_20).values, (high_20 - low_20).values), index=close.index, columns=close.columns).fillna(0.5)

market_ret = ret.mean(axis=1)
excess_20 = (ret.sub(market_ret, axis=0)).rolling(20, min_periods=5).sum().fillna(0)

# 高級對比特徵
dollar_vol = close * vol

high_60 = high.rolling(60, min_periods=15).max()
dist_high60 = pd.DataFrame(safe_div((close - high_60).values, high_60.values), index=close.index, columns=close.columns).fillna(0)

low_60 = low.rolling(60, min_periods=15).min()
dist_low60 = pd.DataFrame(safe_div((close - low_60).values, low_60.values), index=close.index, columns=close.columns).fillna(0)

# ------ 組合 15 項特徵 ------
feat_dict = {
    'ret': ret, 'amp': amp, 'gap': gap, 'vol_chg': vol_chg, 'rel_vol': rel_vol,
    'bias5': bias5, 'bias20': bias20, 'volatility5': volatility5, 'mom20': mom20,
    'turnover': turnover, 'price_pos': price_pos, 'excess_20': excess_20,
    'dollar_vol': dollar_vol, 'dist_high60': dist_high60, 'dist_low60': dist_low60
}
df_feat = pd.concat(feat_dict, axis=1, keys=feat_dict.keys())
df_feat = df_feat.swaplevel(0, 1, axis=1).sort_index(axis=1)   
F = len(feat_dict)
print(f"終極總特徵數: {F}")

# ------ 遮罩 ------
mask = close.notna() & (vol > 0)
mask = mask.astype(np.uint8)

# -------------------- 標籤：未來20日累積超額報酬前 20% --------------------
print("計算標籤 (未來20日超額報酬)...")
HOLD_DAYS = 20  
fwd_cum_ret = close.shift(-HOLD_DAYS) / close - 1
fwd_cum_ret = fwd_cum_ret.replace([np.inf, -np.inf], np.nan)

mkt_ret = pd.Series(np.nan, index=fwd_cum_ret.index)
for date in fwd_cum_ret.index:
    row = fwd_cum_ret.loc[date]
    valid = row.notna() & (mask.loc[date] == 1)
    if valid.sum() >= 10:
        mkt_ret.loc[date] = row[valid].mean()

excess_ret = fwd_cum_ret.sub(mkt_ret, axis=0)   

labels = pd.DataFrame(-1, index=fwd_cum_ret.index, columns=fwd_cum_ret.columns, dtype=np.int8)
for date in tqdm(excess_ret.index, desc="標籤計算"):
    row = excess_ret.loc[date]
    valid_mask = row.notna() & (mask.loc[date] == 1)
    if valid_mask.sum() < 10:
        continue
    threshold = row[valid_mask].quantile(0.8)
    strong = row[valid_mask].index[row[valid_mask] >= threshold]
    weak   = row[valid_mask].index[row[valid_mask] < threshold]
    labels.loc[date, strong] = 1
    labels.loc[date, weak]   = 0

# ------ 🚀 核心回滾：時間序列 Expanding Z-score 標準化 (擁抱 GRU 的時序連續性) ------
print("執行單向擴展窗口 (Expanding Window) 標準化...")
feat_list = list(feat_dict.keys())
normed_arrays = []

for feat in tqdm(feat_list, desc="時序標準化"):
    sub = df_feat.xs(feat, axis=1, level=1)
    sub = sub.reindex(columns=stocks)

    # 💡 迎回核心引擎：每檔股票計算自己過去的平均與標準差，並 shift(1) 杜絕未來數據
    exp_mean = sub.expanding(min_periods=5).mean().shift(1)
    exp_std  = sub.expanding(min_periods=5).std().shift(1)
    exp_std  = exp_std.replace(0, 1e-8)

    normed = (sub - exp_mean) / exp_std
    normed = normed.replace([np.inf, -np.inf], np.nan)
    normed = normed.fillna(0)
    normed = np.clip(normed, -10.0, 10.0)  # 保持寬廣的數值空間

    normed_arrays.append(normed.values.astype(np.float32))

feat_array = np.stack(normed_arrays, axis=2)
print(f"個股特徵處理完畢，形狀: {feat_array.shape}")

# NumPy 陣列轉換
mask_array = mask[stocks].values.astype(np.uint8)
label_array = labels[stocks].values.astype(np.int8)

assert not np.isnan(feat_array).any(), "❌ 特徵仍含 NaN"
assert not np.isinf(feat_array).any(), "❌ 特徵仍含 Inf"
print("✅ 特徵數值正常")

# ------ 儲存最終打包資料 ------
print(f"\n儲存最終壓縮特徵矩陣至 {output_dir}...")
np.savez_compressed(
    os.path.join(output_dir, 'features.npz'),
    features=feat_array,       # (T, N, F=15)
    mask=mask_array,           # (T, N)
    labels=label_array,        # (T, N)
    stocks=np.array(stocks),   
    dates=df_feat.index.values 
)
print(f"🎉 V4 完全體特徵工程封裝完畢！最終矩陣維度: {feat_array.shape}")