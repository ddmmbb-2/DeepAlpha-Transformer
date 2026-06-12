import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
import os

# ================== 參數設定 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='processed_data/features.npz')
# 💡 預設改為指向 V5.1 的權重檔
parser.add_argument('--model', type=str, default='checkpoints/fold_3_best.pth', help='模型權重路徑')
parser.add_argument('--seq_len', type=int, default=60)
parser.add_argument('--gru_hidden', type=int, default=192, help='V5 大腦：隱含層對齊 192')
parser.add_argument('--embed_dim', type=int, default=32, help='股票實體嵌入的維度 32')
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.2, help='對齊訓練時的設定')
parser.add_argument('--top_k', type=int, default=50, help='顯示前幾名推薦股票')
parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
args = parser.parse_args()

# 確保自動偵測裝置安全
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

# ================== 模型定義（對齊 V5 完全體架構） ==================
class ScaledGRUTransformer(nn.Module):
    def __init__(self, num_stocks, feat_dim, gru_hidden, embed_dim, nhead, num_layers, dropout):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.embed_dim = embed_dim
        
        self.stock_embed = nn.Embedding(num_stocks, embed_dim)
        self.gru = nn.GRU(feat_dim, gru_hidden, batch_first=True)
        self.project = nn.Linear(gru_hidden + embed_dim, gru_hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gru_hidden, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.stock_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        B, S, N, F = x.shape
        x = x * mask.unsqueeze(-1)
        
        # 1. 跑個股時序 GRU
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, S, F)
        gru_out, _ = self.gru(x_flat)
        gru_out = gru_out.reshape(B, N, S, self.gru_hidden)
        last_feat = gru_out[:, :, -1, :]  
        
        # 2. 獲取個股 Embedding 身份
        stock_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        embed_feat = self.stock_embed(stock_ids)  
        
        # 3. 拼接與投影
        combined = torch.cat([last_feat, embed_feat], dim=-1)  
        transformer_input = self.project(combined)            
        
        # 4. 送入截面 Transformer
        valid_mask = (mask[:, -1, :] == 1).bool()
        stock_out = self.stock_transformer(transformer_input, src_key_padding_mask=~valid_mask)
        
        logits = self.fc(stock_out).squeeze(-1)
        return logits

# ================== 載入資料 ==================
print("載入最新打包特徵資料...")
data = np.load(args.data)
features = data['features']   # (T, N, F)
mask = data['mask']           # (T, N)
stocks = data['stocks']
dates = data['dates']

T, N, F = features.shape
print(f"✅ 資料載入成功。特徵形狀: T={T}, N={N}, F={F}")
print(f"📅 資料夾最新交易日期: {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

# 取最後 seq_len 天的資料
if T < args.seq_len:
    raise ValueError(f"資料天數不足 (需至少 {args.seq_len} 天)")

x = torch.FloatTensor(features[-args.seq_len:]).unsqueeze(0)  # (1, seq_len, N, F)
m = torch.FloatTensor(mask[-args.seq_len:]).unsqueeze(0)      # (1, seq_len, N)

# ================== 載入模型與權重 ==================
print("初始化 V5.1 模型架構...")
model = ScaledGRUTransformer(
    num_stocks=N,            
    feat_dim=F,
    gru_hidden=args.gru_hidden,
    embed_dim=args.embed_dim,
    nhead=args.nhead,
    num_layers=args.num_layers,
    dropout=args.dropout
).to(device)

print(f"正在載入權重檔案: {args.model} ...")
if not os.path.exists(args.model):
    raise FileNotFoundError(f"❌ 找不到模型權重：'{args.model}'，請確認是否已經跑完 v1.py 訓練")

model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

# ================== 預測 ==================
print("進行截面預測 (已啟用 3060 AMP 混合精度推論對齊)...")
with torch.no_grad():
    x, m = x.to(device), m.to(device)
    
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
        logits = model(x, m)                    # (1, N)
        
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

# ================== 產生推薦清單 ==================
# 考慮最後一天流動性合格且可交易的股票
last_mask = mask[-1] == 1
valid_indices = np.where(last_mask)[0]

# 只排序可交易的股票
valid_probs = probs[valid_indices]
valid_stocks = stocks[valid_indices]

# 取 Top-K
top_k = min(args.top_k, len(valid_stocks))
top_idx = np.argsort(valid_probs)[-top_k:][::-1]  # 從大到小

print(f"\n===== 👑 V5.1 完全體實盤推薦 Top-{top_k} =====\n")
for rank, idx in enumerate(top_idx, 1):
    stock = valid_stocks[idx]
    prob = valid_probs[idx]
    print(f"{rank:2d}. {stock:10s}  預測勝率評分: {prob:.4f}")

# 取得最後一天資料的日期字串
last_date_str = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
out_filename = f'recommendations_{last_date_str}.csv'

# 存成 CSV 檔案
result_df = pd.DataFrame({
    'rank': np.arange(1, top_k + 1),
    'stock': valid_stocks[top_idx],
    'score': valid_probs[top_idx]
})
result_df.to_csv(out_filename, index=False, encoding='utf-8-sig')
print(f"\n🎉 實盤推薦結果已成功儲存至 {out_filename}")
print("-" * 60)