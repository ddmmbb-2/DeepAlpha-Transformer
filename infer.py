import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd

# ================== 參數 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='processed_data/features.npz')
parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
parser.add_argument('--seq_len', type=int, default=60)
parser.add_argument('--gru_hidden', type=int, default=128)   # 需與訓練時一致
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=20, help='顯示前幾名')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()
parser.add_argument('--embed_dim', type=int, default=32, help='股票實體嵌入的維度')

device = torch.device(args.device)

# ================== 模型定義（與訓練相同） ==================
class ScaledGRUTransformer(nn.Module):
    def __init__(self, feat_dim, gru_hidden, nhead, num_layers, dropout):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(feat_dim, gru_hidden, batch_first=True)
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
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, S, F)
        gru_out, _ = self.gru(x_flat)
        gru_out = gru_out.reshape(B, N, S, self.gru_hidden)
        last_feat = gru_out[:, :, -1, :]
        valid_mask = (mask[:, -1, :] == 1).bool()
        stock_out = self.stock_transformer(last_feat, src_key_padding_mask=~valid_mask)
        logits = self.fc(stock_out).squeeze(-1)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return logits

# ================== 載入資料 ==================
print("載入資料...")
data = np.load(args.data)
features = data['features']   # (T, N, F)
mask = data['mask']           # (T, N)
stocks = data['stocks']
dates = data['dates']

T, N, F = features.shape
print(f"資料維度: T={T}, N={N}, F={F}")
print(f"最新日期: {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

# 取最後 seq_len 天的資料（模型輸入需為 batch=1, seq_len, N, F）
if T < args.seq_len:
    raise ValueError(f"資料天數不足 (需至少 {args.seq_len} 天)")

x = torch.FloatTensor(features[-args.seq_len:]).unsqueeze(0)  # (1, seq_len, N, F)
m = torch.FloatTensor(mask[-args.seq_len:]).unsqueeze(0)      # (1, seq_len, N)

# ================== 載入模型 ==================
print("載入模型...")
model = ScaledGRUTransformer(
    num_stocks=N,              # 推論時的 N 是從 features.shape[1] 自動獲取
    feat_dim=F,
    gru_hidden=args.gru_hidden,
    embed_dim=args.embed_dim,
    nhead=args.nhead,
    num_layers=args.num_layers,
    dropout=args.dropout
).to(device)

# ================== 預測 ==================
print("進行預測...")
with torch.no_grad():
    x, m = x.to(device), m.to(device)
    logits = model(x, m)                    # (1, N)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

# ================== 產生推薦清單 ==================
# 考慮最後一天可交易的股票（mask[最後一天] == 1）
last_mask = mask[-1] == 1
valid_indices = np.where(last_mask)[0]

# 只排序可交易的股票
valid_probs = probs[valid_indices]
valid_stocks = stocks[valid_indices]

# 取 Top-K
top_k = min(args.top_k, len(valid_stocks))
top_idx = np.argsort(valid_probs)[-top_k:][::-1]  # 從大到小

print(f"\n===== 推薦 Top-{top_k} 股票 (依超額報酬機率排序) =====")
for rank, idx in enumerate(top_idx, 1):
    stock = valid_stocks[idx]
    prob = valid_probs[idx]
    print(f"{rank:2d}. {stock:10s}  機率: {prob:.4f}")

# 可選：存成 CSV
result_df = pd.DataFrame({
    'stock': valid_stocks[top_idx],
    'score': valid_probs[top_idx]
})
result_df.to_csv('recommendations.csv', index=False, encoding='utf-8-sig')
print("\n結果已儲存至 recommendations.csv")