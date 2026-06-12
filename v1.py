import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import argparse

# ================== 1. V5 模型架構 ==================
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
        x = x * mask.unsqueeze(-1)  # 遮罩過濾無效數據
        
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, S, F)
        gru_out, _ = self.gru(x_flat)
        gru_out = gru_out.reshape(B, N, S, self.gru_hidden)
        last_feat = gru_out[:, :, -1, :]  
        
        stock_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        embed_feat = self.stock_embed(stock_ids)  
        
        combined = torch.cat([last_feat, embed_feat], dim=-1)  
        transformer_input = self.project(combined)            
        
        valid_mask = (mask[:, -1, :] == 1).bool()
        stock_out = self.stock_transformer(transformer_input, src_key_padding_mask=~valid_mask)
        
        logits = self.fc(stock_out).squeeze(-1)
        return logits

# ================== 2. 資料集定義 ==================
class StockDataset(Dataset):
    def __init__(self, features, labels, mask, start_t, end_t, seq_len=60):
        self.features = features
        self.labels = labels
        self.mask = mask
        self.seq_len = seq_len
        
        # 確保 end_t 最大不超過 T - 20，避免採樣到未來的 NaN 標籤
        max_t = min(end_t, features.shape[0] - 20)
        
        # 建立有效索引池 (必須有足夠的 seq_len 歷史，且當天有股票可買)
        self.valid_indices = [
            i for i in range(max(seq_len, start_t), max_t) 
            if mask[i].sum() > 0
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len
        
        x = self.features[start_idx:end_idx]
        m = self.mask[start_idx:end_idx]
        y = self.labels[end_idx]
        
        return torch.FloatTensor(x), torch.FloatTensor(m), torch.FloatTensor(y)

# ================== 3. 訓練主程式 ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='processed_data/features.npz')
    parser.add_argument('--max_stocks', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gru_hidden', type=int, default=192)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    # 相容 GUI 給的參數 (避免 argparse 報錯)
    parser.add_argument('--train_window', type=int, default=2000)
    parser.add_argument('--step_size', type=int, default=250)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔥 啟動 V5.1 訓練引擎 (使用裝置: {device})")

    # 1. 載入資料
    data = np.load(args.data)
    features = data['features']
    labels = data['labels']
    mask = data['mask']
    T, N, F = features.shape
    
    print(f"資料載入完成: 時間 {T} 天, 股票 {N} 檔, 特徵 {F} 維")

    # 2. 嚴謹的資料切割 (切除最後 20 天無效標籤區)
    valid_T = T - 20
    train_end = int(valid_T * 0.9)
    
    print(f"資料切割邊界: 總長度 {T} | 有效邊界 {valid_T} | 訓練集切點 {train_end}")
    
    # 傳入完整的陣列，由 start_t 和 end_t 負責索引控制
    train_dataset = StockDataset(features, labels, mask, start_t=0, end_t=train_end)
    val_dataset = StockDataset(features, labels, mask, start_t=train_end, end_t=valid_T)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. 初始化模型與優化器
    model = ScaledGRUTransformer(
        num_stocks=N, feat_dim=F, gru_hidden=args.gru_hidden,
        embed_dim=args.embed_dim, nhead=8, num_layers=args.num_layers, dropout=0.2
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    # 4. 訓練迴圈
    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                logits = model(x, m)
                valid_mask = (m[:, -1, :] == 1).bool()
                
                if valid_mask.sum() == 0:
                    continue
                    
                loss = criterion(logits[valid_mask], y[valid_mask])
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # 驗證階段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                    logits = model(x, m)
                    valid_mask = (m[:, -1, :] == 1).bool()
                    if valid_mask.sum() > 0:
                        val_loss += criterion(logits[valid_mask], y[valid_mask]).item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 儲存最佳模型 (直接相容 backtest.py 預設的路徑)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/fold_3_best.pth')
            print(f"  👉 驗證損失下降，已儲存最新最佳權重！")

    print("✅ 訓練任務全部完成！")

if __name__ == '__main__':
    main()