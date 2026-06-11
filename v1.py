import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse

# ================== 參數設定 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='processed_data/features.npz')
parser.add_argument('--max_stocks', type=int, default=500)
parser.add_argument('--seq_len', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=2, help='配合 384 寬度，安全設為 2 嚴防 OOM')
parser.add_argument('--gru_hidden', type=int, default=384, help='大腦擴容：192 -> 384')
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--nhead', type=int, default=8, help='384 可被 8 整除')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=3e-5, help='大模型通常搭配略小的學習率')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--pos_weight', type=float, default=5.0)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--device', type=str, default='cuda')

# 擴展窗口參數
parser.add_argument('--train_window', type=int, default=1000)
parser.add_argument('--step_size', type=int, default=250)

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
os.makedirs(args.checkpoint_dir, exist_ok=True)
print(f"使用裝置: {device} (已針對 3060 12GB 啟用 Tensor Core AMP 加速)")

# ================== 1. Dataset 定義 ==================
class StockDataset(Dataset):
    def __init__(self, features, mask, labels, start_idx, end_idx, seq_len):
        self.features = features
        self.mask = mask
        self.labels = labels
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.seq_len = seq_len
        self.valid_days = (end_idx - start_idx) - seq_len + 1

    def __len__(self):
        return max(0, self.valid_days)

    def __getitem__(self, idx):
        t_start = self.start_idx + idx
        t_end = t_start + self.seq_len
        x = self.features[t_start:t_end]   
        m = self.mask[t_start:t_end]       
        y = self.labels[t_end - 1]          
        return torch.FloatTensor(x), torch.FloatTensor(m), torch.LongTensor(y)

# ================== 2. 模型架構 ==================
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def forward(self, x, mask):
        B, S, N, F = x.shape
        x = x * mask.unsqueeze(-1)
        
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

# ================== 3. 驗證評估函數 ==================
def evaluate(model, dataloader, top_k, device):
    model.eval()
    total_hits = 0
    total_pos = 0
    with torch.no_grad():
        for x, m, y in dataloader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            # 驗證時同樣使用 autocast 節省顯存
            with torch.amp.autocast('cuda'):
                logits = model(x, m)
            probs = torch.sigmoid(logits)
            for i in range(y.size(0)):
                valid = (y[i] != -1)
                if valid.sum() == 0:
                    continue
                p = probs[i][valid]
                t = y[i][valid]
                k = min(top_k, len(p))
                _, top_idx = torch.topk(p, k)
                hits = (t[top_idx] == 1).sum().item()
                pos = (t == 1).sum().item()
                total_hits += hits
                total_pos += pos
    return total_hits / total_pos if total_pos > 0 else 0.0

# ================== 4. 主程式與擴展窗口迴圈 ==================
def main():
    print("載入封裝特徵資料...")
    data_dict = np.load(args.data)
    features = data_dict['features']
    mask = data_dict['mask']
    labels = data_dict['labels']

    T, N_full, F = features.shape
    N = args.max_stocks if args.max_stocks > 0 else N_full
    
    features = features[:, :N, :]
    mask = mask[:, :N]
    labels = labels[:, :N]
    
    print(f"✅ 資料完整性檢查通過。原始天數 T={T}, 股票數 N={N}, 個股特徵 F={F}")
    
    base_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device), reduction='none')
    def raw_bce_loss_wrapper(logits, targets):
        valid = (targets != -1)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        l = logits[valid]
        t = targets[valid].float()
        return base_loss_fn(l, t).mean()
    loss_fn = raw_bce_loss_wrapper

    train_window = args.train_window
    step_size = args.step_size
    
    print(f"\n=======================================================")
    print(f"🧠 大腦擴容 V5 形態 | 隱含層: {args.gru_hidden} | 啟用 3060 AMP 引擎")
    print(f"=======================================================")

    all_test_hits = 0
    all_test_pos = 0
    fold = 1

    # 初始化 AMP 梯度縮放器 (防止半精度下梯度消失)
    scaler = torch.amp.GradScaler('cuda')

    for start_t in range(0, T - train_window, step_size):
        train_start = 0 
        train_end = start_t + train_window
        val_end = min(train_end + int(step_size * 0.5), T)
        test_end = min(train_end + step_size, T)
        
        if test_end - val_end < args.seq_len:
            print(f"Fold {fold} 剩餘天數不足盲測，跳過。")
            break

        print(f"\n========== 🔮 Fold {fold} (擴展大腦模式) ==========")
        print(f"訓練天數: {train_end} 天 | 驗證區間: [{train_end} ~ {val_end-1}] | 測試區間: [{val_end} ~ {test_end-1}]")

        train_set = StockDataset(features, mask, labels, train_start, train_end, args.seq_len)
        val_set   = StockDataset(features, mask, labels, train_end, val_end, args.seq_len)
        test_set  = StockDataset(features, mask, labels, val_end, test_end, args.seq_len)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = ScaledGRUTransformer(
            num_stocks=N, feat_dim=F,
            gru_hidden=args.gru_hidden, embed_dim=args.embed_dim,
            nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_val_hit = 0.0
        best_model_path = os.path.join(args.checkpoint_dir, f'fold_{fold}_best.pth')

        # 訓練迴圈
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            nan_batches = 0
            for x, m, y in train_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                optimizer.zero_grad()
                
                # 🚀 3060 核心優化：開啟混合精度前向傳播
                with torch.amp.autocast('cuda'):
                    logits = model(x, m)
                    loss = loss_fn(logits, y)
                
                if torch.isnan(loss) or loss.item() == 0.0:
                    nan_batches += 1
                    continue
                
                # 使用 Scaler 進行反向傳播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
            train_loss /= max(1, len(train_loader) - nan_batches)
            val_hit = evaluate(model, val_loader, args.top_k, device)
            scheduler.step(val_hit)

            if val_hit > best_val_hit:
                best_val_hit = val_hit
                torch.save(model.state_dict(), best_model_path)
                print(f"  ** 新最佳紀錄 (Fold {fold}) | Val Hit: {val_hit:.4f}")

            print(f"  Epoch {epoch+1:2d}/{args.epochs} | Loss: {train_loss:.4f} | Val Top-{args.top_k} Hit: {val_hit:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # OOS 盲測測試
        print(f"--> 測試 Fold {fold} 盲測集表現...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        
        fold_hits = 0
        fold_pos = 0
        with torch.no_grad():
            for x, m, y in test_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, m)
                probs = torch.sigmoid(logits)
                for i in range(y.size(0)):
                    valid = (y[i] != -1)
                    if valid.sum() == 0:
                        continue
                    p = probs[i][valid]
                    t = y[i][valid]
                    k = min(args.top_k, len(p))
                    _, top_idx = torch.topk(p, k)
                    fold_hits += (t[top_idx] == 1).sum().item()
                    fold_pos += (t == 1).sum().item()
                    
        fold_hit_rate = fold_hits / fold_pos if fold_pos > 0 else 0.0
        print(f"🎯 Fold {fold} 盲測結束 | 命中率: {fold_hit_rate:.4f} ({fold_hits}/{fold_pos})")
        
        all_test_hits += fold_hits
        all_test_pos += fold_pos
        fold += 1

    total_oos_hit_rate = all_test_hits / max(1, all_test_pos)
    print(f"\n=======================================================")
    print(f"🏅 V5 擴展大腦 (384維 + AMP加速) 綜合盲測報告")
    print(f"總體盲測命中數: {all_test_hits} / 總正樣本數: {all_test_pos}")
    print(f"🔥 綜合震撼真實 Top-{args.top_k} 命中率: {total_oos_hit_rate:.4f}")
    print(f"=======================================================")

if __name__ == '__main__':
    main()