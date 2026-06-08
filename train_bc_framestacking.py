import os
import glob
import json
import h5py
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# ==========================================================
# 1. DATASET PYTORCH 
# ==========================================================
class TorcsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self): 
        return len(self.X)
        
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

# ==========================================================
# 2. ARCHITETTURA STABILIZZATA CON LAYERNORM (ANTI-ZIGZAG)
# ==========================================================
class TorcsDriverNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        
        # Sostituito BatchNorm1d con LayerNorm per garantire stabilità millimetrica a Batch Size = 1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
            ) for _ in range(2)
        ])
        self.res_act = nn.ReLU(inplace=True)
        
        # Ridotto il dropout a 0.05 per eliminare l'incertezza sui decimali dello sterzo
        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden, 128), 
            nn.LayerNorm(128), 
            nn.ReLU(inplace=True)
        )
        
        self.head_steer = nn.Sequential(nn.Linear(128, 1), nn.Tanh())
        self.head_accel = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.head_brake = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        for block in self.res_blocks: 
            h = self.res_act(h + block(h))
        h = self.bottleneck(h)
        return torch.cat([self.head_steer(h), self.head_accel(h), self.head_brake(h)], dim=1)

# ==========================================================
# 3. MODELLO JIT END-TO-END SENZA PCA
# ==========================================================
class TorcsEndToEndNet(nn.Module):
    def __init__(self, net, scaler, input_dim_raw):
        super().__init__()
        self.net = net
        self.register_buffer("scaler_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(scaler.scale_, dtype=torch.float32))

    def forward(self, x_raw):
        # Normalizzazione diretta senza alterare le relazioni temporali dello stacking
        x = (x_raw - self.scaler_mean) / self.scaler_scale
        return self.net(x)

# ==========================================================
# 4. LOSS PESATA AD ALTA PRECISIONE
# ==========================================================
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sterzo alzato a 5.0 e freno a 2.0 per forzare la rete a inseguire la massima precisione
        self.register_buffer("weights", torch.tensor([5.0, 1.0, 2.0]))
        
    def forward(self, pred, target):
        return ((pred - target) ** 2 * self.weights).mean()

# ==========================================================
# CREAZIONE DELLO STATE STACKING (K=3)
# ==========================================================
def stack_lap_states(states, k=3):
    N, F = states.shape
    stacked = np.zeros((N, k * F), dtype=states.dtype)
    for i in range(N):
        f2 = states[max(0, i - 2)]
        f1 = states[max(0, i - 1)]
        f0 = states[i]
        stacked[i] = np.concatenate([f2, f1, f0])
    return stacked

# ==========================================================
# 5. LOOP DI ADDESTRAMENTO PRINCIPALE
# ==========================================================
def main():
    data_dir = "train_set/laps"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*60)
    print("  🧠 BEHAVIORAL CLONING — NO-PCA & LAYERNORM EDITION")
    print("="*60)
    
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[ERRORE] Nessun file .h5 trovato in {data_dir}.")
        return
        
    all_states, all_actions = [], []
    for f_path in h5_files:
        try:
            with h5py.File(f_path, 'r') as h5f:
                raw_lap_states = np.array(h5f['states'])
                stacked_lap_states = stack_lap_states(raw_lap_states, k=3)
                all_states.append(stacked_lap_states)
                actions = np.array(h5f['actions'])
                all_actions.append(actions[:, :3])
        except Exception as e:
            pass
            
    X_raw = np.concatenate(all_states, axis=0)
    y_raw = np.concatenate(all_actions, axis=0).astype(np.float32)
    input_dim_raw = X_raw.shape[1] 
    
    # ── Sottocampionamento rettilinei bilanciato correttamente ──
    steer_actions = np.abs(y_raw[:, 0])
    is_straight = steer_actions < 0.05
    is_curve = ~is_straight
    curve_indices = np.where(is_curve)[0]
    straight_indices = np.where(is_straight)[0]
    
    # Portato al 45% reali per equilibrare curve, rettilinei e recuperi
    keep_straights = np.random.choice(straight_indices, size=int(len(straight_indices)*0.45), replace=False)
    valid_indices = np.concatenate([curve_indices, keep_straights])
    np.random.shuffle(valid_indices)
    
    X_raw = X_raw[valid_indices]
    y_raw = y_raw[valid_indices]

    print("[PRE-PROC] Divisione dei dati in Train e Validation...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42
    )
    
    print("[PRE-PROC] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f: 
        pickle.dump(scaler, f)
    print(f"[DATI] Feature totali in ingresso alla rete (No PCA): {input_dim_raw}")

    train_dataset = TorcsDataset(X_train_scaled, y_train)
    val_dataset = TorcsDataset(X_val_scaled, y_val)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = TorcsDriverNet(input_dim=input_dim_raw).to(device)
    criterion = WeightedMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float('inf')
    patience = 15
    epochs_no_improve = 0
    
    print(f"\n[TRAIN] Avvio su: {device} | Parametri Rete: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(xb), yb)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
                
            train_loss += loss.item() * len(xb)
            
        avg_train_loss = train_loss / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        avg_val_loss = val_loss / len(val_dataset)
        
        scheduler.step()
        saved_flag = ""
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_weights.pth"))
            saved_flag = " ★ BEST"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:>3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f} | Tempo: {time.time()-t0:.1f}s{saved_flag}")

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Fermato all'epoca {epoch}. Modello ottimale salvato.")
            break

    print("\n[EXPORT] Compilazione JIT del modello End-to-End...")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_weights.pth"), map_location=device))
    model.to(torch.device("cpu"))
    model.eval()
    
    end_to_end_model = TorcsEndToEndNet(model, scaler, input_dim_raw)
    dummy_input = torch.zeros(1, input_dim_raw, dtype=torch.float32)
    
    traced_model = torch.jit.trace(end_to_end_model, dummy_input)
    traced_path = os.path.join(model_dir, "torcs_driver_jit.pt")
    traced_model.save(traced_path)
    
    print(f"[EXPORT] Modello JIT Stacked pronto (Senza PCA): {traced_path}")

if __name__ == "__main__":
    main()