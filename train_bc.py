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
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================================
# 1. DATASET PYTORCH (Legge i file .h5 della tua data collection)
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
# 2. ARCHITETTURA AVANZATA CPU-OPTIMIZED (RESNET + BATCHNORM)
# ==========================================================
class TorcsDriverNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        # Encoder iniziale con BatchNorm per stabilizzare i gradienti
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), 
            nn.BatchNorm1d(hidden), 
            nn.ReLU(inplace=True)
        )
        # Blocco residuale per relazioni profonde
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            ) for _ in range(1)
        ])
        self.res_act = nn.ReLU(inplace=True)
        
        # Bottleneck di uscita
        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True)
        )
        # Teste separate per i 3 comandi principali
        self.head_steer = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.head_accel = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_brake = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        for block in self.res_blocks: 
            h = self.res_act(h + block(h))
        h = self.bottleneck(h)
        # Ritorna un vettore a 3 colonne: [Sterzo, Acceleratore, Freno]
        return torch.cat([self.head_steer(h), self.head_accel(h), self.head_brake(h)], dim=1)

# ==========================================================
# 3. MODELLO JIT END-TO-END PER IL DEPLOYMENT SU CPU (CORRETTO)
# ==========================================================
class TorcsEndToEndNet(nn.Module):
    def __init__(self, net, scaler, pca, input_dim_raw):
        super().__init__()
        self.net = net
        # Costanti matematiche incorporate direttamente nel grafo PyTorch
        self.register_buffer("scaler_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("pca_comps", torch.tensor(pca.components_.T, dtype=torch.float32))

    def forward(self, x_raw):
        # Normalizzazione in tempo reale pulita: usa direttamente lo scaler addestrato
        x = (x_raw - self.scaler_mean) / self.scaler_scale
        x = torch.matmul(x - self.pca_mean, self.pca_comps)
        return self.net(x)

# ==========================================================
# 4. LOSS PESATA (3 ELEMENTI: STERZO, ACCELERATORE, FRENO)
# ==========================================================
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Pesi ottimizzati: Sterzo=2.0, Acceleratore=1.0, Freno=5.0 (alta sensibilità sulle staccate)
        self.register_buffer("weights", torch.tensor([2.0, 1.0, 5.0]))
        
    def forward(self, pred, target):
        return ((pred - target) ** 2 * self.weights).mean()

# ==========================================================
# 5. LOOP DI ADDESTRAMENTO PRINCIPALE (SOLO CPU)
# ==========================================================
def main():
    data_dir = "train_set/laps"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*60)
    print("  🧠 BEHAVIORAL CLONING (Ottimizzato CPU - 3 Output - FIX)")
    print("="*60)
    
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[ERRORE] Nessun file .h5 trovato in {data_dir}.")
        return
        
    all_states, all_actions = [], []
    for f_path in h5_files:
        try:
            with h5py.File(f_path, 'r') as h5f:
                all_states.append(np.array(h5f['states']))
                # Isoliamo rigorosamente le prime 3 colonne (Sterzo, Accel, Brake)
                actions = np.array(h5f['actions'])
                all_actions.append(actions[:, :3])
        except Exception as e:
            pass
            
    X_raw = np.concatenate(all_states, axis=0)
    y_raw = np.concatenate(all_actions, axis=0).astype(np.float32)
    input_dim_raw = X_raw.shape[1]
    
    # ── Sottocampionamento rettilinei per dare priorità alle curve ──
    steer_actions = np.abs(y_raw[:, 0])
    is_straight = steer_actions < 0.05
    is_curve = ~is_straight
    curve_indices = np.where(is_curve)[0]
    straight_indices = np.where(is_straight)[0]
    
    keep_straights = np.random.choice(straight_indices, size=int(len(straight_indices)*0.2), replace=False)
    valid_indices = np.concatenate([curve_indices, keep_straights])
    np.random.shuffle(valid_indices)
    
    X_raw = X_raw[valid_indices]
    y_raw = y_raw[valid_indices]

    # ── Pre-Processing (StandardScaler + PCA 95%) ──
    print("[PRE-PROC] Fitting StandardScaler di Scikit-Learn...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("[PRE-PROC] Fitting PCA (95% varianza trattenuta)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    input_dim_pca = pca.n_components_
    
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "pca.pkl"), "wb") as f: pickle.dump(pca, f)
    print(f"[DATI] Feature Originali: {input_dim_raw} -> Ridotte via PCA a: {input_dim_pca}")

    # ── Dataset e Dataloader ──
    full_dataset = TorcsDataset(X_pca, y_raw)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Forziamo l'ambiente a girare interamente su CPU
    device = torch.device("cpu")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # ── Configurazione Modello e Ottimizzatore ──
    model = TorcsDriverNet(input_dim=input_dim_pca).to(device)
    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float('inf')
    print(f"\n[TRAIN] Avvio su: {device} | Parametri Rete: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Esecuzione standard FP32 pura per CPU
            loss = criterion(model(xb), yb)
            loss.backward()
            
            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
                
            train_loss += loss.item() * len(xb)
            
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validazione
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
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:>3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f} | Tempo: {time.time()-t0:.1f}s{saved_flag}")

    # ==========================================================
    # 6. COMPILAZIONE JIT END-TO-END FINALIZZATA PER CPU
    # ==========================================================
    print("\n[EXPORT] Compilazione JIT del modello End-to-End...")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_weights.pth")))
    model.eval()
    
    end_to_end_model = TorcsEndToEndNet(model, scaler, pca, input_dim_raw)
    dummy_input = torch.zeros(1, input_dim_raw, dtype=torch.float32)
    
    # Tracciamento ed esportazione nativa JIT
    traced_model = torch.jit.trace(end_to_end_model, dummy_input)
    traced_path = os.path.join(model_dir, "torcs_driver_jit.pt")
    traced_model.save(traced_path)
    
    print(f"[EXPORT] Modello JIT per CPU pronto e salvato in: {traced_path}")
    print("  Il file include al suo interno Scaler e PCA perfetti. Output generati: [Sterzo, Accel, Brake].")

if __name__ == "__main__":
    main()