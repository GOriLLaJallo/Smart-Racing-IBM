import os
import glob
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================================
# 1. CLASSE PER LA NORMALIZZAZIONE DEI DATI
# ==========================================================
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-6] = 1.0
        
    def transform(self, X):
        return (X - self.mean) / self.std
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'mean': self.mean.tolist(), 'std': self.std.tolist()}, f)
            print(f"  [+] Scaler salvato in {path}")
            
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.mean = np.array(data['mean'], dtype=np.float32)
            self.std = np.array(data['std'], dtype=np.float32)

# ==========================================================
# 2. DATASET PYTORCH (SINGOLO FRAME)
# ==========================================================
class TorcsDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        # Nessun Frame Stacking, passiamo l'istante puro
        return self.states[idx], self.actions[idx]

# ==========================================================
# 3. MODELLO DI RETE NEURALE (CON DROPOUT)
# ==========================================================
class BCModel(nn.Module):
    def __init__(self, input_dim=29, output_dim=3):
        super(BCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # <-- NOVITÀ: Spegne il 20% dei neuroni (Anti-Overfitting)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # <-- NOVITÀ: Costringe la rete a generalizzare
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        out = self.net(x)
        steer = torch.tanh(out[:, 0:1])
        pedals = torch.sigmoid(out[:, 1:3])
        return torch.cat([steer, pedals], dim=1)

# ==========================================================
# 4. LOSS PERSONALIZZATA (WEIGHTED MSE)
# ==========================================================
def weighted_mse_loss(predictions, targets, device):
    sq_error = (predictions - targets) ** 2
    weights = torch.tensor([1.0, 1.0, 5.0], device=device)
    weighted_sq_error = sq_error * weights
    return weighted_sq_error.mean()

# ==========================================================
# 5. FUNZIONE PRINCIPALE DI ADDESTRAMENTO
# ==========================================================
def main():
    data_dir = "train_set/laps"
    
    print("="*60)
    print("  🧠 BEHAVIORAL CLONING (Anti-Overfitting & Stabile)")
    print("="*60)
    
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"ERRORE: Nessun file .h5 trovato in {data_dir}.")
        return
        
    all_states, all_actions = [], []
    for f_path in h5_files:
        try:
            with h5py.File(f_path, 'r') as h5f:
                all_states.append(np.array(h5f['states']))
                all_actions.append(np.array(h5f['actions']))
        except Exception as e:
            pass
            
    X_raw = np.concatenate(all_states, axis=0)
    Y_raw = np.concatenate(all_actions, axis=0)[:, :3]
    
    # ── Normalizzazione ──
    normalizer = Normalizer()
    normalizer.fit(X_raw)
    X_norm = normalizer.transform(X_raw)
    normalizer.save("bc_scaler.json")
    
    print(f"\n[+] Caricati {len(X_norm)} campioni totali (Senza filtri sui rettilinei).")
    
    # ── Validation Split (80/20) ──
    full_dataset = TorcsDataset(X_norm, Y_raw)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # ── Inizializzazione ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel(input_dim=29, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # NOVITÀ: Parametri molto più stringenti per evitare la "memoria fotografica"
    epochs = 150       # Massimo 80 epoche per non farlo studiare troppo
    patience = 10     # Si ferma prima se non vede miglioramenti netti
    patience_counter = 0
    best_val_loss = float('inf')
    
    print(f"\n[+] Addestramento su: {device} | Input: 29D")
    
    for epoch in range(1, epochs + 1):
        # -- Fase di Training --
        model.train()
        train_loss = 0.0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            
            predictions = model(states)
            loss = weighted_mse_loss(predictions, actions, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # -- Fase di Validazione --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                predictions = model(states)
                loss = weighted_mse_loss(predictions, actions, device)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "bc_model.pth")
            saved_flag = " ★ SAVED"
        else:
            patience_counter += 1
            
        print(f"  Epoca [{epoch:03d}/{epochs}] - Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}{saved_flag}")
        
        if patience_counter >= patience:
            print(f"\n  ⏹ Early Stopping: nessun miglioramento per {patience} epoche.")
            break
            
    print("\n[+] ADDESTRAMENTO COMPLETATO!")
    print("  Il modello è pronto in 'bc_model.pth'.")

if __name__ == "__main__":
    main()