import os
import glob
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# 1. CLASSE PER LA NORMALIZZAZIONE DEI DATI
# ==========================================================
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        # Calcola media e deviazione standard per ogni colonna (sensore)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Evita divisioni per zero se un sensore ha sempre lo stesso valore
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
# 2. DATASET PYTORCH
# ==========================================================
class TorcsDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ==========================================================
# 3. MODELLO DI RETE NEURALE (MLP)
# ==========================================================
class BCModel(nn.Module):
    def __init__(self, input_dim=29, output_dim=3):
        super(BCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
            # Nessuna attivazione finale: lasciamo che il MSELoss adatti i valori al range corretto
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================================
# 4. FUNZIONE PRINCIPALE DI ADDESTRAMENTO
# ==========================================================
def main():
    data_dir = "train_set/laps"
    
    print("="*60)
    print(" 🧠 DATA PREPROCESSING E ADDESTRAMENTO (BEHAVIORAL CLONING) ")
    print("="*60)
    
    # ── 1. Cerca i file HDF5 ──
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
    if not h5_files:
        print(f"ERRORE: Nessun file .h5 trovato nella cartella {data_dir}.")
        print("Assicurati di aver guidato qualche giro valido con data_collection.py!")
        return
        
    print(f"Trovati {len(h5_files)} file .h5. Caricamento in corso...")
    
    all_states = []
    all_actions = []
    
    # ── 2. Carica i Dati ──
    for f_path in h5_files:
        try:
            with h5py.File(f_path, 'r') as h5f:
                s = np.array(h5f['states'])
                a = np.array(h5f['actions'])
                all_states.append(s)
                all_actions.append(a)
        except Exception as e:
            print(f"Errore caricando {f_path}: {e}")
            
    # Unisci tutti i giri in un unico grande array
    X_raw = np.concatenate(all_states, axis=0)
    Y_raw = np.concatenate(all_actions, axis=0)
    
    # Rimuoviamo la Marcia (gear) da Y. 
    # Y originale = [steer, accel, brake, gear]. Y nuovo = [steer, accel, brake]
    Y_raw = Y_raw[:, :3]
    
    print(f"Totale campioni caricati (step): {len(X_raw)}")
    print(f"Dimensioni Stati (X): {X_raw.shape}")
    print(f"Dimensioni Azioni (Y): {Y_raw.shape}")
    
    # ── 3. Normalizzazione ──
    print("\n[+] Normalizzazione dei sensori...")
    normalizer = Normalizer()
    normalizer.fit(X_raw)
    X_norm = normalizer.transform(X_raw)
    
    normalizer.save("bc_scaler.json")
    
    # ── 4. Creazione Dataset e DataLoader ──
    # Mescola e crea batch da 64 campioni
    dataset = TorcsDataset(X_norm, Y_raw)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # ── 5. Inizializzazione Rete e Ottimizzatore ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Addestramento su dispositivo: {device}")
    
    model = BCModel(input_dim=29, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # ── 6. Ciclo di Addestramento (Epoche) ──
    epochs = 30
    print("\n[+] Inizio Addestramento della Rete Neurale...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (states, actions) in enumerate(dataloader):
            states, actions = states.to(device), actions.to(device)
            
            # Forward pass (predizione)
            predictions = model(states)
            
            # Calcolo dell'errore (Loss)
            loss = criterion(predictions, actions)
            
            # Backward pass (Correzione dei pesi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoca [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}")
        
    # ── 7. Salvataggio del Modello ──
    torch.save(model.state_dict(), "bc_model.pth")
    print("\n[+] ADDESTRAMENTO COMPLETATO!")
    print("  Il modello è stato salvato in 'bc_model.pth'")
    print("  Ora l'Agente è pronto per guidare da solo!")

if __name__ == "__main__":
    main()
