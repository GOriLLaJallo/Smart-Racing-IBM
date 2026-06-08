import h5py
import numpy as np
import glob

# Cerca il primo file .h5 disponibile
files = glob.glob("train_set/laps/*.h5")
if not files:
    print("Nessun file trovato!")
    exit()

file_path = files[0]
print(f"Ispezionando il file: {file_path}\n")

with h5py.File(file_path, 'r') as f:
    states = np.array(f['states'])
    actions = np.array(f['actions'])
    
    print(f"Dimensioni (Shape):")
    print(f" - Sensori (Stati): {states.shape} -> Dev'essere (Numero_Frame, 29)")
    print(f" - Comandi (Azioni): {actions.shape} -> Dev'essere (Numero_Frame, X)\n")
    
    # Prendiamo un frame a metà del giro (dove l'auto sta andando veloce)
    mid_idx = len(states) // 2
    
    print(f"--- FRAME {mid_idx} ---")
    print("AZIONI DEL PILOTA (Sterzo, Gas, Freno):")
    print(np.round(actions[mid_idx][:3], 3))
    
    print("\nSENSORI GREZZI (I 29 valori):")
    print(" [ 0] Angolo rispetto alla pista :", round(states[mid_idx][0], 3))
    print(" [ 1] Velocità X (Longitudinale) :", round(states[mid_idx][1], 3))
    print(" [ 2] Velocità Y (Laterale)      :", round(states[mid_idx][2], 3))
    print(" [ 3] Velocità Z (Verticale)     :", round(states[mid_idx][3], 3))
    print(" [ 4] Giri Motore (RPM)          :", round(states[mid_idx][4], 3))
    print(" [5-23] Sensori Laser Pista      :", np.round(states[mid_idx][5:24], 1))
    print(" [24-28] Altri sensori (Ruote/ecc):", np.round(states[mid_idx][24:], 2))