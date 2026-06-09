"""
Training Pipeline — KNN Regressor per TORCS (Corkscrew Optimized)

1. Carica tutti i file .h5 validi dalla cartella dei log.
2. Sfoltisce lo spazio delle feature da 29D a 22D (solo geometria + velocità X).
3. Separa i dati in Training e Test set per validare la precisione.
4. Applica StandardScaler (Z-Score) per uniformare le scale metriche/radianti.
5. Applica un vettore di Custom Weights per dare priorità a velocità, traiettoria e specchio frontale.
6. Addestra un KNeighborsRegressor multi-output (Steer, Accel, Brake).
7. Salva il modello, lo scaler e i pesi per la fase di guida (Inference).
"""

import os
import glob
import h5py
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(laps_dir: str):
    """Carica e unisce tutti i file HDF5 presenti nella cartella."""
    h5_files = glob.glob(os.path.join(laps_dir, "*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"Nessun file .h5 trovato in {laps_dir}. Esegui prima la data collection!")
    
    print(f"📦 Trovati {len(h5_files)} file di giri completi/segmenti. Caricamento in corso...")
    
    all_states = []
    all_actions = []
    
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as h5f:
            states = h5f['states'][:]
            actions = h5f['actions'][:]
            all_states.append(states)
            all_actions.append(actions)
            
    return np.concatenate(all_states, axis=0), np.concatenate(all_actions, axis=0)

def main():
    LAPS_DIR = os.path.join("train_set", "laps")
    MODEL_OUTPUT = "knn_corkscrew_model.pkl"
    SCALER_OUTPUT = "knn_scaler.pkl"
    WEIGHTS_OUTPUT = "knn_custom_weights.npy"
    
    # 1. Caricamento Dati
    try:
        X_raw, Y_raw = load_dataset(LAPS_DIR)
    except Exception as e:
        print(f"❌ Errore durante il caricamento: {e}")
        return
        
    print(f"📊 Dataset grezzo caricato con successo. Righe totali (step): {X_raw.shape[0]}")
    
    # 2. Selezione Feature (Riduzione da 29D a 22D)
    # Teniamo: index 0 (angle), 1..19 (track), 20 (trackPos), 21 (speedX)
    X_22d = X_raw[:, :22]
    
    # Isoliamo i target (0: steer, 1: accel, 2: brake). 
    # Escludiamo l'indice 3 (marcia) perché la marcia è intercettata da gearing.py
    Y_3d = Y_raw[:, :3]
    
    # 3. Suddivisione Train / Test (80% / 20%) per validazione interna
    X_train, X_test, Y_train, Y_test = train_test_split(X_22d, Y_3d, test_size=0.2, random_state=42)
    print(f"✂️  Dataset suddiviso: {X_train.shape[0]} step di train, {X_test.shape[0]} step di test.")
    
    # 4. Standardizzazione (Z-score Normalization)
    print("⚖️  Calcolo e applicazione dello StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Configurazione e Applicazione del Feature Weighting (Ottimizzazione Corkscrew)
    print("🎯 Applicazione dei pesi personalizzati alle feature...")
    custom_weights = np.ones(22, dtype=np.float32)
    
    custom_weights[0] = 3.0       # ANGOLO: Cruciale per evitare micro-oscillazioni e mantenere il centro
    custom_weights[8:13] = 2.0    # LASER CENTRALI (Indici track centrali): Più importanza a cosa c'è dritto davanti
    custom_weights[20] = 2.5      # TRACKPOS: Forza il KNN a capire la sponda corretta in ingresso curva
    custom_weights[21] = 5.0      # SPEEDX: Peso massimo. Fondamentale per far capire al KNN quando staccare sul Corkscrew!
    
    # Moltiplicazione BroadCast dei pesi sulle matrici scalate
    X_train_weighted = X_train_scaled * custom_weights
    X_test_weighted = X_test_scaled * custom_weights
    
    # 6. Configurazione e Addestramento dell'Algoritmo KNN
    print("🧠 Addestramento del KNeighborsRegressor (K=5, weights='distance')...")
    # n_jobs=-1 sfrutta tutti i core della CPU per velocizzare la ricerca dei vicini
    knn_agent = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_agent.fit(X_train_weighted, Y_train)
    
    print("✅ Addestramento completato!")
    
    # 7. Validazione del Modello sul Test Set
    Y_pred = knn_agent.predict(X_test_weighted)
    
    print("\n📈 --- METRICHE DI VALIDAZIONE ---")
    actions_labels = ["STEER (Sterzo)", "ACCEL (Gas)", "BRAKE (Freno)"]
    for i, label in enumerate(actions_labels):
        mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"     {label:15} -> MSE: {mse:.5f} | R² Score: {r2:7.2%}")
    print("----------------------------------\n")
    
    # 8. Salvataggio degli artefatti sul disco
    print(f"💾 Salvataggio dei file sul disco...")
    joblib.dump(knn_agent, MODEL_OUTPUT, compress=3)
    joblib.dump(scaler, SCALER_OUTPUT, compress=3)
    np.save(WEIGHTS_OUTPUT, custom_weights)
    
    print(f"✨ Successo! File pronti per l'Inference Agent:")
    print(f"   - Modello: {MODEL_OUTPUT}")
    print(f"   - Scaler:  {SCALER_OUTPUT}")
    print(f"   - Pesi:    {WEIGHTS_OUTPUT}")

if __name__ == "__main__":
    main()