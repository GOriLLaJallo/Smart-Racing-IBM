"""
Training Pipeline — KNN Regressor per TORCS (Corkscrew Optimized)

Questo script si occupa di addestrare il "cervello" del pilota automatico basato 
sull'algoritmo K-Nearest Neighbors (KNN). 
A differenza delle Reti Neurali classiche che cercano di approssimare una funzione matematica complessa, 
il KNN "memorizza" l'intero dataset di guida umana. Durante la gara (Inference), il modello confronterà 
i dati dei sensori in tempo reale con il suo database, cercherà i 5 (k) momenti (neighbors) matematicamente 
più simili e ne farà una media per decidere come sterzare e accelerare.

Fasi principali del flusso di lavoro:
1. Carica tutti i file .h5 validi dalla cartella dei log.
2. Sfoltisce lo spazio delle feature da 29 dimensioni (29D) a 22 (22D), mantenendo solo geometria della pista e velocità longitudinale.
3. Separa i dati in Training e Test set per validare la precisione a fine addestramento.
4. Applica uno StandardScaler (Z-Score) per uniformare le diverse scale di misura (es. radianti vs km/h vs metri).
5. Applica un vettore di Pesi Personalizzati (Custom Weights) per alterare artificialmente la percezione delle distanze del KNN, dando priorità assoluta alla velocità e alla traiettoria.
6. Addestra un KNeighborsRegressor multi-output (che prevede in contemporanea Sterzo, Acceleratore e Freno).
7. Salva il modello, lo scaler e i pesi su disco per la fase di guida (Inference).
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
    """
    Cerca, carica e concatena tutti i file HDF5 presenti nella cartella dei giri salvati.
    
    Args:
        laps_dir (str): Il percorso alla cartella contenente i file .h5.
        
    Returns:
        tuple: (Array degli stati concatenati, Array delle azioni concatenate).
    """
    h5_files = glob.glob(os.path.join(laps_dir, "*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"Nessun file .h5 trovato in {laps_dir}. Esegui prima la data collection!")
    
    print(f"📦 Trovati {len(h5_files)} file di giri completi/segmenti. Caricamento in corso...")
    
    all_states = []
    all_actions = []
    
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as h5f:
            # Estrae i tensori dei dati dai file .h5 creati dalla fase di Data Collection
            states = h5f['states'][:]
            actions = h5f['actions'][:]
            all_states.append(states)
            all_actions.append(actions)
            
    # Unisce tutte le liste di array in due enormi matrici (X per gli input, Y per gli output desiderati)
    return np.concatenate(all_states, axis=0), np.concatenate(all_actions, axis=0)

def main():
    # Definizioni dei percorsi di input e output
    LAPS_DIR = os.path.join("train_set", "laps")
    MODEL_OUTPUT = "knn_corkscrew_model.pkl"
    SCALER_OUTPUT = "knn_scaler.pkl"
    WEIGHTS_OUTPUT = "knn_custom_weights.npy"
    
    # =========================================================================
    # 1. CARICAMENTO DATI
    # =========================================================================
    try:
        X_raw, Y_raw = load_dataset(LAPS_DIR)
    except Exception as e:
        print(f"❌ Errore durante il caricamento: {e}")
        return
        
    print(f"📊 Dataset grezzo caricato con successo. Righe totali (step): {X_raw.shape[0]}")
    
    # =========================================================================
    # 2. SELEZIONE E RIDUZIONE DELLE FEATURE (FEATURE ENGINEERING)
    # =========================================================================
    # Il dataset grezzo aveva 29 variabili, incluse velocità delle ruote e RPM.
    # Tagliamo le ultime 7 colonne tenendo solo le prime 22:
    # - Indice 0: angle (Angolo della vettura rispetto all'asse della pista)
    # - Indici 1..19: track (I 19 telemetri laser che misurano la distanza dai bordi pista)
    # - Indice 20: trackPos (Posizione dell'auto da -1 a 1, dove 0 è il centro esatto)
    # - Indice 21: speedX (Velocità longitudinale)
    X_22d = X_raw[:, :22]
    
    # Per il target (ciò che vogliamo prevedere), teniamo solo i primi 3 valori (Sterzo, Acceleratore, Freno).
    # L'indice 3 originale (la marcia) viene escluso in quanto il cambio marcia è gestito esternamente dallo script deterministico gearing.py.
    Y_3d = Y_raw[:, :3]
    
    # =========================================================================
    # 3. SPLIT DEL DATASET (TRAINING / TEST)
    # =========================================================================
    # Prendiamo l'80% dei dati per "insegnare" al modello e teniamo segreto il restante 20%.
    # Useremo questo 20% alla fine per interrogare il modello e vedere se ha imparato davvero o ha solo imparato a memoria (overfitting).
    X_train, X_test, Y_train, Y_test = train_test_split(X_22d, Y_3d, test_size=0.2, random_state=42)
    print(f"✂️  Dataset suddiviso: {X_train.shape[0]} step di train, {X_test.shape[0]} step di test.")
    
    # =========================================================================
    # 4. STANDARDIZZAZIONE (Z-SCORE)
    # =========================================================================
    # Il KNN si basa sulla misurazione della "distanza" geometrica (es. Distanza Euclidea) tra due situazioni.
    # Se la speedX varia da 0 a 300, e il trackPos varia da -1 a 1, la speedX dominerebbe completamente la formula matematica.
    # Lo StandardScaler trasforma tutte le variabili per avere media 0 e deviazione standard 1, mettendole ad armi pari.
    print("⚖️  Calcolo e applicazione dello StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # 5. FEATURE WEIGHTING (IL CUORE DELL'OTTIMIZZAZIONE PER IL CORKSCREW)
    # =========================================================================
    # Dopo aver appiattito tutte le variabili con lo Scaler, alteriamo intenzionalmente i pesi.
    # Questo serve a "dire" all'algoritmo KNN a cosa prestare più attenzione quando cerca i momenti passati simili.
    print("🎯 Applicazione dei pesi personalizzati alle feature...")
    custom_weights = np.ones(22, dtype=np.float32)
    
    # Moltiplicatore 3.0 all'angolo dell'auto. Aiuta a mantenere il veicolo dritto evitando lo "swaying" (oscillazioni) sul rettilineo.
    custom_weights[0] = 3.0       
    # Moltiplicatore 2.0 sui laser centrali (da 8 a 12). Dice al modello che "cosa c'è davanti" è più importante di "cosa c'è di lato".
    custom_weights[8:13] = 2.0    
    # Moltiplicatore 2.5 alla posizione trasversale. Fondamentale per forzare il bot a impostare correttamente l'ingresso in curva sulle sponde giuste.
    custom_weights[20] = 2.5      
    # Moltiplicatore 5.0 (Ponderazione Massima) alla Velocità. 
    # Senza questo peso enorme, il bot frenerebbe troppo tardi nei punti critici come il Cavatappi (Corkscrew), scambiando una staccata a 250km/h per una a 100km/h.
    custom_weights[21] = 5.0      
    
    # Applichiamo matematicamente i pesi moltiplicando l'array per le matrici precedentemente scalate.
    X_train_weighted = X_train_scaled * custom_weights
    X_test_weighted = X_test_scaled * custom_weights
    
    # =========================================================================
    # 6. ADDESTRAMENTO DEL MODELLO K-NEAREST NEIGHBORS
    # =========================================================================
    print("🧠 Addestramento del KNeighborsRegressor (K=5, weights='distance')...")
    # Impostazioni chiave:
    # - n_neighbors=5: Prenderà le 5 situazioni più simili passate per prendere una decisione.
    # - weights='distance': Dà più importanza ai vicini matematicamente più prossimi rispetto a quelli più lontani.
    # - n_jobs=-1: Usa tutti i processori disponibili nel computer per accelerare il tempo di ricerca (che nel KNN è oneroso).
    knn_agent = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
    knn_agent.fit(X_train_weighted, Y_train)
    
    print("✅ Addestramento completato!")
    
    # =========================================================================
    # 7. VALIDAZIONE E METRICHE
    # =========================================================================
    # Passiamo al modello le domande segrete del Test Set di cui non ha mai visto le risposte.
    Y_pred = knn_agent.predict(X_test_weighted)
    
    print("\n📈 --- METRICHE DI VALIDAZIONE ---")
    actions_labels = ["STEER (Sterzo)", "ACCEL (Gas)", "BRAKE (Freno)"]
    for i, label in enumerate(actions_labels):
        # Il Mean Squared Error (MSE) calcola l'errore quadratico medio. Più è vicino a 0, meglio è.
        mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
        # R² (Coefficiente di Determinazione) valuta la precisione predittiva percentuale. 
        # Valori sopra l'85-90% sono considerati eccellenti.
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"     {label:15} -> MSE: {mse:.5f} | R² Score: {r2:7.2%}")
    print("----------------------------------\n")
    
    # =========================================================================
    # 8. ESPORTAZIONE DEGLI ARTEFATTI
    # =========================================================================
    # Salviamo l'intero cervello, lo standardizzatore e i pesi. 
    # Lo script di gara dovrà caricarli esattamente in quest'ordine per poter guidare l'auto.
    print(f"💾 Salvataggio dei file sul disco...")
    # compress=3 applica una leggera compressione zlib ai modelli per risparmiare RAM e spazio su disco
    joblib.dump(knn_agent, MODEL_OUTPUT, compress=3)
    joblib.dump(scaler, SCALER_OUTPUT, compress=3)
    np.save(WEIGHTS_OUTPUT, custom_weights)
    
    print(f"✨ Successo! File pronti per l'Inference Agent:")
    print(f"   - Modello: {MODEL_OUTPUT}")
    print(f"   - Scaler:  {SCALER_OUTPUT}")
    print(f"   - Pesi:    {WEIGHTS_OUTPUT}")

if __name__ == "__main__":
    main()