"""
Inference Pipeline — KNN Autonomous Driving Agent per TORCS (Corkscrew Optimized)

<<<<<<< HEAD
Include il fix "Blind Crest Override": un freno di emergenza automatico 
sulla staccata cieca del Cavatappi per sopperire alla mancanza di visione dei laser.
=======
Questo script rappresenta il "pilota".
Prende i file generati dalla fase di addestramento (modello KNN, Scaler e Pesi) e li 
utilizza per guidare autonomamente l'auto all'interno del simulatore TORCS.

Include il fix speciale "Blind Crest Override": un freno di emergenza automatico 
programmato sulla staccata cieca del Cavatappi (Corkscrew). Questo è necessario perché i 
sensori laser del gioco puntano dritti: in presenza di un dosso cieco, leggono il cielo 
(distanza infinita) invece della curva, ingannando l'algoritmo.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
"""

import os
import sys
import time
import numpy as np
import joblib

<<<<<<< HEAD
# Forza la visualizzazione della GUI di TORCS per vedere l'agente guidare
os.environ['SHOW_GUI'] = '1'

# Aggiungo gym_torcs al path
=======
# ==============================================================================
# SETUP E CONFIGURAZIONE AMBIENTE
# ==============================================================================

# Forza la visualizzazione della GUI di TORCS per vedere fisicamente l'agente guidare.
os.environ['SHOW_GUI'] = '1'

# Aggiunge la cartella gym_torcs al path di sistema per consentire l'importazione del simulatore.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"❌ ERRORE FATALE: Impossibile importare gym_torcs.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)

<<<<<<< HEAD
# Importiamo le funzioni di utilità riutilizzabili dai moduli originali
=======
# Importa le funzioni di utilità già scritte e validate nei moduli precedenti:
# - flatten_state: per appiattire e normalizzare la telemetria grezza.
# - apply_tcs: per il controllo di trazione in uscita di curva.
# - compute_gear: per la gestione del cambio anti-hunting.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
from data_collection import flatten_state, apply_tcs
from gearing import compute_gear

def main():
<<<<<<< HEAD
=======
    # ==============================================================================
    # 1. VERIFICA E CARICAMENTO DEGLI ARTEFATTI
    # ==============================================================================
    # Definisce i percorsi dei file generati dallo script train_knn.py.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
    MODEL_PATH = "knn_corkscrew_model.pkl"
    SCALER_PATH = "knn_scaler.pkl"
    WEIGHTS_PATH = "knn_custom_weights.npy"
    
<<<<<<< HEAD
    # 1. Verifica e caricamento degli artefatti del modello KNN
=======
    # Controllo di sicurezza: se manca uno dei "pezzi di cervello", il pilota non può guidare.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(WEIGHTS_PATH)):
        print("❌ ERRORE: File del modello non trovati!")
        print("Assicurati di aver eseguito 'train_knn.py' prima di avviare l'agente.")
        return

    print("🧠 Caricamento del cervello KNN e delle pipeline di normalizzazione...")
<<<<<<< HEAD
=======
    # Ricostruisce in memoria gli oggetti scikit-learn e gli array numpy.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
    knn_agent = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    custom_weights = np.load(WEIGHTS_PATH)
    print("✅ Modello caricato correttamente.")

<<<<<<< HEAD
    # 2. Inizializzazione dell'ambiente TORCS
    print("🏎️  Connessione a TORCS Environment...")
=======
    # ==============================================================================
    # 2. INIZIALIZZAZIONE DEL SIMULATORE TORCS
    # ==============================================================================
    print("🏎️  Connessione a TORCS Environment...")
    # Avvia l'ambiente indicando che vogliamo controllare gas e marce, ma non usiamo la visione ottica a pixel.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    
    lap_count = 0

    try:
<<<<<<< HEAD
=======
        # Loop esterno: gestisce le intere sessioni/giri.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
        while True:
            lap_count += 1
            print(f"\n🚀 [Giro {lap_count}] Inizializzazione e reset della pista...")
            
<<<<<<< HEAD
            env.reset(relaunch=(lap_count == 1))
            ob = env.client.S.d
            
=======
            # Reset dell'ambiente. Riavvia fisicamente il motore C++ di TORCS solo al primo giro per pulizia.
            env.reset(relaunch=(lap_count == 1))
            ob = env.client.S.d
            
            # Variabili di stato iniziali per la vettura.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
            current_gear = 1
            steps_since_shift = 0
            step = 0
            
            print("🟢 Agente KNN in controllo del veicolo. Guida autonoma attiva!")
<<<<<<< HEAD
            last_steer = 0.0  # Variabile per il filtro passa-basso dello sterzo
            while True:
                step += 1
                
                # A) Flatten dello stato corrente a 29D
                full_state = flatten_state(ob)
                
                # B) Taglio geometrico (22 feature) e normalizzazione
                state_22d = full_state[:22].reshape(1, -1)
                state_scaled = scaler.transform(state_22d)
                state_weighted = state_scaled * custom_weights
                
                # C) Predizione pura del modello KNN
                predicted_actions = knn_agent.predict(state_weighted)[0]
                
                # ---------------------------------------------------------
                # D) FIX "BLIND CREST": OVERRIDE PER LA STACCATA CAVATAPPI
                # ---------------------------------------------------------
                dist_curr_raw = ob.get('distFromStart', 0.0)
                dist_curr = float(dist_curr_raw.flat[0] if isinstance(dist_curr_raw, np.ndarray) else dist_curr_raw)
                
                # Modifica questi due valori in base alla telemetria per anticipare/posticipare la frenata
                if 1410.0 <= dist_curr <= 1460.0:
                    raw_steer = float(predicted_actions[0])  # Mantieni lo sterzo del KNN per preparare la curva
                    # Applica il filtro passa-basso: 70% nuova decisione, 30% inerzia precedente
                    steer = (raw_steer * 0.8) + (last_steer * 0.2)
                    last_steer = steer  # Salva in memoria per il frame successivo
                    accel = 0.0                          # Forza il rilascio del gas
                    brake = 0.8                          # Forza una frenata decisa all'80%
                else:
                    # Comportamento normale guidato dal KNN nel resto del tracciato
                    raw_steer = float(predicted_actions[0])
                    # Applica il filtro passa-basso: 70% nuova decisione, 30% inerzia precedente
                    steer = (raw_steer * 0.8) + (last_steer * 0.2)
                    last_steer = steer  # Salva in memoria per il frame successivo
                    accel = float(predicted_actions[1])
                    brake = float(predicted_actions[2])
                # ---------------------------------------------------------
                
                # ---------------------------------------------------------
                # D.2) GUARDRAIL VIRTUALE (Safety Override)
                # ---------------------------------------------------------
                # Estraiamo la posizione attuale (0 = centro, 1 = sinistra, -1 = destra)
                track_pos_raw = ob.get('trackPos', 0.0)
                track_pos = float(track_pos_raw.flat[0] if isinstance(track_pos_raw, np.ndarray) else track_pos_raw)
                
                SOGLIA_PERICOLO = 1  # Modifica questo valore (1.0 è l'erba)
                
                if abs(track_pos) > SOGLIA_PERICOLO:
                    # Calcola una sterzata correttiva verso il centro. 
                    # Il segno meno inverte la direzione: se siamo a +0.9 (sinistra), sterza a - (destra).
                    # Il moltiplicatore 1.2 decide la violenza della sterzata.
                    steer = -track_pos * 0.2 
                    
                    # Assicuriamoci che lo sterzo non superi i limiti fisici di TORCS (-1.0, +1.0)
                    steer = max(-1.0, min(1.0, steer))
                    
                    # Alziamo il piede dal gas per non uscire a velocità folli
                    accel = min(accel, 0.4) 
                    
                    # Stampiamo un avviso a schermo senza inondare il terminale
                    if step % 5 == 0:
                        print(f" 🛡️ GUARDRAIL ATTIVO! Pos: {track_pos:.2f} -> Correzione: {steer:.2f} ", end='\r')
                    
                action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
                
                # E) Filtro TCS (Traction Control)
                action = apply_tcs(action, ob, slip_threshold=5.0)
                
                # F) Cambio Automatico Deterministico
                speed_x_raw = ob.get('speedX', 0.0)
                speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                speed_kmh = speed_x_val 
=======
            
            # ==============================================================================
            # 3. LOOP IN TEMPO REALE (TICK DI GARA)
            # ==============================================================================
            while True:
                step += 1
                
                # --- A) Estrazione e Preparazione Dati ---
                # Estrae il dizionario di TORCS e lo appiattisce a un array di 29 valori.
                full_state = flatten_state(ob)
                
                # --- B) Allineamento con il Training ---
                # Il modello è stato addestrato su 22 dimensioni. Tagliamo le eccedenze e rimodelliamo 
                # a matrice (1 riga, N colonne) come richiesto da scikit-learn.
                state_22d = full_state[:22].reshape(1, -1)
                # Normalizzazione Z-Score e applicazione dei pesi per il Cavatappi.
                state_scaled = scaler.transform(state_22d)
                state_weighted = state_scaled * custom_weights
                
                # --- C) Predizione (L'Agente Pensa) ---
                # Il KNN cerca nel suo database i 5 istanti passati più simili a state_weighted
                # e calcola la media ponderata per sterzo, acceleratore e freno.
                predicted_actions = knn_agent.predict(state_weighted)[0]
                
                # --- D) FIX "BLIND CREST": OVERRIDE PER LA STACCATA CAVATAPPI ---
                # Ricava a quanti metri dalla linea di partenza ci troviamo.
                dist_curr_raw = ob.get('distFromStart', 0.0)
                dist_curr = float(dist_curr_raw.flat[0] if isinstance(dist_curr_raw, np.ndarray) else dist_curr_raw)
                
                # Zona di "cecità" sensoriale: tra i 1410 e i 1460 metri.
                if 1410.0 <= dist_curr <= 1460.0:
                    steer = float(predicted_actions[0])  # Mantieni lo sterzo del KNN per preparare la curva.
                    accel = 0.0                          # OVERRIDE: Forza il rilascio totale del gas.
                    brake = 0.8                          # OVERRIDE: Forza una frenata di emergenza all'80%.
                else:
                    # Fuori dalla zona cieca, usa le azioni predette dal KNN senza modifiche.
                    steer = float(predicted_actions[0])
                    accel = float(predicted_actions[1])
                    brake = float(predicted_actions[2])
                
                # Assembla l'array di azione parziale (la marcia è provvisoria).
                action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
                
                # --- E) Sistemi Elettronici di Bordo ---
                # Applica il Traction Control System per evitare testacoda.
                action = apply_tcs(action, ob, slip_threshold=5.0)
                
                # --- F) Cambio Automatico Deterministico ---
                # Estrae la velocità.
                speed_x_raw = ob.get('speedX', 0.0)
                speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                speed_kmh = speed_x_val * 1 #Abbiamo letto la velocità non filtrata da gymtorcs, quindi non divisa per 50 
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                
                rpm_raw = ob.get('rpm', 0.0)
                rpm_val = float(rpm_raw.flat[0]) if isinstance(rpm_raw, np.ndarray) else float(rpm_raw)
                
<<<<<<< HEAD
=======
                # Chiede al modulo gearing.py la marcia ideale.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                new_gear, shifted = compute_gear(
                    speed_kmh=speed_kmh,
                    accel=action[1],
                    rpm=rpm_val,
                    current_gear=current_gear,
                    steps_since_shift=steps_since_shift
                )
                
<<<<<<< HEAD
=======
                # Aggiorna i contatori del cambio per prevenire inceppamenti.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                if shifted:
                    current_gear = new_gear
                    steps_since_shift = 0
                else:
                    steps_since_shift += 1
                
                action[3] = float(current_gear)
                
<<<<<<< HEAD
                # G) Invio comandi a TORCS
=======
                # --- G) Attuazione nel Simulatore ---
                # Traduce l'array finale nel formato richiesto dal client di TORCS.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                env.client.R.d['steer'] = action[0]
                env.client.R.d['accel'] = action[1]
                env.client.R.d['brake'] = action[2]
                env.client.R.d['gear'] = int(action[3])
                
<<<<<<< HEAD
=======
                # Invia il pacchetto e aspetta la risposta con la nuova fisica del prossimo step.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                env.client.respond_to_server()
                env.client.get_servers_input()
                
                ob_next = env.client.S.d
<<<<<<< HEAD
                done = env.client.R.d.get('meta', 0) == 1
                
                # H) Sicurezza Fuoripista
=======
                # Controlla se la gara è finita lato server.
                done = env.client.R.d.get('meta', 0) == 1
                
                # --- H) Sicurezza e Crash Detection ---
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]
                
<<<<<<< HEAD
=======
                # Se l'auto esce completamente fuori dalla pista (> 1.5), interrompe per evitare logiche "bloccate nel muro".
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                if abs(current_track_pos) > 1.5:
                    print(f"\n⚠️ [AGENTE OFF-TRACK] Auto uscita di pista (trackPos: {current_track_pos:.2f}). Riposizionamento...")
                    break 
                
<<<<<<< HEAD
                # Feedback Telemetria
=======
                # --- Feedback a schermo ---
                # Stampiamo i dati principali ogni 50 step per non inondare la console.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                if step % 50 == 0:
                    status_freno = "!! OVERRIDE FRENO !!" if 1410.0 <= dist_curr <= 1460.0 else ""
                    print(f"   [Step {step:4d} | {dist_curr:6.1f}m] Vel: {speed_kmh:5.1f} | Marcia: {current_gear} | RPM: {rpm_val:5.0f} {status_freno}", end='\r')
                
<<<<<<< HEAD
=======
                # Prepara l'osservazione per il ciclo successivo.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
                ob = ob_next
                
                if done:
                    print("\n🏁 Sessione conclusa dal server.")
                    break

    except KeyboardInterrupt:
<<<<<<< HEAD
        print("\n\n🛑 GUIDA AUTONOMA INTERROTTA MANUALE (Ctrl+C). Uscita...")
    finally:
=======
        # Permette l'uscita pulita se l'utente preme Ctrl+C.
        print("\n\n🛑 GUIDA AUTONOMA INTERROTTA MANUALE (Ctrl+C). Uscita...")
    finally:
        # Chiude correttamente il socket di connessione.
>>>>>>> d2545a29ed93a188101e239960f069de3b8307ff
        env.end()

if __name__ == "__main__":
    main()