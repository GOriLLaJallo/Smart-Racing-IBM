"""
Inference Pipeline — KNN Autonomous Driving Agent per TORCS (Corkscrew Optimized)

Include il fix "Blind Crest Override": un freno di emergenza automatico 
sulla staccata cieca del Cavatappi per sopperire alla mancanza di visione dei laser.
"""

import os
import sys
import time
import numpy as np
import joblib

# Forza la visualizzazione della GUI di TORCS per vedere l'agente guidare
os.environ['SHOW_GUI'] = '1'

# Aggiungo gym_torcs al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"❌ ERRORE FATALE: Impossibile importare gym_torcs.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)

# Importiamo le funzioni di utilità riutilizzabili dai moduli originali
from data_collection import flatten_state, apply_tcs
from gearing import compute_gear

def main():
    MODEL_PATH = "knn_corkscrew_model.pkl"
    SCALER_PATH = "knn_scaler.pkl"
    WEIGHTS_PATH = "knn_custom_weights.npy"
    
    # 1. Verifica e caricamento degli artefatti del modello KNN
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(WEIGHTS_PATH)):
        print("❌ ERRORE: File del modello non trovati!")
        print("Assicurati di aver eseguito 'train_knn.py' prima di avviare l'agente.")
        return

    print("🧠 Caricamento del cervello KNN e delle pipeline di normalizzazione...")
    knn_agent = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    custom_weights = np.load(WEIGHTS_PATH)
    print("✅ Modello caricato correttamente.")

    # 2. Inizializzazione dell'ambiente TORCS
    print("🏎️  Connessione a TORCS Environment...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    
    lap_count = 0

    try:
        while True:
            lap_count += 1
            print(f"\n🚀 [Giro {lap_count}] Inizializzazione e reset della pista...")
            
            env.reset(relaunch=(lap_count == 1))
            ob = env.client.S.d
            
            current_gear = 1
            steps_since_shift = 0
            step = 0
            
            print("🟢 Agente KNN in controllo del veicolo. Guida autonoma attiva!")
            
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
                    steer = float(predicted_actions[0])  # Mantieni lo sterzo del KNN per preparare la curva
                    accel = 0.0                          # Forza il rilascio del gas
                    brake = 0.8                          # Forza una frenata decisa all'80%
                else:
                    # Comportamento normale guidato dal KNN nel resto del tracciato
                    steer = float(predicted_actions[0])
                    accel = float(predicted_actions[1])
                    brake = float(predicted_actions[2])
                # ---------------------------------------------------------

                action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
                
                # E) Filtro TCS (Traction Control)
                action = apply_tcs(action, ob, slip_threshold=5.0)
                
                # F) Cambio Automatico Deterministico
                speed_x_raw = ob.get('speedX', 0.0)
                speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                speed_kmh = speed_x_val * 50.0 
                
                rpm_raw = ob.get('rpm', 0.0)
                rpm_val = float(rpm_raw.flat[0]) if isinstance(rpm_raw, np.ndarray) else float(rpm_raw)
                
                new_gear, shifted = compute_gear(
                    speed_kmh=speed_kmh,
                    accel=action[1],
                    rpm=rpm_val,
                    current_gear=current_gear,
                    steps_since_shift=steps_since_shift
                )
                
                if shifted:
                    current_gear = new_gear
                    steps_since_shift = 0
                else:
                    steps_since_shift += 1
                
                action[3] = float(current_gear)
                
                # G) Invio comandi a TORCS
                env.client.R.d['steer'] = action[0]
                env.client.R.d['accel'] = action[1]
                env.client.R.d['brake'] = action[2]
                env.client.R.d['gear'] = int(action[3])
                
                env.client.respond_to_server()
                env.client.get_servers_input()
                
                ob_next = env.client.S.d
                done = env.client.R.d.get('meta', 0) == 1
                
                # H) Sicurezza Fuoripista
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]
                
                if abs(current_track_pos) > 1.5:
                    print(f"\n⚠️ [AGENTE OFF-TRACK] Auto uscita di pista (trackPos: {current_track_pos:.2f}). Riposizionamento...")
                    break 
                
                # Feedback Telemetria
                if step % 50 == 0:
                    status_freno = "!! OVERRIDE FRENO !!" if 1410.0 <= dist_curr <= 1460.0 else ""
                    print(f"   [Step {step:4d} | {dist_curr:6.1f}m] Vel: {speed_kmh:5.1f} | Marcia: {current_gear} | RPM: {rpm_val:5.0f} {status_freno}", end='\r')
                
                ob = ob_next
                
                if done:
                    print("\n🏁 Sessione conclusa dal server.")
                    break

    except KeyboardInterrupt:
        print("\n\n🛑 GUIDA AUTONOMA INTERROTTA MANUALE (Ctrl+C). Uscita...")
    finally:
        env.end()

if __name__ == "__main__":
    main()