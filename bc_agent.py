import os
import sys
import time
import json
import numpy as np
import torch
import snakeoil3_gym as snakeoil3

# Rimosso l'import di Normalizer e BCModel da train_bc poiché il modello JIT è autonomo
from data_collection import flatten_state, apply_tcs
from gearing import compute_gear

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("ERRORE: Impossibile importare gym_torcs.")
    sys.exit(1)

def main():
    print("="*60)
    print(" 🤖 AGENTE AUTONOMO - TEST SOTTOSTERZO (Ottimizzato CPU - JIT) ")
    print("="*60)

    # Il modello JIT End-to-End ora racchiude al suo interno Scaler e PCA
    model_path = "models/torcs_driver_jit.pt"
    if not os.path.exists(model_path):
        print(f"ERRORE: Modello JIT non trovato in '{model_path}'. Esegui prima lo script di training.")
        return
        
    # Forza l'esecuzione su CPU come concordato per il training
    device = torch.device("cpu")
    print(f"[+] Caricamento del modello JIT End-to-End su {device}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval() # Fondamentale per disattivare BatchNorm e Dropout in fase di test

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    env.reset(relaunch=True)
    
    current_gear = 1
    steps_since_shift = 0
    prev_steer = 0.0

    print("\n🏎️ L'AGENTE STA GUIDANDO CON IL NUOVO MODELLO! Premi Ctrl+C per fermare.")
    
    try:
        while True:
            raw_ob = env.client.S.d
            
            # 1. Estrazione del vettore di stato GREZZO (29 feature)
            state_vec = flatten_state(raw_ob)
            
            # Trasformazione in tensore PyTorch [1, 29] senza applicare lo scaler a mano
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            
            # 2. Predizione End-to-End (Normalizzazione, PCA e ResNet avvengono dentro il file .pt)
            with torch.no_grad():
                pred = model(state_tensor).numpy()[0]
                
# L'output contiene esattamente i 3 comandi continui principali
            raw_steer = pred[0]
            raw_accel = pred[1]
            raw_brake = pred[2]
            
            # Lettura telemetria dal gioco (Spostata più in alto per poterla usare subito)
            spd = float(np.array(raw_ob.get('speedX', 0.0)).flat[0])
            rpm = float(np.array(raw_ob.get('rpm', 0.0)).flat[0])

            # ==========================================
            # 1. DEADZONE FRENO (Ignora le micro-frenate sotto il 15%)
            # ==========================================
            if raw_brake < 0.15:
                raw_brake = 0.0
            
            # ==========================================
            # 2. ANTI-STALLO E RECUPERO PANICO
            # ==========================================
            # Se l'auto sta andando a meno di 15 km/h (quasi ferma)
            if spd < 20.0:
                raw_brake = 0.0                # Rilascia il freno forzatamente
                raw_accel = max(raw_accel, 0.65) # Dai almeno il 50% di gas per ripartire

            # ==========================================
            # 3. FILTRO VOLANTE (Anti-ZigZag)
            # ==========================================
            # Riduciamo il moltiplicatore (da 1.4 a 1.2) e abbassiamo alpha (da 0.3 a 0.15)
            # Questo rende il volante molto più fluido e "pigro", assorbendo le vibrazioni.
            raw_steer = raw_steer * 1.2 
            alpha_steer = 0.15 
            steer = (alpha_steer * raw_steer) + ((1.0 - alpha_steer) * prev_steer)
            
            # ==========================================
            # 4. GUARDRAIL VIRTUALE (Anti-Panico)
            # ==========================================
            # trackPos va da -1.0 (bordo destro) a +1.0 (bordo sinistro). 0.0 è il centro.
            track_pos = float(np.array(raw_ob.get('trackPos', 0.0)).flat[0])
            
            # Se la rete la sta mandando troppo vicina all'erba, forziamo la correzione
            '''if track_pos > 0.80:  # Pericolo a Sinistra!
                steer = min(steer, -0.2)  # Forza una sterzata verso Destra
            elif track_pos < -0.80: # Pericolo a Destra!
                steer = max(steer, 0.2)   # Forza una sterzata verso Sinistra'''

            prev_steer = steer
            steer = max(-1.0, min(1.0, steer))
            accel = max(0.0, min(1.0, raw_accel))
            brake = max(0.0, min(1.0, raw_brake))
            
            # Applica freno
            accel = accel * (1.0 - brake)
            
            # Stampa a schermo la telemetria per debugging
            print(f"Velocità: {spd:5.1f} km/h | Freno Rete: {raw_brake:.2f} | Marcia: {current_gear}", end='\r')
            
            # ==========================================
            # LIMITATORE DI VELOCITÀ RIGIDO
            # ==========================================
            SPEED_LIMIT = 80.0
            if spd > SPEED_LIMIT:
                accel = 0.0 # Taglia il gas
                if spd > SPEED_LIMIT + 5.0:
                    brake = max(brake, 0.4) 
            # ==========================================
            
            # 3. Gestione DETERMINISTICA del cambio marcia (Invariato)
            current_gear, shifted = compute_gear(spd, accel, rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if shifted else steps_since_shift + 1
            
            action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
            action = apply_tcs(action, raw_ob, slip_threshold=5.0)
            
            # 4. Invia comandi al simulatore TORCS
            env.client.R.d['steer'] = action[0]
            env.client.R.d['accel'] = action[1]
            env.client.R.d['brake'] = action[2]
            env.client.R.d['gear']  = int(action[3])
            
            env.client.respond_to_server()
            env.client.get_servers_input()
            
            if env.client.R.d.get('meta', 0) == 1:
                print("\n\n[!] Traguardo raggiunto o fuori pista. Riavvio...")
                env.reset(relaunch=False)
                current_gear = 1
                steps_since_shift = 0
                prev_steer = 0.0 
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta.")
    finally:
        env.end()

if __name__ == "__main__":
    main()