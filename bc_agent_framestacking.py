import os
import sys
import time
import json
import numpy as np
import torch
import snakeoil3_gym as snakeoil3
from collections import deque 

from data_collection import flatten_state, apply_tcs
from gearing import compute_gear

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("ERRORE: Impossibile importare gym_torcs.")
    sys.exit(1)

def main():
    print("="*60)
    print(" 🤖 AGENTE AUTONOMO STABILIZZATO (K=3, Input Dim = 87) ")
    print("="*60)

    model_path = "models/torcs_driver_jit.pt"
    if not os.path.exists(model_path):
        print(f"ERRORE: Modello JIT non trovato in '{model_path}'.")
        return
        
    device = torch.device("cpu")
    print(f"[+] Caricamento del modello JIT End-to-End...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval() 

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    env.reset(relaunch=True)
    
    current_gear = 1
    steps_since_shift = 0
    prev_steer = 0.0
    prev_brake = 0.0  # <--- NUOVO: Stato storico per la frenata progressiva

    state_history = deque(maxlen=3)

    print("\n🏎️ L'AGENTE STA GUIDANDO! Premi Ctrl+C per fermare.")
    
    try:
        while True:
            raw_ob = env.client.S.d
            state_vec = flatten_state(raw_ob)
            
            if len(state_history) == 0:
                state_history.append(state_vec)
                state_history.append(state_vec)
                state_history.append(state_vec)
            else:
                state_history.append(state_vec)
                
            stacked_state = np.concatenate(list(state_history))
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(state_tensor).numpy()[0]
                
            raw_steer = pred[0]
            raw_accel = pred[1]
            raw_brake = pred[2]
            
            spd = float(np.array(raw_ob.get('speedX', 0.0)).flat[0])
            rpm = float(np.array(raw_ob.get('rpm', 0.0)).flat[0])

            # ==========================================
            # 1. FILTRO FRENO PROGRESSIVO
            # ==========================================
            if raw_brake < 0.12:
                raw_brake = 0.0
            else:
                raw_brake = raw_brake * 0.45  # Parzializzazione del picco
            
            # Filtro EMA sul freno per renderlo analogico e fluido
            alpha_brake = 0.15
            brake = (alpha_brake * raw_brake) + ((1.0 - alpha_brake) * prev_brake)
            prev_brake = brake
            
            # ==========================================
            # 2. ANTI-STALLO ED EVITAMENTO BLOCCHI ALLA PARTENZA
            # ==========================================
            if spd < 30.0:
                brake = 0.0                
                prev_brake = 0.0
                raw_accel = max(raw_accel, 0.80) 

            # ==========================================
            # 3. FILTRO STERZO SMORZATO (ELIMINA ZIGZAG IN PARTENZA)
            # ==========================================
            # Portato a 0.22 per eliminare le micro-oscillazioni e assorbire le imperfezioni ad alta frequenza
            alpha_steer = 0.22 
            steer = (alpha_steer * raw_steer) + ((1.0 - alpha_steer) * prev_steer)
            prev_steer = steer
            
            steer = max(-1.0, min(1.0, steer))
            accel = max(0.0, min(1.0, raw_accel))
            brake = max(0.0, min(1.0, brake))
            
            # Impedisce la sovrapposizione distruttiva di freno e acceleratore
            accel = accel * (1.0 - brake)
            
            print(f"Velocità: {spd:5.1f} km/h | Sterzo Filtrato: {steer:.3f} | Freno: {brake:.2f}", end='\r')
            
            # ==========================================
            # LIMITATORE DI VELOCITÀ MORBIDO
            # ==========================================
            SPEED_LIMIT = 90.0
            if spd > SPEED_LIMIT:
                accel = max(0.0, accel - 0.5)
            # ==========================================
            
            current_gear, shifted = compute_gear(spd, accel, rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if shifted else steps_since_shift + 1
            
            action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
            action = apply_tcs(action, raw_ob, slip_threshold=5.0)
            
            env.client.R.d['steer'] = action[0]
            env.client.R.d['accel'] = action[1]
            env.client.R.d['brake'] = action[2]
            env.client.R.d['gear']  = int(action[3])
            
            env.client.respond_to_server()
            env.client.get_servers_input()
            
            if env.client.R.d.get('meta', 0) == 1:
                print("\n\n[!] Fine sessione o fuori pista. Reset totale...")
                env.reset(relaunch=False)
                current_gear = 1
                steps_since_shift = 0
                prev_steer = 0.0 
                prev_brake = 0.0
                state_history.clear() 
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta.")
    finally:
        env.end()

if __name__ == "__main__":
    main()