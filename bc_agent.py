import os
import sys
import time
import json
import numpy as np
import torch
import snakeoil3_gym as snakeoil3

from data_collection import flatten_state, apply_tcs
from train_bc import Normalizer, BCModel
from gearing import compute_gear

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("ERRORE: Impossibile importare gym_torcs.")
    sys.exit(1)

def main():
    print("="*60)
    print(" 🤖 AGENTE AUTONOMO - TEST SOTTOSTERZO (Limite 90 km/h) ")
    print("="*60)

    scaler_path = "bc_scaler.json"
    if not os.path.exists(scaler_path):
        print("ERRORE: Scaler non trovato.")
        return
    normalizer = Normalizer()
    normalizer.load(scaler_path)

    model_path = "bc_model.pth"
    if not os.path.exists(model_path):
        print("ERRORE: Modello non trovato.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel(input_dim=29, output_dim=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    env.reset(relaunch=True)
    
    current_gear = 1
    steps_since_shift = 0
    prev_steer = 0.0

    print("\n🏎️ L'AGENTE STA GUIDANDO! Premi Ctrl+C per fermare.")
    
    try:
        while True:
            raw_ob = env.client.S.d
            
            # 1. Estrazione e normalizzazione
            state_vec = flatten_state(raw_ob)
            state_norm = normalizer.transform(np.expand_dims(state_vec, axis=0))[0]
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            
            # 2. Predizione
            with torch.no_grad():
                pred = model(state_tensor).cpu().numpy()[0]
                
            raw_steer = pred[0]
            raw_accel = pred[1]
            raw_brake = pred[2]
            
            # Moltiplicatore e Filtro Volante
            raw_steer = raw_steer * 1.4 
            alpha_steer = 0.3 
            steer = (alpha_steer * raw_steer) + ((1.0 - alpha_steer) * prev_steer)
            prev_steer = steer
            
            steer = max(-1.0, min(1.0, steer))
            accel = max(0.0, min(1.0, raw_accel))
            brake = max(0.0, min(1.0, raw_brake))
            
            accel = accel * (1.0 - brake)
            
            # ==========================================
            # CORREZIONE BUG VELOCITÀ
            # Togliendo il "* 50.0" leggiamo la velocità reale del gioco
            # ==========================================
            spd = float(np.array(raw_ob.get('speedX', 0.0)).flat[0])
            
            # (Se la velocità dovesse risultare bassissima, es. max 20, 
            # significa che è in m/s e basterà moltiplicarla per 3.6, ma di base TORCS usa km/h)
            rpm = float(np.array(raw_ob.get('rpm', 0.0)).flat[0])
            
            # Stampa a schermo la telemetria per debugging
            print(f"Velocità: {spd:5.1f} km/h | Marcia: {current_gear}", end='\r')
            
            # ==========================================
            # LIMITATORE DI VELOCITÀ RIGIDO
            # ==========================================
            SPEED_LIMIT = 90.0
            if spd > SPEED_LIMIT:
                accel = 0.0 # Taglia il gas
                if spd > SPEED_LIMIT + 5.0:
                    brake = max(brake, 0.4) 
            # ==========================================
            
            # Cambio automatico
            current_gear, shifted = compute_gear(spd, accel, rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if shifted else steps_since_shift + 1
            
            action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
            action = apply_tcs(action, raw_ob, slip_threshold=5.0)
            
            # 4. Invia comandi
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