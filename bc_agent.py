import os
import sys
import time
import json
import numpy as np
import torch
import snakeoil3_gym as snakeoil3

from data_collection import flatten_state, apply_tcs
from train_bc import Normalizer, BCModel

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("ERRORE: Impossibile importare gym_torcs.")
    sys.exit(1)

def main():
    print("="*60)
    print(" 🤖 AGENTE AUTONOMO - BEHAVIORAL CLONING ")
    print("="*60)

    # ── 1. Carica lo Scaler (Normalizer) ──
    scaler_path = "bc_scaler.json"
    if not os.path.exists(scaler_path):
        print(f"ERRORE: Scaler non trovato in {scaler_path}.")
        print("Devi prima addestrare il modello usando train_bc.py!")
        return

    normalizer = Normalizer()
    normalizer.load(scaler_path)
    print("[+] Scaler caricato con successo.")

    # ── 2. Carica il Modello (Rete Neurale) ──
    model_path = "bc_model.pth"
    if not os.path.exists(model_path):
        print(f"ERRORE: Modello non trovato in {model_path}.")
        print("Devi prima addestrare il modello usando train_bc.py!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel(input_dim=29, output_dim=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Imposta il modello in modalità inferenza
    print(f"[+] Modello PyTorch caricato su {device}.")

    # ── 3. Inizializza Ambiente TORCS ──
    print("\n[+] Avvio simulatore TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    env.reset(relaunch=True)
    
    # Marcia iniziale
    current_gear = 1

    print("\n🏎️ L'AGENTE STA GUIDANDO! Premi Ctrl+C per fermare.")
    
    try:
        while True:
            # Recupera l'osservazione grezza aggirando i limiti di gym_torcs
            raw_ob = env.client.S.d
            
            # 1. Estrai e formatta i sensori (29D)
            state_vec = flatten_state(raw_ob)
            
            # 2. Normalizza i sensori
            # np.expand_dims per creare una "batch" di 1 singolo elemento
            state_norm = normalizer.transform(np.expand_dims(state_vec, axis=0))
            
            # 3. Converti in tensore PyTorch
            state_tensor = torch.FloatTensor(state_norm).to(device)
            
            # 4. Predizione della Rete Neurale
            with torch.no_grad():
                pred = model(state_tensor).cpu().numpy()[0]
                
            steer = pred[0]
            accel = pred[1]
            brake = pred[2]
            
            # Applica dei limiti fisici di sicurezza alle uscite
            steer = max(-1.0, min(1.0, steer))
            accel = max(0.0, min(1.0, accel))
            brake = max(0.0, min(1.0, brake))
            
            # 5. Cambio Automatico (indipendente dalla rete neurale)
            rpm = raw_ob.get('rpm', 0)
            if rpm > 7500 and current_gear < 6:
                current_gear += 1
            elif rpm < 3000 and current_gear > 1:
                current_gear -= 1
                
            # 6. Assembla l'azione: [steer, accel, brake, gear]
            action = np.array([steer, accel, brake, float(current_gear)], dtype=np.float32)
            
            # 7. Applica il Traction Control System (TCS)
            action = apply_tcs(action, raw_ob, slip_threshold=5.0)
            
            # 8. Invia l'azione a TORCS (Aggirando env.step come in data_collection)
            env.client.R.d['steer'] = action[0]
            env.client.R.d['accel'] = action[1]
            env.client.R.d['brake'] = action[2]
            env.client.R.d['gear']  = int(action[3])
            
            env.client.respond_to_server()
            env.client.get_servers_input()
            
            # Se la gara finisce o la macchina esce di pista
            done = env.client.R.d.get('meta', 0) == 1
            if raw_ob.get('track', [0])[0] < 0:
                done = True
                
            if done:
                print("\n[!] Traguardo raggiunto o fuori pista. Riavvio...")
                env.reset(relaunch=False)
                current_gear = 1
                
            time.sleep(0.01) # Piccolo delay per non intasare la CPU

    except KeyboardInterrupt:
        print("\n🛑 Guida interrotta.")
    finally:
        env.end()

if __name__ == "__main__":
    main()
