"""
Data Collection — Giro Secco TORCS

Registra singoli giri con partenza da fermo usando un controller PS5 DualSense o Tastiera.
Ogni giro viene validato (nessuna uscita di pista + lap time registrato).
Solo i giri validi vengono salvati in file HDF5 separati.

Funzionalità extra:
    - Cambio automatico deterministico anti-hunting (--auto_gear)
    - Toggle registrazione in tempo reale (Tasto Cerchio sul DualSense / 'R' su Tastiera)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pygame
from datetime import datetime

# Importa la logica del cambio dal modulo esterno centralizzato
from gearing import compute_gear

# ==============================================================================
# CONFIGURAZIONE INIZIALE
# ==============================================================================

# Forza la visualizzazione della GUI di TORCS per permettere al pilota di vedere la pista
os.environ['SHOW_GUI'] = '1'

# Aggiunge la cartella gym_torcs al path di sistema per consentire l'importazione
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"ERRORE FATALE: Impossibile importare gym_torcs o una sua dipendenza.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)

# ==============================================================================
# CLASSI CONTROLLER
# ==============================================================================

class DualSenseController:
    """Gestisce l'input da un controller PlayStation 5 tramite la libreria Pygame."""

    # Mappatura standard degli assi e pulsanti per DualSense su PC
    AXIS_STEER = 0
    AXIS_L2 = 4       # Freno (Grilletto sinistro)
    AXIS_R2 = 5       # Acceleratore (Grilletto destro)

    BTN_SQUARE = 0    # Marcia su
    BTN_CIRCLE = 1    # Attiva/Disattiva registrazione in tempo reale
    BTN_CROSS = 2     # Marcia giù

    DEBOUNCE_MS = 250 # Ritardo anti-rimbalzo per evitare letture multiple dei pulsanti

    def __init__(self, steering_deadzone: float = 0.20):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un DualSense e riprova.")

        # Inizializza il primo controller rilevato
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  Controller inizializzato: {self.joystick.get_name()}")

        self.steering_deadzone = steering_deadzone
        self.gear = 1  
        self.recording_active = True  # Flag per sospendere il salvataggio dei dati in corsa

        # Variabili di stato per evitare spike improvvisi sui grilletti alla prima pressione
        self._r2_initialized = False
        self._l2_initialized = False
        self._last_shift_time = 0
        self._last_toggle_time = 0

    def get_action(self) -> np.ndarray:
        """Legge lo stato attuale del controller e restituisce l'azione: [sterzo, accel, freno, marcia]."""
        
        # Svuotiamo la coda degli eventi in modo sicuro per non bloccare il sistema operativo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
                
        now = pygame.time.get_ticks()

        # -- Toggle Registrazione (Cerchio) --
        if now - self._last_toggle_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_CIRCLE):
                self.recording_active = not self.recording_active
                status = "ATTIVATA" if self.recording_active else "IN PAUSA"
                print(f"\n  🎥 [REGISTRAZIONE] {status}")
                self._last_toggle_time = now

        # -- Sterzo con Deadzone --
        # Rimuove le piccole fluttuazioni della levetta quando è a riposo
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # -- Acceleratore (R2) --
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1:
                self._r2_initialized = True
            accel = 0.0
        else:
            # Mappa l'asse [-1, 1] al range [0, 1]
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05: accel = 0.0

        # -- Freno (L2) --
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1:
                self._l2_initialized = True
            brake = 0.0
        else:
            # Mappa l'asse [-1, 1] al range [0, 1]
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05: brake = 0.0

        # -- Cambio marcia manuale (Quadrato / Croce) --
        # Viene ignorato dal main se --auto_gear è attivo
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > 1:  
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """Attiva la vibrazione aptica del controller (utile per feedback sui settori)."""
        try:
            self.joystick.rumble(0.0, float(min(0.5, intensity)), int(duration_ms))
        except Exception:
            pass


class KeyboardController:
    """Gestisce la guida tramite la tastiera (WASD + Frecce) creando una finestrella di focus."""
    DEBOUNCE_MS = 250

    def __init__(self):
        pygame.init()
        # Crea una piccola finestra di sistema obbligatoria per catturare gli input di tastiera in Pygame
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Input Focus")
        
        self.gear = 1
        self.steer_val = 0.0
        self.recording_active = True
        self._last_shift_time = 0
        self._last_toggle_time = 0
        print("  [Keyboard] MANTIENI IL FOCUS sulla finestra nera per guidare!")

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        # Nessun feedback aptico disponibile su tastiera
        pass

    def get_action(self) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()
        now = pygame.time.get_ticks()

        # -- Toggle Registrazione (Tasto R) --
        if now - self._last_toggle_time > self.DEBOUNCE_MS:
            if keys[pygame.K_r]:
                self.recording_active = not self.recording_active
                status = "ATTIVATA" if self.recording_active else "IN PAUSA"
                print(f"\n  🎥 [REGISTRAZIONE] {status}")
                self._last_toggle_time = now

        # -- Sterzo graduale (A/D) --
        # Rende l'input da tastiera meno scattoso interpolando progressivamente il valore
        steer_target = 0.0
        if keys[pygame.K_a]: steer_target = 1.0  
        elif keys[pygame.K_d]: steer_target = -1.0  

        if self.steer_val < steer_target:
            self.steer_val = min(steer_target, self.steer_val + 0.08)
        elif self.steer_val > steer_target:
            self.steer_val = max(steer_target, self.steer_val - 0.08)

        # -- Acceleratore e Freno (W/S) --
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        # Disattiva l'acceleratore se si sta frenando a fondo
        if brake > 0.1:
            accel = 0.0

        # -- Cambio marcia manuale (Frecce SU/GIÙ) --
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if keys[pygame.K_UP]:
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif keys[pygame.K_DOWN]:
                if self.gear > 1:
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        # Aggiorna la finestrella nera di input
        self.screen.fill((30, 30, 40))
        pygame.display.flip()

        return np.array([self.steer_val, accel, brake, float(self.gear)], dtype=np.float32)

# ==============================================================================
# UTILITY PER LO STATO E L'ELABORAZIONE DATI
# ==============================================================================

# Settori della pista considerati complessi (es. curve strette o punti di staccata)
PROBLEM_ZONES = [
    (340, 530), (670, 810), (940, 1070), (1420, 1590), (1870, 1980),
    (2380, 2530), (2570, 2780), (2890, 3020), (3190, 3300),
]

def _parse_zones(spec):
    """Converte una stringa di input 'start:end,start:end' in una lista di tuple per gestire zone custom."""
    if not spec: return list(PROBLEM_ZONES)
    return [tuple(map(float, part.split(':'))) for part in spec.split(',')]

def _zone_index(dist, zones):
    """Verifica se una data distanza percorsa rientra in una delle 'PROBLEM_ZONES' e ne restituisce l'indice."""
    for zi, (a, b) in enumerate(zones):
        if a <= dist <= b: return zi
    return None

def _extract_segments(dists, zones, margin_steps=15):
    """Estrae gli indici (start, end) degli step di simulazione che cadono nelle zone critiche, aggiungendo un margine."""
    n = len(dists)
    in_zone = [(_zone_index(d, zones) is not None) for d in dists]
    segs, i = [], 0
    while i < n:
        if in_zone[i]:
            j = i
            while j < n and in_zone[j]: j += 1
            segs.append((max(0, i - margin_steps), j))
            i = j
        else: i += 1
    return segs

def flatten_state(state_dict: dict) -> np.ndarray:
    """
    Estrae le variabili grezze dal dizionario di TORCS e le compatta in un array 1D coerente.
    Standardizza grandezze (es. i giri motore vengono scalati per non far impazzire le reti neurali future).
    """
    def _scalar(key: str, default: float = 0.0) -> float:
        val = state_dict.get(key, default)
        if val is None: return default
        return float(val.flat[0]) if isinstance(val, np.ndarray) else float(val)

    def _array(key: str, size: int) -> np.ndarray:
        val = state_dict.get(key, None)
        if val is None: return np.zeros(size, dtype=np.float32)
        arr = np.array(val, dtype=np.float32).flatten()
        if arr.shape[0] != size:
            padded = np.zeros(size, dtype=np.float32)
            padded[:min(size, arr.shape[0])] = arr[:min(size, arr.shape[0])]
            return padded
        return arr

    try:
        return np.concatenate([
            np.array([_scalar('angle')]),
            _array('track', 19),             # Sensori del telemetro laser (distanze dai bordi pista)
            np.array([_scalar('trackPos')]), # Posizione relativa al centro della pista (-1.0, 1.0)
            np.array([_scalar('speedX')]),   # Velocità longitudinale
            np.array([_scalar('speedY')]),   # Velocità laterale
            np.array([_scalar('speedZ')]),
            _array('wheelSpinVel', 4) / 100.0, # Velocità di rotazione delle ruote (scalata)
            np.array([_scalar('rpm') / 10000.0]), # Giri motore (scalati)
        ]).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Errore critico in flatten_state. Formato dizionario anomalo: {e}")
        return np.zeros(29, dtype=np.float32)

def _get_dist_from_start(obs: dict) -> float:
    dfs = obs.get('distFromStart', 0.0)
    return float(dfs.flat[0]) if isinstance(dfs, np.ndarray) else float(dfs)

def _get_cur_lap_time(obs: dict) -> float:
    clt = obs.get('curLapTime', 0.0)
    return float(clt.flat[0]) if isinstance(clt, np.ndarray) else float(clt)

def _get_last_lap_time(obs: dict) -> float:
    llt = obs.get('lastLapTime', 0.0)
    return float(llt.flat[0]) if isinstance(llt, np.ndarray) else float(llt)

def apply_tcs(action: np.ndarray, obs: dict, slip_threshold: float = 5.0) -> np.ndarray:
    """
    Sistema di Controllo della Trazione (TCS) semplice.
    Taglia il gas (azione[1]) in modo proporzionale se rileva una differenza di slittamento
    tra le ruote motrici (posteriori) e quelle anteriori maggiore della soglia.
    """
    wsv = obs.get('wheelSpinVel', None)
    if wsv is None: return action
    wsv = np.array(wsv, dtype=np.float64).flatten()
    if wsv.shape[0] < 4: return action

    # Differenza di rotazione ruote (Posteriore - Anteriore)
    slip = ((wsv[2] + wsv[3]) / 2.0) - ((wsv[0] + wsv[1]) / 2.0)
    if slip > slip_threshold:
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction
    return action

# ==============================================================================
# LOOP PRINCIPALE DI RACCOLTA DATI
# ==============================================================================

def main():
    # Parsing degli argomenti riga di comando
    parser = argparse.ArgumentParser(description="Data Collection TORCS")
    parser.add_argument("--output_dir", type=str, default="train_set")
    parser.add_argument("--device", type=str, choices=["controller", "keyboard"], default="controller")
    parser.add_argument("--steering_deadzone", type=float, default=0.05)
    parser.add_argument("--relaunch_every", type=int, default=10) # Riavvia il simulatore in toto ogni tot giri per prevenire memory leaks
    parser.add_argument("--tcs", action="store_true", default=True)
    parser.add_argument("--tcs_slip", type=float, default=5.0)
    parser.add_argument("--zones", type=str, default=None)
    parser.add_argument("--segment_only", action="store_true") # Se vero, salva solo dati relativi alle zone problematiche
    parser.add_argument("--auto_gear", action="store_true")
    args = parser.parse_args()

    # HACK: Svuota sys.argv preservando solo il nome del file.
    # Questo impedisce a gym_torcs (o sue dipendenze vecchie) di crashare cercando di leggere gli argomenti di argparse.
    sys.argv = [sys.argv[0]]
    
    # Setup directory di salvataggio
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    zones = _parse_zones(args.zones)
    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "session_logs", "giri")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.log")

    # Inizializzazione del dispositivo di input scelto
    if args.device == "keyboard":
        controller = KeyboardController()
    else:
        try:
            controller = DualSenseController(steering_deadzone=args.steering_deadzone)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    # Conta quanti giri validi esistono già per aggiornare la numerazione dei file
    existing_laps = sorted([f for f in os.listdir(laps_dir) if f.startswith("lap_") and f.endswith(".h5")])
    lap_counter = len(existing_laps)
    session_saved, session_discarded = 0, 0

    # Avvio dell'ambiente TORCS in modalità testuale lato client (ma la GUI lato server è forzata su ON)
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    lap_attempt = 0
    force_relaunch = False

    try:
        while True:
            lap_attempt += 1
            
            # Gestione del memory leak di TORCS riavviando il processo C++ periodicamente
            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0) or force_relaunch
            env.reset(relaunch=need_relaunch)
            ob = env.client.S.d
            force_relaunch = False

            state_vec = flatten_state(ob)
            lap_states, lap_actions, lap_dists = [], [], []
            active_zone_idx = None
            lap_valid, lap_completed, went_off_track = True, False, False

            # Valori base per capire quando un giro si chiude o un'auto parte
            prev_last_lap_time = _get_last_lap_time(ob)
            prev_cur_lap_time = _get_cur_lap_time(ob)
            prev_dist = _get_dist_from_start(ob)

            controller.gear = 1
            steps_since_shift = 0
            step = 0

            print(f"\n🚀 TENTATIVO GIRO #{lap_attempt} (Salvati: {lap_counter})")

            # --- CICLO DEL SINGOLO GIRO (TICK DEL SIMULATORE) ---
            while True:
                step += 1
                action = controller.get_action()

                # Se abilitato, limita elettronicamente l'acceleratore
                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                # Se abilitato, calcola in autonomia la marcia
                if args.auto_gear:
                    speed_x_raw = ob.get('speedX', 0.0)
                    speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                    speed_kmh = speed_x_val * 1  #Abbiamo letto la velocità non filtrata da gymtorcs, quindi non divisa per 50
                    
                    rpm_raw = ob.get('rpm', 0.0)
                    rpm_val = float(rpm_raw.flat[0]) if isinstance(rpm_raw, np.ndarray) else float(rpm_raw)
                    
                    new_gear, shifted = compute_gear(speed_kmh, action[1], rpm_val, controller.gear, steps_since_shift)
                    if shifted:
                        controller.gear = new_gear
                        steps_since_shift = 0
                    else:
                        steps_since_shift += 1
                    action[3] = float(controller.gear)

                # Applica l'azione al simulatore
                env.client.R.d['steer'] = action[0]
                env.client.R.d['accel'] = action[1]
                env.client.R.d['brake'] = action[2]
                env.client.R.d['gear'] = int(action[3])
                
                env.client.respond_to_server()
                env.client.get_servers_input()
                
                ob_next = env.client.S.d
                done = env.client.R.d.get('meta', 0) == 1
                next_state_vec = flatten_state(ob_next)

                # -- SALVATAGGIO DEI DATI --
                # Salva lo stato solo se il flag di registrazione in tempo reale è attivo
                if controller.recording_active:
                    lap_states.append(state_vec.copy())
                    lap_actions.append(action.copy())
                    lap_dists.append(_get_dist_from_start(ob))

                state_vec = next_state_vec
                ob = ob_next

                # -- CONTROLLO FUORI PISTA --
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]
                
                # Se trackPos > 1.5 o < -1.5 l'auto è oltre l'erba.
                if abs(current_track_pos) > 1.5:
                    went_off_track, lap_completed, lap_valid = True, True, False
                    force_relaunch = True # Riavvia fisicamente l'auto al pit perché potrebbe essersi schiantata
                    break

                # -- AGGIORNAMENTO PROGRESSI E STAMPA A SCHERMO --
                current_last_lap = _get_last_lap_time(ob_next)
                current_cur_lap = _get_cur_lap_time(ob_next)
                current_dist = _get_dist_from_start(ob_next)

                # Se si entra in un nuovo settore critico, fai vibrare il DualSense
                cur_zone = _zone_index(current_dist, zones)
                if cur_zone is not None and cur_zone != active_zone_idx:
                    controller.rumble(0.3, 180)
                active_zone_idx = cur_zone

                if step % 50 == 0:
                    rec_status = "🔴 REC" if controller.recording_active else "⏸️ PAUSE"
                    print(f"    [{rec_status}] Step {step:4d} | Time: {current_cur_lap:5.1f}s | Dist: {current_dist:6.1f}m", end='\r')

                # -- CONDIZIONI DI TERMINE GIRO --
                if step > 10000: 
                    # Safety net per evitare loop infiniti (es. auto bloccata contro un muro non rilevata dal trackPos)
                    lap_completed, lap_valid = True, False
                    break
                    
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap_time) > 0.0001:
                    # Il giro si è concluso al traguardo (il last_lap_time in TORCS è appena stato aggiornato)
                    lap_completed = True
                    lap_valid = not went_off_track
                    lap_time = current_last_lap
                    break
                elif current_cur_lap < 1.5 and prev_cur_lap_time > 5.0:
                    # Rilevato un reset improvviso del tempo (es. bug di TORCS)
                    lap_completed, lap_valid = True, False
                    break
                elif current_dist < 50.0 and prev_dist > 500.0 and step > 500:
                    # Il tracciatore di distanza si è resettato in un punto non canonico
                    lap_completed, lap_valid = True, False
                    break

                prev_cur_lap_time = current_cur_lap
                prev_dist = current_dist
                if done: break

            # --- GESTIONE DEI DATI DEL GIRO APPENA CONCLUSO ---
            if lap_completed and lap_valid and len(lap_states) > 0:
                # Converti le liste in tensori NumPy per l'export
                states_np = np.stack(lap_states)
                actions_np = np.stack(lap_actions)
                dists_np = np.asarray(lap_dists, dtype=np.float32)

                # Funzione interna per salvare il file HDF5 compresso (risparmia spazio su disco)
                def _write_h5(path, st, ac, di):
                    with h5py.File(path, 'w') as h5f:
                        h5f.create_dataset('states', data=st, compression="gzip")
                        h5f.create_dataset('actions', data=ac, compression="gzip")
                        h5f.create_dataset('dist_from_start', data=di, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(st)

                if args.segment_only:
                    # Modalità parziale: salva solo porzioni di giro intorno alle curve complesse
                    segs = [(s, e) for (s, e) in _extract_segments(dists_np, zones, margin_steps=15) if e - s >= 20]
                    for (s, e) in segs:
                        lap_counter += 1
                        _write_h5(os.path.join(laps_dir, f"lap_seg_{lap_counter:03d}.h5"), states_np[s:e], actions_np[s:e], dists_np[s:e])
                    print(f"\n  ✅ SALVATI {len(segs)} SEGMENTI")
                else:
                    # Modalità standard: salva l'intero giro
                    lap_counter += 1
                    _write_h5(os.path.join(laps_dir, f"lap_{lap_counter:03d}.h5"), states_np, actions_np, dists_np)
                    print(f"\n  ✅ GIRO COMPLETO SALVATO: lap_{lap_counter:03d}.h5")
                session_saved += 1
            else:
                session_discarded += 1
                print(f"\n  ❌ GIRO SCARTATO O VUOTO (Uscita, Interruzione o Pausa Totale)")

    except KeyboardInterrupt:
        print(f"\n🛑 SESSIONE INTERROTTA. Salvati: {session_saved}, Scartati: {session_discarded}")
    finally:
        env.end()
        pygame.quit()

if __name__ == "__main__":
    main()