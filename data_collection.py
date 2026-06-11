"""
Data Collection — Giro Secco TORCS

Registra singoli giri con partenza da fermo usando un controller PS4 Dualshock/PS5 DualSense o Tastiera.
Ogni giro viene validato (nessuna uscita di pista + lap time registrato).
Solo i giri validi vengono salvati in file HDF5 separati.

Loop infinito: registra → valida → salva (se valido) → riavvia → ripeti.
Interrompere con Ctrl+C. Il giro corrente incompleto NON viene salvato.

Funzionalità supportate:
    - Cambio automatico deterministico anti-hunting (--auto_gear) tramite modulo esterno
    - Modalità recovery per ignorare i fuori pista (--recovery)
    - Traction Control System configurabile (--tcs)
    - Toggle registrazione manuale (Triangolo su DualSense / 'R' su Tastiera)

Formato output (se il giro è valido e contiene dati registrati):
    lap_001.h5, lap_002.h5, ...              (un file per giro valido)
    session_logs/giri/session_YYYYMMDD.log   (log di sessione testuale)
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

# Forza la visualizzazione della GUI di TORCS per la data collection
os.environ['SHOW_GUI'] = '1'

# Aggiunge la cartella gym_torcs al path di sistema
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
    """Gestisce l'input del controller PlayStation 5 tramite Pygame.

    Mappatura:
        Left Stick X → Sterzo continuo (con deadzone)
        R2 (asse 5)  → Acceleratore [0, 1]
        L2 (asse 2)  → Freno [0, 1]
        Quadrato     → Upshift (Marcia su)
        X (Cross)    → Downshift (Marcia giù)
        Triangolo    → Toggle Registrazione (REC)
    """

    AXIS_STEER = 0
    AXIS_L2 = 4       # Freno
    AXIS_R2 = 5       # Acceleratore

    BTN_CROSS = 2     # Downshift
    BTN_SQUARE = 0    # Upshift
    BTN_TRIANGLE = 3  # Tasto REC

    DEBOUNCE_MS = 200      # Debounce standard per le marce
    REC_DEBOUNCE_MS = 400  # Debounce prolungato per il tasto REC

    def __init__(self, steering_deadzone: float = 0.20):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un DualSense e riprova.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  [Controller] Inizializzato: {self.joystick.get_name()}")

        self.steering_deadzone = steering_deadzone
        self.gear = 1  # Partenza in prima marcia

        # Flag di warm-up per evitare letture spurie dei grilletti all'avvio
        self._r2_initialized = False
        self._l2_initialized = False

        self._last_shift_time = 0
        self._last_rec_time = 0

    def check_record_toggle(self) -> bool:
        """Verifica se il tasto di registrazione è stato premuto, applicando il debounce."""
        now = pygame.time.get_ticks()
        if now - self._last_rec_time > self.REC_DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_TRIANGLE):
                self._last_rec_time = now
                return True
        return False

    def get_action(self) -> np.ndarray:
        """Legge lo stato del controller. Ritorna: [steering, accel, brake, gear]."""

        # Svuota la coda eventi di Pygame per azzerare l'input lag
        pygame.event.clear()

        # ── Sterzo ──
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # ── Acceleratore (R2) ──
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1: self._r2_initialized = True
            accel = 0.0
        else:
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05: accel = 0.0

        # ── Freno (L2) ──
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1: self._l2_initialized = True
            brake = 0.0
        else:
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05: brake = 0.0

        # ── Cambio marcia manuale ──
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > -1:  # Permette la retromarcia (-1)
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """Attiva la vibrazione aptica del controller per feedback fisici."""
        try:
            self.joystick.rumble(0.0, float(min(0.5, intensity)), int(duration_ms))
        except Exception:
            pass


class KeyboardController:
    """Gestisce la guida tramite la tastiera (WASD + Frecce).
    Richiede il focus su una finestra Pygame per catturare gli input.
    """
    DEBOUNCE_MS = 250
    REC_DEBOUNCE_MS = 400

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Input Focus")
        
        self.gear = 1
        self.steer_val = 0.0
        self._last_shift_time = 0
        self._last_rec_time = 0
        print("  [Keyboard] Inizializzato. MANTIENI IL FOCUS sulla finestra nera per guidare!")

    def check_record_toggle(self) -> bool:
        """Verifica se il tasto 'R' (REC) è stato premuto."""
        now = pygame.time.get_ticks()
        keys = pygame.key.get_pressed()
        if now - self._last_rec_time > self.REC_DEBOUNCE_MS:
            if keys[pygame.K_r]:
                self._last_rec_time = now
                return True
        return False

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        pass # La tastiera non supporta il feedback aptico

    def get_action(self) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()

        # ── Sterzo (A/D con interpolazione fluida) ──
        steer_target = 0.0
        if keys[pygame.K_a]: steer_target = 1.0  
        elif keys[pygame.K_d]: steer_target = -1.0  

        if self.steer_val < steer_target:
            self.steer_val = min(steer_target, self.steer_val + 0.08)
        elif self.steer_val > steer_target:
            self.steer_val = max(steer_target, self.steer_val - 0.08)

        # ── Acceleratore (W) e Freno (S) ──
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        if brake > 0.1:
            accel = 0.0 # Il freno disattiva l'acceleratore

        # ── Cambio marcia (Frecce) ──
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if keys[pygame.K_UP]:
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif keys[pygame.K_DOWN]:
                if self.gear > -1:
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        # Aggiorna la schermata per non farla freezare a livello di SO
        self.screen.fill((30, 30, 40))
        pygame.display.flip()

        return np.array([self.steer_val, accel, brake, float(self.gear)], dtype=np.float32)

# ==============================================================================
# UTILITY STATO E TELEMETRIA
# ==============================================================================

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce il dizionario di osservazione in un array 1D (29 dimensioni)."""
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
            _array('track', 19),
            np.array([_scalar('trackPos')]),
            np.array([_scalar('speedX')]),
            np.array([_scalar('speedY')]),
            np.array([_scalar('speedZ')]),
            _array('wheelSpinVel', 4) / 100.0,    # Scalato
            np.array([_scalar('rpm') / 10000.0]), # Scalato
        ]).astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Errore in flatten_state: {e}. Ritorno array zeri.")
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
    """Traction Control System. Riduce il gas se rileva slittamento asse post. vs ant."""
    wsv = obs.get('wheelSpinVel', None)
    if wsv is None: return action

    wsv = np.array(wsv, dtype=np.float64).flatten()
    if wsv.shape[0] < 4: return action

    rear_avg = (wsv[2] + wsv[3]) / 2.0
    front_avg = (wsv[0] + wsv[1]) / 2.0
    slip = rear_avg - front_avg

    if slip > slip_threshold:
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction  # Taglio progressivo dell'acceleratore

    return action

# ==============================================================================
# MAIN LOOP DI DATA COLLECTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Data Collection TORCS — Giro Secco (Manuale Toggle REC)")
    parser.add_argument("--output_dir", type=str, default="train_set", help="Cartella di destinazione")
    parser.add_argument("--device", type=str, choices=["controller", "keyboard"], default="controller")
    parser.add_argument("--steering_deadzone", type=float, default=0.05)
    parser.add_argument("--relaunch_every", type=int, default=10, help="Previene memory leak")
    parser.add_argument("--tcs", action="store_true", default=True, help="Abilita il TCS")
    parser.add_argument("--no-tcs", dest="tcs", action="store_false", help="Disabilita il TCS")
    parser.add_argument("--tcs_slip", type=float, default=5.0)
    parser.add_argument("--auto_gear", action="store_true", help="Usa il modulo gearing esterno")
    parser.add_argument("--recovery", action="store_true", help="Parte in pausa, ignora penalità fuori pista")
    args = parser.parse_args()

    # Previene crash passando argomenti al parser interno di TORCS
    sys.argv = [sys.argv[0]]

    # ── Configurazione Cartelle ──
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "session_logs", "giri")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.log")

    # ── Configurazione Controller ──
    if args.device == "keyboard":
        controller = KeyboardController()
    else:
        try:
            controller = DualSenseController(steering_deadzone=args.steering_deadzone)
        except RuntimeError as e:
            print(f"  ❌ Errore controller: {e}")
            sys.exit(1)

    existing_laps = sorted([f for f in os.listdir(laps_dir) if f.startswith("lap_") and f.endswith(".h5")])
    lap_counter = len(existing_laps)

    session_saved = 0
    session_discarded = 0

    print("\n" + "=" * 64)
    print("   🏎️  DATA COLLECTION — Giro Secco TORCS (Manuale Toggle)")
    print("   Premi Ctrl+C nel terminale per terminare la sessione")
    print("=" * 64)

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    lap_attempt = 0
    force_relaunch = False

    try:
        while True:
            lap_attempt += 1

            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0) or force_relaunch
            env.reset(relaunch=need_relaunch)
            force_relaunch = False
            
            ob = env.client.S.d
            state_vec = flatten_state(ob)

            # Buffer dati in RAM per il giro corrente
            lap_states, lap_actions, lap_dists = [], [], []

            lap_valid = True
            invalidation_reason = ""
            lap_completed = False
            lap_time = 0.0
            went_off_track = False

            prev_last_lap_time = _get_last_lap_time(ob)
            prev_cur_lap_time = _get_cur_lap_time(ob)
            prev_dist = _get_dist_from_start(ob)

            controller.gear = 1
            steps_since_shift = 0

            print(f"\n{'─' * 64}")
            print(f"  🏁 TENTATIVO GIRO #{lap_attempt}  (giri salvati finora: {lap_counter})")
            print(f"{'─' * 64}")

            step = 0
            # Se siamo in recovery partiamo in pausa, altrimenti iniziamo a registrare da subito
            is_recording = not args.recovery

            while True:
                step += 1

                # ── Gestione Tasto Manuale REC ──
                if controller.check_record_toggle():
                    is_recording = not is_recording
                    stato = "🔴 REC ATTIVO" if is_recording else "⏸️ REC IN PAUSA"
                    print(f"\n  [MANUAL REC] {stato}")
                    controller.rumble(intensity=0.8, duration_ms=250)

                action = controller.get_action()

                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                # ── Cambio Automatico Esterno ──
                if args.auto_gear:
                    speed_x_raw = ob.get('speedX', 0.0)
                    speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                    speed_kmh = speed_x_val * 50.0  
                    
                    rpm_raw = ob.get('rpm', 0.0)
                    rpm_val = float(rpm_raw.flat[0]) if isinstance(rpm_raw, np.ndarray) else float(rpm_raw)
                    
                    accel_applied = action[1] 
                    
                    new_gear, shifted = compute_gear(
                        speed_kmh=speed_kmh, 
                        accel=accel_applied, 
                        rpm=rpm_val, 
                        current_gear=controller.gear, 
                        steps_since_shift=steps_since_shift
                    )
                    
                    if shifted:
                        controller.gear = new_gear
                        steps_since_shift = 0
                    else:
                        steps_since_shift += 1
                    
                    action[3] = float(controller.gear)

                # ── Step di Simulazione ──
                env.client.R.d['steer'] = action[0]
                env.client.R.d['accel'] = action[1]
                env.client.R.d['brake'] = action[2]
                env.client.R.d['gear'] = int(action[3])
                
                env.client.respond_to_server()
                env.client.get_servers_input()
                
                ob_next = env.client.S.d
                done = env.client.R.d.get('meta', 0) == 1
                next_state_vec = flatten_state(ob_next)

                # Salva i dati solo se l'utente ha il REC attivo
                if is_recording:
                    lap_states.append(state_vec.copy())
                    lap_actions.append(action.copy())
                    lap_dists.append(_get_dist_from_start(ob))

                state_vec = next_state_vec
                ob = ob_next

                # ── Controllo Fuori Pista ──
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]
                
                if abs(current_track_pos) > 1.5:
                    if not args.recovery:
                        print(f"\n  ❌ [OFF-TRACK] trackPos: {current_track_pos:.2f} - Riavvio immediato.")
                        went_off_track = True
                        lap_completed = True
                        lap_valid = False
                        invalidation_reason = f"Fuori pista (trackPos: {current_track_pos:.2f})"
                        force_relaunch = True
                        break
                    else:
                        if step % 50 == 0:
                            print(f"  ⚠️ [RECOVERY] Sei fuori pista... (trackPos: {current_track_pos:.2f})", end='\r')

                # ── Log in tempo reale e fine giro ──
                current_last_lap = _get_last_lap_time(ob_next)
                current_cur_lap = _get_cur_lap_time(ob_next)
                current_dist = _get_dist_from_start(ob_next)

                if is_recording and step % 50 == 0:
                    saved_steps = len(lap_states) 
                    print(f"    🔴 [REC Steps: {saved_steps:4d}] CurTime: {current_cur_lap:6.2f} | Dist: {current_dist:7.1f}       ", end='\r')

                # Verifica traguardo
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap_time) > 0.0001:
                    lap_completed = True
                    if went_off_track:
                        lap_valid = False
                        invalidation_reason = "Giro invalidato (fuori pista pregresso)"
                    else:
                        lap_valid = True
                        lap_time = current_last_lap
                        print(f"\n  🏁 TRAGUARDO! Lap time: {lap_time:.3f}s")
                elif current_cur_lap < 1.5 and prev_cur_lap_time > 5.0:
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = "Giro invalidato (curTime resettato dal server)"
                elif current_dist < 50.0 and prev_dist > 500.0 and step > 500:
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = "Reset anomalo della distanza"

                prev_cur_lap_time = current_cur_lap
                prev_dist = current_dist

                if lap_completed or done:
                    break

            # ── Salvataggio HDF5 ──
            if lap_completed and lap_valid:
                if len(lap_states) == 0:
                    session_discarded += 1
                    log_entry = f"[EMPTY] Att. #{lap_attempt} | Motivo: Giro valido ma REC sempre spento | {datetime.now().isoformat()}"
                    print(f"  ⚠️ GIRO VALIDO MA VUOTO — Non hai mai attivato il tasto REC, nessun file salvato.")
                else:
                    states_np = np.stack(lap_states)
                    actions_np = np.stack(lap_actions)
                    dists_np = np.asarray(lap_dists, dtype=np.float32)

                    lap_counter += 1
                    filename = f"lap_{lap_counter:03d}.h5"
                    
                    with h5py.File(os.path.join(laps_dir, filename), 'w') as h5f:
                        h5f.create_dataset('states', data=states_np, compression="gzip")
                        h5f.create_dataset('actions', data=actions_np, compression="gzip")
                        h5f.create_dataset('dist_from_start', data=dists_np, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(states_np)
                        h5f.attrs['has_dist_meta'] = True
                        h5f.attrs['timestamp'] = datetime.now().isoformat()

                    print(f"  ✅ GIRO SALVATO — File: {filename} ({len(states_np)} frame registrati)")
                    session_saved += 1
                    log_entry = f"[SAVED] | Lap Time: {lap_time:.3f}s | Steps: {len(states_np)} | {datetime.now().isoformat()}"
            else:
                session_discarded += 1
                reason = invalidation_reason if invalidation_reason else "Non completato"
                log_entry = f"[DISCARDED] Att. #{lap_attempt} | Motivo: {reason} | Steps: {len(lap_states)} | {datetime.now().isoformat()}"
                print(f"  ❌ GIRO SCARTATO — {reason}")

            # Scrittura Log di sessione
            with open(log_path, 'a') as f:
                f.write(log_entry + "\n")

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 64}")
        print(f"  🛑 SESSIONE TERMINATA (Ctrl+C)")
        print(f"     Giri salvati con successo:  {session_saved}")
        print(f"     Tentativi scartati/vuoti:   {session_discarded}")
        print(f"{'=' * 64}")

    finally:
        try:
            with open(log_path, 'a') as f:
                f.write(f"\n--- RIEPILOGO SESSIONE ---\n")
                f.write(f"Giri salvati: {session_saved}\n")
                f.write(f"Giri scartati: {session_discarded}\n")
        except Exception: pass
        env.end()
        pygame.quit()

if __name__ == "__main__":
    main()