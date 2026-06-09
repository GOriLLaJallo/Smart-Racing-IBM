"""
Data Collection — Giro Secco TORCS (Human-in-the-Loop)

Registra singoli giri con partenza da fermo usando un controller PS5 DualSense o Tastiera.
Ogni giro viene validato (nessuna uscita di pista + lap time registrato).
Solo i giri validi vengono salvati in file HDF5 separati.

Funzionalità extra:
    - Cambio automatico deterministico anti-hunting (--auto_gear)
    - Toggle registrazione in tempo reale (Tasto Cerchio sul DualSense / 'R' su Tastiera)
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import pygame
from datetime import datetime
from typing import Optional

# Forza la visualizzazione della GUI di TORCS per la data collection
os.environ['SHOW_GUI'] = '1'

# Aggiungo gym_torcs al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"ERRORE FATALE: Impossibile importare gym_torcs o una sua dipendenza.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
#  Controller PS5 DualSense
# ──────────────────────────────────────────────────────────────────────

class DualSenseController:
    """Gestisce il polling del controller PlayStation 5 tramite Pygame."""

    AXIS_STEER = 0
    AXIS_L2 = 4       # Brake
    AXIS_R2 = 5       # Accel

    BTN_SQUARE = 0     # Upshift
    BTN_CIRCLE = 1     # TOGGLE REGISTRAZIONE (DualSense / Circle)
    BTN_CROSS = 2      # Downshift
    BTN_TRIANGLE = 3   # REC / Triangle

    DEBOUNCE_MS = 200  # Millisecondi di debounce per i pulsanti del cambio
    REC_DEBOUNCE_MS = 400  # Debounce più lungo per il tasto REC

    def __init__(self, steering_deadzone: float = 0.20):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un DualSense e riprova.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  Controller inizializzato: {self.joystick.get_name()}")

        self.steering_deadzone = steering_deadzone
        self.gear = 1
        self.recording_active = True  # Stato di registrazione dinamico

        self._r2_initialized = False
        self._l2_initialized = False
        self._last_shift_time = 0
        self._last_rec_time = 0

    def check_record_toggle(self) -> bool:
        now = pygame.time.get_ticks()
        if now - self._last_rec_time > self.REC_DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_TRIANGLE) or self.joystick.get_button(self.BTN_CIRCLE):
                self._last_rec_time = now
                return True
        return False

    def get_action(self) -> np.ndarray:
        pygame.event.clear()
        now = pygame.time.get_ticks()

        # ── Sterzo con deadzone ──
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # ── Acceleratore (R2) ──
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1:
                self._r2_initialized = True
            accel = 0.0
        else:
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05:
                accel = 0.0

        # ── Freno (L2) ──
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1:
                self._l2_initialized = True
            brake = 0.0
        else:
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05:
                brake = 0.0

        # ── Cambio marcia manuale (usato solo se --auto_gear è disattivato) ──
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > -1:  # Min gear -1 (retromarcia nella raccolta dati)
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        try:
            self.joystick.rumble(0.0, float(min(0.5, intensity)), int(duration_ms))
        except Exception:
            pass


class KeyboardController:
    """Gestisce la guida di TORCS tramite la tastiera (WASD + Frecce)."""
    DEBOUNCE_MS = 250
    REC_DEBOUNCE_MS = 400

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Input Focus")
        
        self.gear = 1
        self.steer_val = 0.0
        self.recording_active = True
        self._last_shift_time = 0
        self._last_rec_time = 0
        print("  [Keyboard] Inizializzato. MANTIENI IL FOCUS sulla finestra nera 'Input Focus' per guidare!")

    # [NUOVO METODO]
    def check_record_toggle(self) -> bool:
        now = pygame.time.get_ticks()
        keys = pygame.key.get_pressed()
        if now - self._last_rec_time > self.REC_DEBOUNCE_MS:
            if keys[pygame.K_r]:
                self._last_rec_time = now
                return True
        return False

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        pass

    def get_action(self) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()
        now = pygame.time.get_ticks()

        # 1. Sterzo graduale
        steer_target = 0.0
        if keys[pygame.K_a]:
            steer_target = 1.0  
        elif keys[pygame.K_d]:
            steer_target = -1.0  

        if self.steer_val < steer_target:
            self.steer_val = min(steer_target, self.steer_val + 0.08)
        elif self.steer_val > steer_target:
            self.steer_val = max(steer_target, self.steer_val - 0.08)

        # 2. Acceleratore e Freno
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        if brake > 0.1:
            accel = 0.0

        # 3. Cambio marcia manuale
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

        self.screen.fill((30, 30, 40))
        pygame.display.flip()

        return np.array([self.steer_val, accel, brake, float(self.gear)], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Zone Problematiche di Default & Utility
# ──────────────────────────────────────────────────────────────────────
PROBLEM_ZONES = [
    (340, 530), (670, 810), (940, 1070), (1420, 1590), (1870, 1980),
    (2380, 2530), (2570, 2780), (2890, 3020), (3190, 3300),
]

def _parse_zones(spec):
    if not spec: return list(PROBLEM_ZONES)
    return [tuple(map(float, part.split(':'))) for part in spec.split(',')]

def _zone_index(dist, zones):
    for zi, (a, b) in enumerate(zones):
        if a <= dist <= b: return zi
    return None

def _extract_segments(dists, zones, margin_steps=15):
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
            _array('wheelSpinVel', 4) / 100.0,
            np.array([_scalar('rpm') / 10000.0]),
        ]).astype(np.float32)
    except Exception as e:
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
    wsv = obs.get('wheelSpinVel', None)
    if wsv is None: return action
    wsv = np.array(wsv, dtype=np.float64).flatten()
    if wsv.shape[0] < 4: return action

    slip = ((wsv[2] + wsv[3]) / 2.0) - ((wsv[0] + wsv[1]) / 2.0)
    if slip > slip_threshold:
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction
    return action

# ──────────────────────────────────────────────────────────────────────
#  Gearing Deterministico (Importato logicamente da gearing.py)
# ──────────────────────────────────────────────────────────────────────
UP_SPEED = [55.0, 118.0, 200.0, 258.0, 286.0]
DN_SPEED = [40.0, 92.0, 165.0, 232.0, 272.0]
UP_RPM_GATE = 15500.0   
UP_ACCEL_GATE = 0.4     
SHIFT_COOLDOWN = 5      

def compute_gear(speed_kmh: float, accel: float, rpm: float, current_gear: int, steps_since_shift: int):
    g = int(current_gear)
    if g < 1: g = 1
    if steps_since_shift < SHIFT_COOLDOWN: return g, False

    if g < 6 and speed_kmh > UP_SPEED[g - 1] and accel > UP_ACCEL_GATE and rpm > UP_RPM_GATE:
        return g + 1, True
    if g > 1 and speed_kmh < DN_SPEED[g - 2]:
        return g - 1, True
    return g, False


# ──────────────────────────────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Data Collection TORCS — Giro Secco con controller PS5"
    )
    parser.add_argument(
        "--output_dir", type=str, default="train_set",
        help="Directory di output per i file HDF5 e il log (default: directory corrente)"
    )
    parser.add_argument(
        "--device", type=str, choices=["controller", "keyboard"], default="controller",
        help="Dispositivo di input: 'controller' (PS5 DualSense) o 'keyboard' (tastiera WASD)"
    )
    parser.add_argument(
        "--steering_deadzone", type=float, default=0.05,
        help="Deadzone dello sterzo [0.0-0.2] (default: 0.05)"
    )
    parser.add_argument(
        "--relaunch_every", type=int, default=10,
        help="Rilancia TORCS ogni N giri per prevenire memory leak (default: 10)"
    )
    parser.add_argument(
        "--tcs", action="store_true", default=True,
        help="Abilita il Traction Control System (default: abilitato)"
    )
    parser.add_argument(
        "--no-tcs", dest="tcs", action="store_false",
        help="Disabilita il Traction Control System"
    )
    parser.add_argument(
        "--tcs_slip", type=float, default=5.0,
        help="Soglia di slip del TCS (default: 5.0)"
    )
    parser.add_argument(
        "--zones", type=str, default=None,
        help="Zone curva target (distFromStart in metri) come 'a:b,c:d'. Default: PROBLEM_ZONES auto-rilevate per geometria."
    )
    parser.add_argument(
        "--segment_only", action="store_true",
        help="Salva SOLO i segmenti dentro le zone (raccolta parziale): guidi giri interi, vengono tenute solo le curve strette."
    )
    parser.add_argument(
        "--auto_gear", action="store_true",
        help="Abilita il cambio automatico deterministico (ignora l'input manuale delle marce)"
    )
    parser.add_argument(
        "--recovery", action="store_true",
        help="Modalità recupero: parte con REC in pausa e non penalizza i fuori pista."
    )
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    zones = _parse_zones(args.zones)
    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "session_logs", "giri")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.log")

    if args.device == "keyboard":
        controller = KeyboardController()
    else:
        try:
            controller = DualSenseController(steering_deadzone=args.steering_deadzone)
        except RuntimeError as e:
            sys.exit(1)

    existing_laps = sorted([f for f in os.listdir(laps_dir) if f.startswith("lap_") and f.endswith(".h5")])
    lap_counter = len(existing_laps)
    session_saved, session_discarded = 0, 0

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    lap_attempt = 0
    force_relaunch = False

    try:
        while True:
            lap_attempt += 1
            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0) or force_relaunch
            env.reset(relaunch=need_relaunch)
            ob = env.client.S.d
            force_relaunch = False

            state_vec = flatten_state(ob)
            lap_states, lap_actions, lap_dists = [], [], []
            active_zone_idx = None
            lap_valid, lap_completed, went_off_track = True, False, False

            prev_last_lap_time = _get_last_lap_time(ob)
            prev_cur_lap_time = _get_cur_lap_time(ob)
            prev_dist = _get_dist_from_start(ob)

            controller.gear = 1
            steps_since_shift = 0
            step = 0

            controller.recording_active = not args.recovery
            print(f"\n🚀 TENTATIVO GIRO #{lap_attempt} (Salvati: {lap_counter})")
            print(f"   Registrazione iniziale: {'ATTIVATA' if controller.recording_active else 'IN PAUSA'}")

            while True:
                step += 1

                if controller.check_record_toggle():
                    controller.recording_active = not controller.recording_active
                    status = "🔴 REC ATTIVO" if controller.recording_active else "⏸️ REC IN PAUSA"
                    print(f"\n  [MANUAL REC] {status}")
                    controller.rumble(intensity=0.8, duration_ms=250)

                action = controller.get_action()

                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                if args.auto_gear:
                    speed_x_raw = ob.get('speedX', 0.0)
                    speed_x_val = float(speed_x_raw.flat[0]) if isinstance(speed_x_raw, np.ndarray) else float(speed_x_raw)
                    speed_kmh = speed_x_val * 50.0
                    
                    rpm_raw = ob.get('rpm', 0.0)
                    rpm_val = float(rpm_raw.flat[0]) if isinstance(rpm_raw, np.ndarray) else float(rpm_raw)
                    
                    new_gear, shifted = compute_gear(speed_kmh, action[1], rpm_val, controller.gear, steps_since_shift)
                    if shifted:
                        controller.gear = new_gear
                        steps_since_shift = 0
                    else:
                        steps_since_shift += 1
                    action[3] = float(controller.gear)

                # Step Simulatore
                env.client.R.d['steer'] = action[0]
                env.client.R.d['accel'] = action[1]
                env.client.R.d['brake'] = action[2]
                env.client.R.d['gear'] = int(action[3])
                
                env.client.respond_to_server()
                env.client.get_servers_input()
                
                ob_next = env.client.S.d
                done = env.client.R.d.get('meta', 0) == 1
                next_state_vec = flatten_state(ob_next)

                if controller.recording_active:
                    lap_states.append(state_vec.copy())
                    lap_actions.append(action.copy())
                    lap_dists.append(_get_dist_from_start(ob))

                state_vec = next_state_vec
                ob = ob_next

                # Controllo fuori pista
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]

                # Usiamo 1.5 come limite per permettere una guida più aggressiva sui cordoli.
                if abs(current_track_pos) > 1.5:
                    if not args.recovery:
                        # Comportamento normale: fallimento e riavvio
                        print(f"\n  ❌ [OFF-TRACK] trackPos: {current_track_pos:.2f} - Riavvio immediato simulazione.")
                        went_off_track = True
                        lap_completed = True
                        lap_valid = False
                        force_relaunch = True
                        break
                    else:
                        # In modalità recovery, ignora l'errore e continua a farci guidare
                        if step % 50 == 0:
                            print(f"  ⚠️ [RECOVERY] Sei molto fuori pista, ma continuo... (trackPos: {current_track_pos:.2f})", end='\r')

                current_last_lap = _get_last_lap_time(ob_next)
                current_cur_lap = _get_cur_lap_time(ob_next)
                current_dist = _get_dist_from_start(ob_next)

                cur_zone = _zone_index(current_dist, zones)
                if cur_zone is not None and cur_zone != active_zone_idx:
                    controller.rumble(0.3, 180)
                active_zone_idx = cur_zone

                # Log ogni secondo circa (50 step) — indicatore zona (solo mentre registra)
                if controller.recording_active and step % 50 == 0:
                    zone_tag = "  🎯 ZONA TARGET" if cur_zone is not None else ""
                    saved_steps = len(lap_states)
                    print(f"    🔴 [REC Steps: {saved_steps:4d}] CurTime: {current_cur_lap:6.2f} | Dist: {current_dist:7.1f}{zone_tag}       ", end='\r')

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap_time) > 0.0001:
                    lap_completed = True
                    lap_valid = not went_off_track
                    lap_time = current_last_lap
                    break
                elif current_cur_lap < 1.5 and prev_cur_lap_time > 5.0:
                    lap_completed, lap_valid = True, False
                    break
                elif current_dist < 50.0 and prev_dist > 500.0 and step > 500:
                    lap_completed, lap_valid = True, False
                    break

                prev_cur_lap_time = current_cur_lap
                prev_dist = current_dist
                if done: break

            # Fine giro
            if lap_completed and lap_valid and len(lap_states) > 0:
                states_np = np.stack(lap_states)
                actions_np = np.stack(lap_actions)
                dists_np = np.asarray(lap_dists, dtype=np.float32)

                def _write_h5(path, st, ac, di):
                    with h5py.File(path, 'w') as h5f:
                        h5f.create_dataset('states', data=st, compression="gzip")
                        h5f.create_dataset('actions', data=ac, compression="gzip")
                        h5f.create_dataset('dist_from_start', data=di, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(st)

                if args.segment_only:
                    segs = [(s, e) for (s, e) in _extract_segments(dists_np, zones, margin_steps=15) if e - s >= 20]
                    for (s, e) in segs:
                        lap_counter += 1
                        _write_h5(os.path.join(laps_dir, f"lap_seg_{lap_counter:03d}.h5"), states_np[s:e], actions_np[s:e], dists_np[s:e])
                    print(f"\n  ✅ SALVATI {len(segs)} SEGMENTI")
                else:
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