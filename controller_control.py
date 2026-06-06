import snakeoil3_jm2 as snakeoil3
import time
import json
import threading

try:
    from inputs import get_gamepad
    HAS_INPUTS = True
except ImportError:
    HAS_INPUTS = False
    print("Modulo 'inputs' non trovato. Assicurati di aver installato inputs (pip install inputs).")

class InputThread(threading.Thread):
    def __init__(self):
        super(InputThread, self).__init__()
        self.daemon = True
        
        # Controller state
        self.ABS_X = 0
        self.ABS_Z = 0 # Left Trigger (Brake)
        self.ABS_RZ = 0 # Right Trigger (Accel)
        
        self.BTN_SOUTH = 0 # A
        self.BTN_EAST = 0 # B
        self.BTN_NORTH = 0 # Y
        self.BTN_WEST = 0 # X
        self.BTN_TL = 0 # LB
        self.BTN_TR = 0 # RB
        
        self.running = True
        self.connected = False

    def run(self):
        if not HAS_INPUTS:
            return
            
        while self.running:
            try:
                events = get_gamepad()
                self.connected = True
                for event in events:
                    if event.ev_type == 'Absolute':
                        if event.code == 'ABS_X': self.ABS_X = event.state
                        elif event.code == 'ABS_Z': self.ABS_Z = event.state
                        elif event.code == 'ABS_RZ': self.ABS_RZ = event.state
                    elif event.ev_type == 'Key':
                        if event.code == 'BTN_SOUTH': self.BTN_SOUTH = event.state
                        elif event.code == 'BTN_EAST': self.BTN_EAST = event.state
                        elif event.code == 'BTN_NORTH': self.BTN_NORTH = event.state
                        elif event.code == 'BTN_WEST': self.BTN_WEST = event.state
                        elif event.code == 'BTN_TL': self.BTN_TL = event.state
                        elif event.code == 'BTN_TR': self.BTN_TR = event.state
            except Exception as e:
                self.connected = False
                time.sleep(1)


class ControllerAgent:
    def __init__(self):
        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }
        
        self.input_thread = InputThread()
        self.input_thread.start()
        
        # Attendiamo un attimo per vedere se si connette il pad
        time.sleep(0.5)
        if self.input_thread.connected:
            print("Gamepad connesso con successo!")
        else:
            print("Nessun gamepad rilevato al momento, in attesa di connessione...")
            
        self.gear_up_pressed = False
        self.gear_down_pressed = False

    def update(self, sensors):
        it = self.input_thread
        
        # ABS_X va da -32768 a 32767
        # Invertito perché in TORCS sinistra è positivo e destra è negativo
        steer_axis = -(it.ABS_X / 32768.0)
        
        # Trigger vanno da 0 a 255
        accel_axis = it.ABS_RZ / 255.0
        brake_axis = it.ABS_Z / 255.0
        
        # Aumentiamo gas/freno anche se si premono A/B per comodita'
        if it.BTN_SOUTH: accel_axis = 1.0 # A button
        if it.BTN_EAST or it.BTN_WEST: brake_axis = 1.0 # B or X button
        
        accel_axis = max(0.0, min(1.0, accel_axis))
        brake_axis = max(0.0, min(1.0, brake_axis))

        speed = sensors.get('speedX', 0)
        rpm = sensors.get('rpm', 0)

        # Gear Up: RB o Y
        gear_up = it.BTN_TR or it.BTN_NORTH
        # Gear Down: LB o X
        gear_down = it.BTN_TL or it.BTN_WEST
        
        if gear_up and not self.gear_up_pressed:
            self.state['gear'] += 1
        if gear_down and not self.gear_down_pressed:
            self.state['gear'] -= 1
            
        self.gear_up_pressed = gear_up
        self.gear_down_pressed = gear_down
        
        # Cambio automatico
        if not gear_up and not gear_down:
            if rpm > 7500 and 0 < self.state['gear'] < 6:
                self.state['gear'] += 1
            elif rpm < 3000 and self.state['gear'] > 1:
                self.state['gear'] -= 1
            
            # Retromarcia e prima marcia automatiche da fermo
            if speed < 2.0 and brake_axis > 0.8 and self.state['gear'] >= 1:
                self.state['gear'] = -1
            elif self.state['gear'] == -1 and accel_axis > 0.5 and speed > -2.0:
                self.state['gear'] = 1

        # Dead zone
        if abs(steer_axis) < 0.1:
            steer_axis = 0.0

        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_target = steer_axis * max_steer
        
        # Smooth steer per non strappare, e assegnazioni dirette per accel/freno
        self.state['steer'] += (steer_target - self.state['steer']) * 0.2
        self.state['accel'] = accel_axis
        self.state['brake'] = brake_axis
        
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['gear'] = max(-1, min(6, self.state['gear']))

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ControllerAgent()

    client.get_servers_input()

    print("=========================================================")
    print(" Modalita' Controller Attiva ")
    print(" Levetta SX  : Sterzo")
    print(" RT / Tasto A: Acceleratore")
    print(" LT / Tasto B: Freno")
    print(" RB / Tasto Y: Marcia Su")
    print(" LB / Tasto X: Marcia Giu")
    print("=========================================================")

    # CSV log
    log_csv = open("controller_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    log_json = []
    t0 = time.time()
    step = 0

    while True:
        S = client.S.d

        controller.update(S)
        a = controller.state
        
        print(f"steer={a['steer']:5.2f} accel={a['accel']:5.2f} brake={a['brake']:5.2f} gear={a['gear']}", end='\r')

        client.R.d['steer'] = a['steer']
        client.R.d['accel'] = a['accel']
        client.R.d['brake'] = a['brake']
        client.R.d['gear'] = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()
        client.get_servers_input()

        current_time = time.time() - t0

        log_csv.write(
            f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
            f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
            f"{S.get('rpm',0)},{S.get('damage',0)}\n"
        )

        log_json.append({
            "step": step,
            "time": current_time,
            "action": {
                "steer": a['steer'],
                "accel": a['accel'],
                "brake": a['brake'],
                "gear": a['gear']
            },
            "state": {
                "speedX": S.get('speedX', 0),
                "trackPos": S.get('trackPos', 0),
                "angle": S.get('angle', 0),
                "rpm": S.get('rpm', 0),
                "damage": S.get('damage', 0)
            }
        })

        step += 1

        if step % 100 == 0:
            with open("controller_log.json", "w") as f:
                json.dump(log_json, f, indent=2)
                
        time.sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUscita.")
