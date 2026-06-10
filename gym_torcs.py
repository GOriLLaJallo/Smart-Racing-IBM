
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time

# ======================================================================
#  FUNZIONE TRASFERITA QUI PER EVITARE L'IMPORT CIRCOLARE
# ======================================================================
def flatten_state(state_dict: dict) -> np.ndarray:
    def _scalar(key: str, default: float = 0.0) -> float:
        val = state_dict.get(key, default)
        if val is None:
            return default
        if isinstance(val, np.ndarray):
            return float(val.flat[0])
        return float(val)

    def _array(key: str, size: int) -> np.ndarray:
        val = state_dict.get(key, None)
        if val is None:
            return np.zeros(size, dtype=np.float32)
        arr = np.array(val, dtype=np.float32).flatten()
        if arr.shape[0] != size:
            padded = np.zeros(size, dtype=np.float32)
            padded[:min(size, arr.shape[0])] = arr[:min(size, arr.shape[0])]
            return padded
        return arr

    try:
        state_vec = np.concatenate([
            np.array([_scalar('angle')]),
            _array('track', 19),
            np.array([_scalar('trackPos')]),
            np.array([_scalar('speedX')]),
            np.array([_scalar('speedY')]),
            np.array([_scalar('speedZ')]),
            _array('wheelSpinVel', 4) / 100.0,
            np.array([_scalar('rpm') / 10000.0]),
        ])
        return state_vec.astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)

class TorcsEnv(gym.Env):
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            # === NUOVA LOGICA MONOPEDALE (GAS/FRENO) ===
            pedal_input = this_action['accel'] # Che ora va da -1.0 a +1.0
            
            if pedal_input > 0:
                action_torcs['accel'] = pedal_input
                action_torcs['brake'] = 0.0
            else:
                action_torcs['accel'] = 0.0
                action_torcs['brake'] = abs(pedal_input)

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            """
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
            """

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = flatten_state(obs)

        # Reward setting Here #######################################
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        angle = obs['angle']
        
        # Recuperiamo la posizione e lo sterzo per le penalità avanzate
        track_pos = obs['trackPos']
        current_steer = action_torcs['steer']

        # 1. PREMIO BASE (Progresso lungo la pista)
        # Se vai dritto a 100 km/h, guadagni 100 punti. Se sbandi, il coseno riduce i punti.
        progress = sp * np.cos(angle) 
        reward = progress

        # 2. PENALITÀ LATERALE (Permette la Traiettoria Ideale)
        # La pista va da -1 a +1. Creiamo una "zona franca" sicura tra -0.8 e +0.8.
        # L'auto è libera di allargarsi per impostare le curve.
        MARGINE_SICURO = 0.8
        
        if abs(track_pos) > MARGINE_SICURO:
            # Calcoliamo di quanto ha superato il margine sicuro (es. è a 0.9 -> invasione = 0.1)
            invasione = abs(track_pos) - MARGINE_SICURO
            # Punizione esponenziale in base alla velocità
            eccentricity_penalty = (invasione ** 2) * (sp * 0.5)
            reward -= eccentricity_penalty

        # 3. PENALITÀ SMOOTHNESS (Niente sterzate a zig-zag)
        # Scoraggia le sterzate violente ad alte velocità per stabilizzare la guida
        steer_penalty = (abs(current_steer) ** 2) * (sp * 0.05)
        reward -= steer_penalty

        # 4. COLLISION DETECTION (Il Muro di Gomma)
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -500.0  # PRIMA ERA -1. Ora è una punizione severissima.

        # Termination judgement #########################
        episode_terminate = False
        
        # 5. FUORI PISTA FATALE
        if track.min() < 0:  
            reward = -500.0  # PRIMA ERA -1.
            episode_terminate = True
            client.R.d['meta'] = True

        # 6. TROPPO LENTO O FERMO
        if self.terminal_judge_start < self.time_step: 
            if progress < self.termination_limit_progress:
                reward = -50.0  # Punizione per essersi arreso/fermato
                episode_terminate = True
                client.R.d['meta'] = True

        # 7. CONTROMANO
        if np.cos(angle) < 0: 
            reward = -500.0
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), float(reward), bool(client.R.d['meta']), False, {}

    def reset(self, seed=None, options=None, relaunch=False, **kwargs):
        super().reset(seed=seed)
        
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = flatten_state(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs(), {}
    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
