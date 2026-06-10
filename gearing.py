"""
gearing.py — Cambio marcia DETERMINISTICO (velocità-primario, anti-hunting).

Questo modulo sostituisce la logica di cambio marcia basata su Rete Neurale (che soffriva 
di "gear hunting", ovvero continue oscillazioni tra marce, fino a 322 cambi ogni 1000 step) 
con un approccio algoritmico e deterministico.

Il problema classico: il downshift in staccata fa salire gli RPM per il freno motore. Un sistema 
basato solo sugli RPM legge il picco e sale di nuovo di marcia, creando un'oscillazione infinita.

La soluzione implementata qui:
  1. DOWNSHIFT (Scalata): Si basa SOLO sulla velocità (che in frenata scende sempre), ignorando i picchi di RPM.
  2. UPSHIFT (Salita): Avviene solo se l'acceleratore è premuto e i giri sono alti. In staccata (gas a 0), l'upshift è bloccato.
  3. Isteresi: Le soglie di salita sono più alte di quelle di scalata per evitare jitter (oscillazioni al limite di velocità).
"""

# ==============================================================================
# SOGLIE E COSTANTI GLOBALI
# ==============================================================================

# Soglie di velocità (km/h) MINIME per SALIRE di marcia: 
# Es: da 1ª a 2ª oltre i 55 km/h, da 2ª a 3ª oltre i 118 km/h, ecc.
UP_SPEED = [55.0, 118.0, 200.0, 258.0, 286.0]

# Soglie di velocità (km/h) MASSIME per SCENDERE di marcia:
# Es: da 2ª a 1ª sotto i 40 km/h. 
# L'isteresi (DN_SPEED < UP_SPEED) crea una "zona morta" che impedisce cambi continui in crociera.
DN_SPEED = [40.0, 92.0, 165.0, 232.0, 272.0]

# Giri motore minimi per permettere l'upshift (evita di "tirare corto" cambiando marcia troppo presto).
UP_RPM_GATE = 15500.0   

# Pressione minima sull'acceleratore (da 0.0 a 1.0) per permettere l'upshift.
# Questa è la chiave anti-hunting in staccata: se sto frenando o veleggiando, non salgo di marcia.
UP_ACCEL_GATE = 0.4     

# Step minimi (tick del simulatore) di attesa obbligatoria dopo un cambio marcia.
# Impedisce raffiche di cambi ("mitragliatrice") e dà tempo alla fisica di stabilizzarsi.
SHIFT_COOLDOWN = 5      

# ==============================================================================
# LOGICA PRINCIPALE
# ==============================================================================

def compute_gear(speed_kmh: float, accel: float, rpm: float, current_gear: int, steps_since_shift: int):
    """
    Calcola la marcia ideale in base alla telemetria attuale dell'auto.
    Cambia al massimo di ±1 marcia per chiamata.

    Args:
        speed_kmh: Velocità longitudinale in km/h (tipicamente obs['speedX'] * 50 in TORCS).
        accel: Pressione sul pedale dell'acceleratore [0.0, 1.0].
        rpm: Giri motore grezzi (obs['rpm']).
        current_gear: La marcia attualmente inserita (1-6).
        steps_since_shift: Contatore di step trascorsi dall'ultimo cambio marcia.

    Returns:
        Tuple[int, bool]: 
            - La marcia calcolata (da 1 a 6).
            - Un flag booleano (True se è avvenuta una cambiata, False altrimenti).
    """
    g = int(current_gear)
    
    # 1. Sanity Check: la marcia non può mai scendere sotto la 1ª (niente folle/retro).
    if g < 1:
        g = 1
        
    # 2. Cooldown Lock: se abbiamo appena cambiato marcia, blocchiamo l'esecuzione
    # restituendo la marcia attuale senza modifiche.
    if steps_since_shift < SHIFT_COOLDOWN:
        return g, False

    # 3. UPSHIFT (Salita di marcia):
    # - g < 6: Non siamo già in 6ª marcia.
    # - speed_kmh > UP_SPEED: Abbiamo superato la velocità minima per la marcia successiva.
    # - accel > UP_ACCEL_GATE: Stiamo premendo sull'acceleratore.
    # - rpm > UP_RPM_GATE: Il motore è sufficientemente su di giri.
    if g < 6 and speed_kmh > UP_SPEED[g - 1] and accel > UP_ACCEL_GATE and rpm > UP_RPM_GATE:
        return g + 1, True

    # 4. DOWNSHIFT (Scalata):
    # - g > 1: Non siamo già in 1ª marcia.
    # - speed_kmh < DN_SPEED: La velocità è scesa sotto la soglia massima per tenere questa marcia.
    if g > 1 and speed_kmh < DN_SPEED[g - 2]:
        return g - 1, True

    # 5. Nessun cambio richiesto: mantiene la marcia attuale.
    return g, False