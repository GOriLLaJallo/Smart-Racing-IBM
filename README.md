# AI TORCS Driver – Smart Racing IBM

## AI Driver basato su Machine Learning (k-NN) per TORCS

---

## 📖 Descrizione

Progetto sviluppato nell'ambito della **IBM AI Racing League** per la realizzazione di un agente di guida autonoma nel simulatore **TORCS (The Open Racing Car Simulator)**.

*Smart Racing IBM* è un progetto di Intelligenza Artificiale Applicata e Guida Autonoma. Sfrutta il *Machine Learning* (algoritmo K-Nearest Neighbors) per insegnare a un agente virtuale a guidare in modo autonomo all'interno del simulatore di corse open-source *TORCS* (The Open Racing Car Simulator) in un circuito complesso (Laguna Seca) con l'obiettivo di minimizzare il tempo su un singolo giro partendo da fermo e massimizzando la stabilità.

---

## 👥 Team

* Menza William
* Sostegno Vincenzo Nicola
* Argenio Letizia
* Amabile Emilia
* Vitolo Teresa

---

## 🚀 Architettura del Modello: K-Nearest Neighbors (k-NN)

Il cuore del progetto è un algoritmo puramente statistico e geometrico implementato tramite *scikit-learn*. L'agente confronta costantemente i sensori in tempo reale dell'auto con un database di migliaia di frame guidati da un umano, cercando le *k* situazioni storiche "più simili" per dedurre istantaneamente sterzo, freno e acceleratore.

Il sistema è composto da due componenti principali:

### 1. Modulo di Machine Learning

Responsabile della previsione delle azioni di guida:

* Sterzo (Steering)
* Acceleratore (Throttle)
* Freno (Brake)

### 2. Modulo Deterministico

Responsabile della gestione del cambio marcia:

* Selezione della marcia ottimale
* Controllo delle soglie RPM
* Isteresi anti oscillazione

---

### ✨ Funzionalità Avanzate dell'Agente (drive_agent.py)
* *Ponderazione dinamica dei sensori (Custom Weights):* Abbiamo applicato dei pesi dinamici personalizzati per la ricerca k-NN (knn_custom_weights.npy), istruendo l'algoritmo a dare altissima priorità alla modulazione della velocità, alla posizione laterale in pista (trackPos) e all'angolo dell'auto.
* *Guardrail Virtuale:* L'agente include routine di sicurezza anti-uscita, correggendo automaticamente la traiettoria qualora un'imperfezione del modello lo portasse sull'erba.
* *Cambio automatico intelligente:* separato dal modello di Machine Learning, implementato tramite logiche deterministiche e isteresi per evitare il fenomeno del gear hunting.

---

### ⚙️ Moduli Condivisi
* *gearing.py*: Un cambio automatico deterministico intelligente. Sostituisce le cambiate della rete neurale (soggette a "hunting" e micro-oscillazioni) con una logica a isteresi basata su velocità, giri motore e percentuale di accelerazione per garantire il 100% di affidabilità meccanica.
* *data_collection.py*: Un sistema di acquisizione dati a 50Hz in formato HDF5. Legge gli input del controller PS5/Tastiera, integra un *Traction Control System (TCS) per limare gli errori umani, e scarta automaticamente i giri invalidati da tagli di curva.

---

## 📂 Struttura del Repository

```text
smart_racing_ibm/
│
├── train_set/             # Dataset e telemetria in formato HDF5
├── vtorcs-RL-color/       # Client del simulatore TORCS
│
├── data_collection.py     # Acquisizione dati e TCS
├── train_knn.py           # Pipeline di training e generazione modello
├── drive_agent.py         # Logica principale di guida autonoma
├── gearing.py             # Gestione intelligente del cambio
├── snakeoil3_gym.py       # Client UDP TORCS
├── gym_torcs.py           # Wrapper ambiente Gym
│
└── README.md
```
---

## 🛠️ Requisiti di Sistema

- *OS:* Windows / Linux (Testato in ambiente compatibile TORCS)
- *Simulatore:* vtorcs-RL-color (TORCS con patch per Reinforcement Learning / UDP)
- *Python:* 3.8+
- *Librerie Principali:*
  - numpy, h5py (Gestione dataset)
  - pygame (Polling del controller PS5/Input)
  - scikit-learn, joblib (Modello k-NN e Scaler)
  - torch (PyTorch per la rete neurale BC)

---

## 🔄 Pipeline di Machine Learning

L'intero workflow è composto da cinque fasi principali.

### 1. Raccolta Dati

Attraverso `data_collection.py` vengono acquisiti:

* Sensori di pista
* Velocità
* Angolo rispetto alla pista
* Posizione relativa
* Input del pilota

La raccolta avviene a circa **50 Hz**.

---

### 2. Preprocessing

I dati vengono:

* Normalizzati tramite **StandardScaler (Z-Score)**
* Filtrati
* Pesati mediante una matrice di **Custom Weights**

L'obiettivo è enfatizzare le feature più importanti per il controllo della traiettoria.

---

### 3. Training

Durante questa fase:

1. Vengono caricati i dataset validati.
2. Si applica la normalizzazione.
3. Viene addestrato il modello k-NN.

---

### 4. Export

Il modello viene serializzato tramite:

* `.pkl` per il modello KNN
* `.npy` per i pesi personalizzati
* scaler serializzato tramite Joblib

---

### 5. Inferenza

Durante la guida autonoma:

1. Viene acquisito lo stato corrente del veicolo.
2. I dati vengono normalizzati.
3. Si applicano i custom weights.
4. Il modello ricerca i **5 vicini più simili**.
5. Viene prodotta la previsione delle azioni.

---

## 🧠 Modello k-NN

### Tipo

Instance-Based Learning

### Algoritmo

K-Nearest Neighbors Regressor

### Input

Feature provenienti dalla telemetria TORCS:

* Track Sensors
* Angle
* Track Position
* Speed X

### Output

* Steering
* Acceleration
* Brake

### Parametri principali

```python
n_neighbors = 5
weights = "distance"
```

Il cambio marcia è escluso dal modello e viene gestito separatamente.

---

## 📈 Prestazioni

Caratteristiche osservate durante i test:

* guida fluida e stabile;
* buona generalizzazione sui circuiti addestrati;
* tempi sul giro competitivi;
* elevata robustezza nelle sezioni tecnicamente difficili.

---

## 🚀 Esecuzione

### Raccolta dati

```bash
python data_collection.py
```

### Training del modello

```bash
python train_knn.py
```

### Avvio della guida autonoma

```bash
python drive_agent.py
```

---

## 📚 Riferimenti

* TORCS Official Documentation
* Gym-TORCS
* IBM SkillsBuild
* IBM AI Racing League

---