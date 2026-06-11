# 🏎️ Smart Racing IBM

## AI Driver basato su Machine Learning (k-NN) per TORCS

---

## 📖 Descrizione

Progetto sviluppato nell'ambito della **IBM AI Racing League** per la realizzazione di un agente di guida autonoma nel simulatore **TORCS (The Open Racing Car Simulator)**.

**Smart Racing IBM** è un progetto di **Intelligenza Artificiale Applicata** e **Guida Autonoma** che sfrutta il **Machine Learning** per insegnare a un agente virtuale a guidare autonomamente all'interno del simulatore open-source TORCS.

L'obiettivo principale è:

* 🏁 **Minimizzare il tempo sul giro**
* 🚗 **Massimizzare la stabilità del veicolo**
* 🎯 **Mantenere una traiettoria ottimale**
* ⚡ **Gestire in modo efficiente accelerazione, frenata e sterzo**

Il progetto è stato addestrato sul circuito **Laguna Seca**, utilizzando dati raccolti da sessioni di guida umana.

---

## 👥 Team

| Nome                     |             
| ------------------------ | 
| William Menza            | 
| Vincenzo Nicola Sostegno | 
| Letizia Argenio          | 
| Emilia Amabile           | 
| Teresa Vitolo            | 

---

# 🚀 Architettura del Sistema

L'agente utilizza un approccio **Behavior Cloning tramite K-Nearest Neighbors (k-NN)** implementato con **scikit-learn**.

Ad ogni istante il sistema confronta i sensori correnti dell'auto con migliaia di situazioni storiche registrate durante la guida umana, individuando gli esempi più simili e stimando le azioni da eseguire.

## Componenti Principali

### 🧠 1. Modulo di Machine Learning

Responsabile della previsione delle azioni di guida:

* Sterzo (**Steering**)
* Acceleratore (**Throttle**)
* Freno (**Brake**)

---

### ⚙️ 2. Modulo Deterministico

Responsabile della gestione della trasmissione:

* Selezione della marcia ottimale
* Controllo delle soglie RPM
* Isteresi anti-oscillazione
* Riduzione del fenomeno di *gear hunting*

---

## ✨ Funzionalità Avanzate dell'Agente (`drive_agent.py`)

### 🎯 Ponderazione Dinamica dei Sensori

L'agente utilizza una matrice di pesi personalizzati (`knn_custom_weights.npy`) per enfatizzare le feature più importanti durante la ricerca dei vicini.

Particolare attenzione viene data a:

* Velocità del veicolo
* Posizione laterale in pista (*trackPos*)
* Angolo rispetto alla direzione della pista

---

### 🛡️ Guardrail Virtuale

Sistema di sicurezza integrato che monitora costantemente la posizione dell'auto.

Quando il modello tende a portare il veicolo fuori pista:

* vengono applicate correzioni automatiche;
* viene favorita la riconvergenza verso il centro della carreggiata;
* si riduce il rischio di perdita di controllo.

---

### 🔄 Cambio Automatico Intelligente

Il cambio marcia è completamente separato dal modello di Machine Learning.

Una logica deterministica basata su:

* RPM motore
* Velocità
* Percentuale di acceleratore

garantisce cambiate affidabili e prive di oscillazioni indesiderate.

---

## ⚙️ Moduli Condivisi

### `gearing.py`

Sistema di cambio automatico intelligente.

Caratteristiche:

* isteresi anti-oscillazione;
* controllo RPM;
* gestione affidabile delle cambiate;
* eliminazione del *gear hunting*.

---

### `data_collection.py`

Sistema di acquisizione dati ad alta frequenza (~50 Hz).

Funzionalità:

* acquisizione input PS5 Controller / Tastiera;
* registrazione telemetria in formato HDF5;
* integrazione di un **Traction Control System (TCS)**;
* eliminazione automatica dei giri invalidati da tagli di curva.

---

# 📂 Struttura del Repository

```text
smart_racing_ibm/
│
├── train_set/             # Dataset e telemetria in formato HDF5
├── vtorcs-RL-color/       # Client TORCS modificato per RL
│
├── data_collection.py     # Acquisizione dati e TCS
├── train_knn.py           # Training del modello
├── drive_agent.py         # Guida autonoma
├── gearing.py             # Cambio automatico intelligente
├── snakeoil3_gym.py       # Client UDP TORCS
├── gym_torcs.py           # Wrapper OpenAI Gym
│
└── README.md
```

---

# 🛠️ Requisiti di Sistema

## Ambiente

* **OS:** Windows / Linux
* **Python:** 3.8+
* **Simulatore:** vtorcs-RL-color

---

## Librerie Principali

```bash
numpy
h5py
pygame
scikit-learn
joblib
torch
```

### Utilizzo

| Libreria     | Scopo                   |
| ------------ | ----------------------- |
| NumPy        | Elaborazione numerica   |
| HDF5         | Gestione dataset        |
| Pygame       | Acquisizione input      |
| Scikit-Learn | Modello k-NN            |
| Joblib       | Serializzazione         |
| PyTorch      | Modelli sperimentali BC |

---

# 🔄 Pipeline di Machine Learning

L'intero workflow è composto da cinque fasi.

## 1️⃣ Raccolta Dati

Attraverso `data_collection.py` vengono acquisiti:

* Track Sensors
* Speed X
* Angle
* Track Position
* Input del pilota

Frequenza di acquisizione:

> **≈ 50 Hz**

---

## 2️⃣ Preprocessing

Operazioni eseguite:

* Normalizzazione tramite **StandardScaler**
* Filtraggio dei dati
* Applicazione dei **Custom Weights**

Obiettivo:

> dare maggiore importanza alle feature più rilevanti per il controllo della traiettoria.

---

## 3️⃣ Training

Durante questa fase:

1. Caricamento dei dataset validati.
2. Normalizzazione delle feature.
3. Addestramento del modello k-NN.
4. Validazione delle prestazioni.

---

## 4️⃣ Export

Artefatti generati:

| File   | Descrizione         |
| ------ | ------------------- |
| `.pkl` | Modello k-NN        |
| `.npy` | Pesi personalizzati |
| Joblib | Scaler normalizzato |

---

## 5️⃣ Inferenza

Durante la guida autonoma:

1. Acquisizione dello stato corrente.
2. Normalizzazione delle feature.
3. Applicazione dei custom weights.
4. Ricerca dei **5 vicini più simili**.
5. Predizione dell'azione ottimale.

---

# 🧠 Modello k-NN

## Tipo

**Instance-Based Learning**

---

## Algoritmo

**K-Nearest Neighbors Regressor**

---

## Input

Feature telemetriche provenienti da TORCS:

* Track Sensors
* Angle
* Track Position
* Speed X

---

## Output

* Steering
* Acceleration
* Brake

---

## Parametri Principali

```python
n_neighbors = 5
weights = "distance"
```

> Il cambio marcia non viene predetto dal modello ma è gestito separatamente dal modulo deterministico.

---

# 📈 Prestazioni

Durante i test l'agente ha mostrato:

* ✅ guida fluida e stabile;
* ✅ buona generalizzazione sui circuiti addestrati;
* ✅ tempi sul giro competitivi;
* ✅ elevata robustezza nelle sezioni tecnicamente più difficili;
* ✅ comportamento coerente anche in presenza di piccole deviazioni dalla traiettoria ideale.

---

# 🚀 Esecuzione

## Raccolta Dati

```bash
python data_collection.py
```

---

## Training del Modello

```bash
python train_knn.py
```

---

## Avvio della Guida Autonoma

```bash
python drive_agent.py
```

---

# 📚 Riferimenti

* TORCS Official Documentation
* Gym-TORCS
* IBM SkillsBuild
* IBM AI Racing League

---

<div align="center">

### 🏁 Smart Racing IBM

Machine Learning • Autonomous Driving • TORCS • IBM AI Racing League

</div>
