# 🏎️ AI TORCS Driver – Smart Racing IBM

### Autonomous Racing Agent based on Machine Learning (k-NN)

---

## 📖 Overview

**Smart Racing IBM** è un progetto di **Intelligenza Artificiale Applicata** sviluppato nell'ambito della **IBM AI Racing League**.

L'obiettivo è progettare un agente di guida autonoma capace di competere nel simulatore **TORCS (The Open Racing Car Simulator)** apprendendo il comportamento di un pilota umano e replicandolo in tempo reale tramite tecniche di **Imitation Learning**.

> 🎯 **Obiettivo principale**
>
> Completare un giro sul circuito di **Laguna Seca** nel minor tempo possibile, mantenendo al contempo elevati livelli di stabilità e precisione di guida.

---

## 👥 Team

| Nome                     | Ruolo       |
| ------------------------ | ----------- |
| William Menza            | Team Member |
| Vincenzo Nicola Sostegno | Team Member |
| Letizia Argenio          | Team Member |
| Emilia Amabile           | Team Member |
| Teresa Vitolo            | Team Member |

---

# 🧠 Architettura del Sistema

L'agente utilizza un approccio di **Instance-Based Learning** basato su **K-Nearest Neighbors Regression**.

Ad ogni frame di simulazione:

1. Viene acquisito lo stato corrente del veicolo.
2. I dati vengono normalizzati e pesati.
3. Il modello ricerca i **5 stati più simili** presenti nel dataset.
4. Vengono generate le azioni di controllo:

   * Steering
   * Throttle
   * Brake

Il cambio marcia viene invece gestito da un modulo separato e deterministico.

---

## ⚙️ Componenti Principali

### 🤖 Modulo Machine Learning

Responsabile della previsione delle azioni di guida.

**Output prodotti:**

* Steering
* Acceleration
* Brake

---

### ⚡ Modulo Deterministico

Responsabile della gestione della trasmissione.

**Funzionalità:**

* Cambio automatico intelligente
* Controllo RPM
* Isteresi anti-oscillazione
* Eliminazione del gear hunting

---

# ✨ Funzionalità Avanzate

## 🎯 Sensor Weighting

Per migliorare la qualità della ricerca k-NN sono stati introdotti **Custom Weights** applicati alle feature più importanti.

Le variabili maggiormente valorizzate sono:

* Track Position
* Vehicle Angle
* Speed X
* Sensori frontali della pista

Questo consente al modello di privilegiare il mantenimento della traiettoria rispetto a caratteristiche meno rilevanti.

---

## 🛡️ Virtual Guardrail

Sistema di sicurezza progettato per limitare le uscite di pista.

Quando il veicolo entra in una situazione critica:

* viene applicata una correzione dello sterzo;
* viene ridotta la velocità;
* viene favorita la convergenza verso il centro pista.

---

## ⚙️ Intelligent Gearbox

Il modulo `gearing.py` implementa un cambio automatico basato su logiche deterministiche.

### Vantaggi

✅ Nessun gear hunting

✅ Maggiore stabilità in curva

✅ Cambiate più realistiche

✅ Affidabilità meccanica costante

---

# 📊 Modello k-NN

| Parametro | Valore                                      |
| --------- | ------------------------------------------- |
| Algoritmo | KNeighborsRegressor                         |
| Neighbors | 5                                           |
| Weights   | Distance                                    |
| Features  | Track Sensors, Angle, Track Position, Speed |
| Outputs   | Steering, Throttle, Brake                   |

```python
n_neighbors = 5
weights = "distance"
```

> Con `weights="distance"` i vicini più simili influenzano maggiormente la previsione rispetto a quelli più lontani, migliorando la precisione del controllo.
