# 📈 DIPLOMSKA NALOGA: Napovedovanje gibanja finančnih trgov z modeli Chronos in TimesFM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Opis projekta

Ta projekt primerja dva napredna modela za napovedovanje časovnih vrst:
- **Amazon Chronos-Bolt** 
- **Google TimesFM**

Analiza temelji na zgodovinskih podatkih o cenah izbranih delnic in vključuje:
- ✅ Oceno uspešnosti modelov na različnih časovnih intervalih (5min, 15min, 1h)
- ✅ Vizualizacijo rezultatov
- ✅ Statistično obdelavo in primerjavo

## 📁 Struktura projekta

```
diploma/
├── 📓 notebooks/          # Jupyter notebooki za analizo
├── 🐍 src/               # Python skripte
├── 📊 data/              # Zgodovinski podatki delnic
│   ├── 5M/              # 5-minutni intervali
│   ├── 15M/             # 15-minutni intervali
│   └── 1H/              # 1-urni intervali
├── 📈 results/           # Napovedi modelov
├── 📋 analysis_results/  # Grafični prikazi in primerjave
└── 📝 notes.md          # Povzetki in ugotovitve
```

## 📊 Uporabljeni podatki

| Karakteristika | Opis |
|---|---|
| **Delnice** | 11 izbranih delnic (IONQ, NVDA, TSLA, VKTX, ...) |
| **Borze** | NASDAQ in NYSE |
| **Intervali** | 5M, 15M, 1H |
| **Metrika** | Closing cena ('Close') |
| **Obdobje** | Zgodovinski podatki |

## 🤖 Uporabljeni modeli

### 🔥 Chronos-Bolt (Amazon)
- **Tip**: Foundation model za časovne vrste (transformer)
- **Context window**: 128 preteklih vrednosti → 20 napovedi
- **Metoda**: Drseče okno (rolling window)

### ⏰ TimesFM (Google)
- **Tip**: Foundation model za časovne vrste  (transformer)
- **Context window**: 2048 (128) preteklih vrednosti → 20 napovedi
- **Metoda**: Drseče okno (rolling window)

## 📈 Analiza in vizualizacija

### Metrični kazalniki
- **MAPE** (Mean Absolute Percentage Error)
- **MAE** (Mean Absolute Error) 
- **RMSE** (Root Mean Square Error)
- **Volatilnost napake**
- **Smerna pravilnost/natančnost**
- **Random Walk**

### Vizualizacije
- 📊 Histogrami
- 📦 Boxploti
- 📉 Distribucije
- 📊 Stolpčni diagrami

### Statistični testi
- 🔬 T-test za preverjanje pomembnosti razlik med modeli

## 🚀 Navodila za uporabo

### Predpogoji
```bash
pip install -r requirements.txt
```

### Zagon analize
1. **Pridobivanje podatkov**: `notebooks/Data_acquistion.ipynb`
2. **Chronos napovedi**: `notebooks/chronos_predictions_5M_15M.ipynb`
3. **TimesFM napovedi**: `notebooks/timesFM_model_5M_15M.ipynb`
4. **Analiza rezultatov**: `src/analysis_scripts.py`

## 🔍 Ključne ugotovitve

> **Napaka v besedilu**: "Model Chronos je v povprečju natančnejši kot Chronos" → Verjetno mislite "Model TimesFM je natančnejši kot Chronos"

### ✅ Glavne ugotovitve
- **TimesFM** dosega v povprečju boljšo natančnost (nižji MAPE) in večjo stabilnost
- **Optimalni intervali**: 5min in 15min za oba modela
- **Problematični interval**: 1h (največje napake in volatilnost)
- **Direkcionalna točnost**: Podobna za oba modela, presega naključno napovedovanje

### 📊 Primerjava po intervalih
| Interval | Chronos | TimesFM | Priporočilo |
|----------|---------|---------|-------------|
| 5min     | ✅ Dobro | ✅ Odlično | TimesFM |
| 15min    | ✅ Dobro | ✅ Odlično | TimesFM |
| 1h       | ⚠️ Slabše | ⚠️ Slabše | Potrebne izboljšave |

## 💡 Priporočila

### 🎯 Za praktično uporabo
- **Splošno napovedovanje**: TimesFM na 5min-15min intervalih
- **Kratkoročno trgovanje**: TimesFM z 5min intervalom
- **Dolgoročna analiza**: Potrebne dodatne izboljšave za 1h interval

### ⚠️ Omejitve
- Modeli niso optimizirani za visoko-frekvenčno trgovanje
- 1h interval zahteva dodatno fino nastavljanje
- Potrebno je upoštevanje tržnih pogojev in volatilnosti

---

## 📧 Kontakt

**Avtor**: Michael Valand 
**Email**:  
**Leto**: 2025


