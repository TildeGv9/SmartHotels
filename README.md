### Tema n. 5 Machine Learning per Processi Aziendali

### Traccia del PW 20 Smistamento recensioni hotel e analisi sentimento con Machine Learning:

## Titolo progetto: ""

## Nome applicativo: Smart Hotels

### Obiettivi

1. Creare un dataset sintetico (200–500 recensioni brevi) con etichette: reparto (Housekeeping, Reception, F&B) e sentiment (pos/neg).
2. Addestrare modelli di base per la classificazione ed analisi di sentimento.
3. Valutare con train/test split (80/20), accuracy e F1 per classe; matrice di confusione semplice.
4. Interfaccia per incollare una recensione e ottenere reparto consigliato e sentiment stimato.

### Caratteristiche Innovative

- **Dataset Configurabile:** Parametro `RANDOMNESS` per controllare la difficoltà del problema
- **Analisi del Paradosso:** Scoperta che il sentiment analysis è più robusto al rumore rispetto alla classificazione per reparto
- **Visualizzazioni Avanzate:** Grafici interattivi per comprendere il comportamento dei modelli
- **Esperimenti Automatici:** Framework per testare sistematicamente diverse configurazioni

Il risultato è un **sistema di studio completo** che simula realisticamente le sfide del machine learning in contesti aziendali reali.

## Struttura del Repository

La cartella del progetto è organizzata come segue:

```
SmartHotels/
├── dataset_generator.py           # 1. Genera il dataset con parametro RANDOMNESS.
├── ml_pipeline_sklearn.py         # 2. Addestra, valuta e salva i modelli (legge da data/sklearn/reviews/)
├── dashboard_app.py               # 3. Interfaccia utente interattiva (Streamlit).
├── visualization_analysis.py      # 4. Grafici e analisi del parametro RANDOMNESS.
├── run_experiments.py             # 5. Esperimenti automatici multi-RANDOMNESS.
├── dependencies.txt               # Dipendenze Python.
├── README.md
├── data/                          # Dataset CSV e risultati delle predizioni.
│   ├── sklearn/
│   │   ├── reviews/               # Dataset per pipeline sklearn
│   │   └── predictions/           # Predizioni sklearn con suffisso _R{R}_sklearn
│   └── pytorch/
│       ├── reviews/               # Dataset per pipeline pytorch
│       └── predictions/           # Predizioni pytorch con suffisso _R{R}_pytorch
├── models/                        # Modelli ML salvati (.joblib e .pth).
├── plots/                         # Grafici e visualizzazioni generate.
│   ├── sklearn/                   # Grafici generati dalla pipeline sklearn
│   │   ├── accuracy_comparison_sklearn.png
│   │   ├── paradox_visualization_sklearn.png
│   │   ├── f1_heatmap_sklearn.png
│   │   ├── dashboard_complete_sklearn.png
│   │   └── current_performance_sklearn.png
│   └── pytorch/                   # Grafici generati dalla pipeline pytorch
│       ├── accuracy_comparison_pytorch.png
│       ├── paradox_visualization_pytorch.png
│       ├── f1_heatmap_pytorch.png
│       ├── dashboard_complete_pytorch.png
│       └── current_performance_pytorch.png
└── experiments/                   # Risultati esperimenti RANDOMNESS.
    └── randomness_experiment_*.json
```

## Istruzioni per l'Esecuzione

Seguire questi passaggi per configurare ed eseguire il progetto.

### 1. Setup dell'Ambiente

# Creare ambiente virtuale (una volta)

```bash
python -m venv venv
```

# Attivare l'ambiente virtuale (ogni volta che si vuole avviare il progetto)

# Su Windows:

```bash
 .\venv\Scripts\activate
```

# Su macOS/Linux:

```bash
source venv/bin/activate
```

#### Requisiti Sistema:

- **Python 3.8+**
- **Pacchetti principali:** pandas, scikit-learn, streamlit, matplotlib, seaborn
- **Installa le dipendenze** necessarie:

  ```bash
  pip install -r dependencies.txt
  ```

### 2. Generazione del Dataset

```bash
   python dataset_generator.py
```

Questo script crea un dataset bilanciato con **complessità configurabile** tramite il parametro `RANDOMNESS`.

## Modifica del Parametro RANDOMNESS

Per cambiare la difficoltà del dataset, modifica la costante nel file `dataset_generator.py`

#### Configurazioni Consigliate:

- **RANDOMNESS = 0.2:** Dataset realistico per produzione (~76% accuracy)
- **RANDOMNESS = 0.4:** Dataset challenging per testing (~57% accuracy)
- **RANDOMNESS = 0.9:** Dataset estremo per studio robustezza (~31% reparto, ~93% sentiment)

_Output atteso:_ Creazione del file `data/hotel_reviews_synthetic_{RANDOMNESS}.csv` (es. `data/hotel_reviews_synthetic_0.9.csv`) che conterrà una lista di recensioni con i relativi reparti e sentiment.

### 3. Addestramento e Valutazione ML

```bash
python ml_pipeline_sklearn.py
```

Questo script (scikit-learn) esegue il preprocessing, addestra i due classificatori, valuta le performance e genera matrici e grafici in base al risultato dell'addestramento.

**Output:**

- **Metriche dettagliate** (Accuracy, F1-Macro, Classification Report)
- **Matrici di Confusione**
- **Grafico performance** salvato in `plots/sklearn/current_performance_sklearn.png`
- **Modelli salvati** in `models/` (file `.joblib` e `.pth`)
- **Risultati test** in `data/sklearn/predictions/predictions_YYYYMMDD_HHMMSS_R{R}_sklearn.csv`
- **Analisi errori** con esempi di classificazioni errate

### 4. Avvio della Dashboard Interattiva

L'interfaccia utente ti permette di testare i modelli in tempo reale e di eseguire predizioni in batch. Dalla sidebar è possibile scegliere la pipeline da utilizzare (scikit-learn o PyTorch).

1. **Avvia l'applicazione Streamlit:**

   ```bash
   streamlit run dashboard_app.py
   ```

2. Il browser si aprirà automaticamente su `http://localhost:8501`.

## Funzionalità Avanzate

### 5. Analisi Visualizzazioni RANDOMNESS

Genera grafici avanzati per analizzare l'impatto del parametro RANDOMNESS:

```bash
python visualization_analysis.py
```

**Grafici generati:**

- `plots/{pipeline}/accuracy_comparison_{pipeline}.png` - Confronto performance tra reparto e sentiment
- `plots/{pipeline}/paradox_visualization_{pipeline}.png` - Visualizzazione del "Paradosso RANDOMNESS"
- `plots/{pipeline}/f1_heatmap_{pipeline}.png` - Heatmap F1-score per reparto
- `plots/{pipeline}/dashboard_complete_{pipeline}.png` - Dashboard completo con 4 subplot

### 6. Esperimenti Multi-RANDOMNESS

Esegue automaticamente esperimenti (genera Dataset e produce output) con diversi valori di RANDOMNESS (0.1 → 0.9):

```bash
python run_experiments.py
```

**Output:**

- Test automatico di N configurazioni diverse
- Risultati salvati in `experiments/randomness_experiment_YYYYMMDD_HHMMSS.json`
- Grafici generati in `plots/{pipeline}/`:\*\*
  - `accuracy_comparison_{pipeline}.png` - Confronto performance tra reparto e sentiment
  - `paradox_visualization_{pipeline}.png` - Visualizzazione del "Paradosso RANDOMNESS"
  - `f1_heatmap_{pipeline}.png` - Heatmap F1-score per reparto
  - `dashboard_complete_{pipeline}.png` - Dashboard completo con 4 subplot
- Tabella riepilogativa con performance per ogni configurazione

## Risultati Chiave e Scoperte

### Il Paradosso del RANDOMNESS

**Scoperta principale:** All'aumentare del rumore (RANDOMNESS), il **sentiment analysis migliora** mentre la **classificazione reparto peggiora**.

### Interpretazione

1. **Sentiment Analysis** si basa su **emozioni universali** → più robusto al rumore
2. **Classificazione Reparto** dipende da **domini specifici** → fragile con ambiguità
3. **RANDOMNESS = 0.2** offre il miglior **bilanciamento** per applicazioni reali

## Librerie utilizate

- **Scikit-Learn:** Framework ML principale
- **Streamlit:** Dashboard interattiva
- **Matplotlib/Seaborn:** Visualizzazioni
- **Pandas:** Manipolazione dati
- **TF-IDF:** Vettorizzazione testo
