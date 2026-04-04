# CLAUDE.md

Questo file fornisce indicazioni a Claude Code (claude.ai/code) quando lavora con il codice di questo repository.

## Panoramica del Progetto

SmartHotels è un progetto accademico di Machine Learning che classifica recensioni alberghiere sintetiche per **reparto** (Housekeeping, Reception, F&B) e **sentiment** (pos, neg, neu). Il contributo principale della ricerca è il "Paradosso del RANDOMNESS": l'analisi del sentiment degrada gradualmente in presenza di rumore nelle etichette, mentre la classificazione per reparto collassa.

La documentazione e i commenti sono in italiano (requisito accademico).

## Comandi Principali

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Workflow standard
python dataset_generator.py          # Genera il dataset sintetico (450 recensioni)
python ml_pipeline_sklearn.py        # Addestra e valuta i modelli scikit-learn
streamlit run dashboard_app.py       # Avvia l'interfaccia utente interattiva

# Alternativa PyTorch
python ml_pipeline_pytorch.py        # Addestra i modelli PyTorch (la dashboard usa comunque joblib)

# Workflow di ricerca
python run_experiments.py                    # Sweep RANDOMNESS 0.1–0.9 (default: pytorch)
python run_experiments.py --pipeline sklearn  # Usa la pipeline scikit-learn
python run_experiments.py --pipeline pytorch  # Usa la pipeline PyTorch
python visualization_analysis.py             # Genera grafici comparativi
```

Non esiste una suite di test; la valutazione avviene tramite metriche stampate, matrici di confusione e predizioni esportate.

## Architettura

Cinque script indipendenti si scambiano dati tramite il filesystem:

```
dataset_generator.py  →  data/hotel_reviews_synthetic_{RANDOMNESS}.csv (temporaneo)
                              ↓
run_experiments.py    →  sposta in data/{sklearn|pytorch}/reviews/
                              ↓
ml_pipeline_sklearn.py →  models/*.joblib + data/sklearn/predictions/predictions_..._R{R}_sklearn.csv + plots/sklearn/
ml_pipeline_pytorch.py →  models/*.pth   + data/pytorch/predictions/predictions_..._R{R}_pytorch.csv + plots/pytorch/
                              ↓
dashboard_app.py      ←  models/*.joblib OPPURE models/*.pth (selezionabile dalla sidebar)
```

### Struttura directory `data/`
```
data/
├── sklearn/
│   ├── reviews/
│   │   └── hotel_reviews_synthetic_{R}.csv    # spostato da run_experiments.py
│   └── predictions/
│       └── predictions_YYYYMMDD_HHMMSS_R{R}_sklearn.csv
└── pytorch/
    ├── reviews/
    │   └── hotel_reviews_synthetic_{R}.csv    # spostato da run_experiments.py
    └── predictions/
        └── predictions_YYYYMMDD_HHMMSS_R{R}_pytorch.csv
```

### Struttura directory `plots/`
```
plots/
├── sklearn/
│   ├── accuracy_comparison_sklearn.png
│   ├── paradox_visualization_sklearn.png
│   ├── f1_heatmap_sklearn.png
│   ├── dashboard_complete_sklearn.png
│   ├── current_performance_sklearn.png
│   └── confusionMatrix/
│       ├── confusion_matrix_dept_R{R}_sklearn.png
│       └── confusion_matrix_sent_R{R}_sklearn.png
└── pytorch/
    ├── accuracy_comparison_pytorch.png
    ├── paradox_visualization_pytorch.png
    ├── f1_heatmap_pytorch.png
    ├── dashboard_complete_pytorch.png
    ├── current_performance_pytorch.png
    └── confusionMatrix/
        ├── confusion_matrix_dept_R{R}_pytorch.png
        └── confusion_matrix_sent_R{R}_pytorch.png
```

- **run_experiments.py** orchestra dataset_generator + ml_pipeline su più valori di RANDOMNESS, salvando i risultati JSON in `experiments/`. Genera i dataset in `data/`, poi li sposta in `data/{pipeline}/reviews/` (il file temporaneo in `data/` viene eliminato dopo la copia). Passa le variabili d'ambiente `DATASET_PATH`/`PREDICTIONS_DIR` alle pipeline. Supporta entrambe le pipeline tramite `--pipeline sklearn|pytorch`.
- **visualization_analysis.py** produce grafici comparativi (accuracy, heatmap F1, visualizzazione del paradosso) in `plots/{pipeline_name}/` quando è impostato il nome della pipeline. Carica automaticamente il JSON più recente da `experiments/` tramite `_load_experiment_data()`: se non trovato, usa dati statici di fallback. La heatmap F1 mostra il F1-Score macro per i due task (Reparto, Sentiment) su tutti i valori di RANDOMNESS testati.

## Decisioni Progettuali Chiave

- **Parametro RANDOMNESS** (0.0–1.0) in `dataset_generator.py` controlla l'iniezione di rumore nelle etichette — la variabile centrale della ricerca.
- **Dual-task, dual-framework**: Modelli separati per reparto e sentiment, implementati sia in scikit-learn che in PyTorch per confronto.
- **Pattern Pipeline**: i modelli scikit-learn incapsulano TF-IDF + LogisticRegression in un oggetto `Pipeline`, serializzato con joblib.
- **Riproducibilità**: `random_state=42` usato ovunque.
- **La dashboard supporta entrambe le pipeline** — il pulsante radio nella sidebar permette di passare tra i modelli scikit-learn (joblib) e PyTorch a runtime.
- **Preprocessing volutamente minimale** (lowercase, rimozione punteggiatura/numeri) — nessuna lemmatizzazione o stemming.
- **Override tramite variabili d'ambiente**: `ml_pipeline_sklearn.py` e `ml_pipeline_pytorch.py` leggono `DATASET_PATH` e `PREDICTIONS_DIR` dalle variabili d'ambiente se impostate (usate da `run_experiments.py`), con fallback rispettivamente su `data/{sklearn|pytorch}/reviews/` e `data/{sklearn|pytorch}/predictions/`.
- **RANDOMNESS nei nomi dei file**: I file dataset includono il valore RANDOMNESS come suffisso (es. `hotel_reviews_synthetic_0.9.csv`). Le pipeline estraggono questo valore dal nome del file per includerlo nei nomi di output delle predizioni (`_R0.9_`).

## Buone Pratiche

- **Documentare ogni modifica**: Quando si modifica il codice (rinomina file, cambio percorsi di output, aggiunta parametri, ecc.), aggiornare sempre tutta la documentazione correlata (CLAUDE.md, README.md) e i riferimenti nel codice (commenti, messaggi di errore) per riflettere le modifiche. Nessuna modifica non documentata.
- **Naming degli output specifico per pipeline**: Tutti i file di output (modelli, predizioni, grafici, JSON degli esperimenti) devono includere il suffisso `_sklearn` o `_pytorch` per distinguere quale pipeline li ha generati ed evitare sovrascritture.

## Dipendenze

Gestite tramite `requirements.txt`. Pacchetti principali: pandas, numpy, scikit-learn, torch, matplotlib, seaborn, streamlit.
