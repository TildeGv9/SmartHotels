import pandas as pd
import numpy as np
import re
import os # Importato per gestire file e directory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load # Importato per salvare/caricare modelli

# --- Configurazione dei Percorsi ---
# Supporta variabili d'ambiente per integrazione con run_experiments.py
DATASET_PATH = os.environ.get('DATASET_PATH', 'data/sklearn/reviews/hotel_reviews_synthetic_0.9.csv')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', 'data/sklearn/predictions')
MODELS_DIR = 'models'

# Estrae il valore di RANDOMNESS dal nome del file dataset (es. hotel_reviews_synthetic_0.9.csv → 0.9)
_randomness_match = re.search(r'_(\d+\.\d+)\.csv$', DATASET_PATH)
RANDOMNESS_VALUE = _randomness_match.group(1) if _randomness_match else 'unknown'

# Crea le directory se non esistono
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Caricamento del dataset generato
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Errore: File {DATASET_PATH} non trovato. Assicurati di aver eseguito lo script di generazione del dataset.")
    exit()

# 1. Preprocessing Semplice
def preprocess(text):
    """Lowercasing, rimozione punteggiatura/numeri e spazi extra."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Rimuove la punteggiatura
    text = re.sub(r'\d+', '', text)     # Rimuove i numeri
    text = re.sub(r'\s+', ' ', text).strip() # Rimuove spazi extra
    return text

df['text'] = df['title'] + ' ' + df['body']
df['processed_text'] = df['text'].apply(preprocess)

# 2. Split Train/Test (80/20)
X = df['processed_text']
y_dept = df['department']
y_sent = df['sentiment']

X_train, X_test, y_dept_train, y_dept_test, y_sent_train, y_sent_test = train_test_split(
    X, y_dept, y_sent, test_size=0.2, random_state=42, stratify=df[['department', 'sentiment']]
)

# 3. Pipeline ML (Feature Engineering + Modello)
# Nota: la tokenizzazione del testo è gestita internamente da TfidfVectorizer,
# che suddivide il testo in token (unigrammi e bigrammi) durante il fit/transform.

# Pipeline di Classificazione Reparto (Department)
dept_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Uso anche bigrammi
    ('clf', LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000))
])

# Pipeline di Classificazione Sentiment
sent_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000))
])

# --- Addestramento ---
print("\n--- Addestramento Modello Reparto ---")
dept_pipeline.fit(X_train, y_dept_train)
y_dept_pred = dept_pipeline.predict(X_test)

print("\n--- Addestramento Modello Sentiment ---")
sent_pipeline.fit(X_train, y_sent_train)
y_sent_pred = sent_pipeline.predict(X_test)
# Confidenza: probabilità massima tra le 3 classi (pos/neg/neu)
y_sent_proba = sent_pipeline.predict_proba(X_test).max(axis=1)

# --- 4. Valutazione ---
print("\n## 📊 Valutazione Modelli\n")

# Funzione di valutazione (omessa per brevità, resta invariata)
def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    print(f"### {name} - Risultati")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1_macro:.4f}")
    print("\nReport di Classificazione:\n", report)
    
    return confusion_matrix(y_true, y_pred), acc, f1_macro

# Valutazione Reparto
cm_dept, acc_dept, f1_dept = evaluate_model(y_dept_test, y_dept_pred, "Classificazione Reparto")

# Valutazione Sentiment
cm_sent, acc_sent, f1_sent = evaluate_model(y_sent_test, y_sent_pred, "Analisi Sentiment")


# --- Matrici di Confusione e Visualizzazione (omessa per brevità, resta invariata) ---
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Vero Etichetta')
    plt.xlabel('Predetto Etichetta')
    plt.show()

plot_confusion_matrix(cm_dept, dept_pipeline.classes_, 'Matrice di Confusione - Reparto')
plot_confusion_matrix(cm_sent, sent_pipeline.classes_, 'Matrice di Confusione - Sentiment')


# --- Esempi di Errori Tipici (omessa per brevità, resta invariata) ---
print("\n## 🔎 Esempi di Errori (Reparto)\n")
error_indices = np.where(y_dept_pred != y_dept_test)[0]
if len(error_indices) > 0:
    for i in error_indices[:3]: # Mostra solo i primi 3
        idx = X_test.index[i]
        true_dept = y_dept_test.iloc[i]
        pred_dept = y_dept_pred[i]
        
        print(f"Recensione ID: {df.loc[idx, 'id']}")
        print(f"  Testo: {df.loc[idx, 'text']}")
        print(f"  Vero Reparto: {true_dept}")
        print(f"  Predetto Reparto: {pred_dept}")
        print("---")
else:
    print("Nessun errore di reparto trovato nel test set (troppo facile il dataset sintetico).")
    
# --- Output CSV con Predizioni Batch (Salvataggio in 'data/') ---
test_results = pd.DataFrame({
    'id': df.loc[X_test.index, 'id'],
    'title': df.loc[X_test.index, 'title'],
    'body': df.loc[X_test.index, 'body'],
    'true_department': y_dept_test,
    'predicted_department': y_dept_pred,
    'true_sentiment': y_sent_test,
    'predicted_sentiment': y_sent_pred,
    'sentiment_confidence': y_sent_proba
})

# Aggiornamento percorso di salvataggio (include valore RANDOMNESS e suffisso pipeline)
OUTPUT_FILENAME = f'predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}_R{RANDOMNESS_VALUE}_sklearn.csv'
OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, OUTPUT_FILENAME)

test_results.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Risultati predizioni salvati in: {OUTPUT_PATH}")


# --- Salvataggio Modelli ('models/') ---
DEPT_MODEL_PATH = os.path.join(MODELS_DIR, 'department_classifier_sklearn.joblib')
SENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_classifier_sklearn.joblib')

dump(dept_pipeline, DEPT_MODEL_PATH)
dump(sent_pipeline, SENT_MODEL_PATH)
print(f"\n✅ Modelli salvati per la dashboard in: {MODELS_DIR}/")

# --- Generazione Visualizzazioni Avanzate (opzionale) ---
def generate_performance_visualizations(acc_dept, f1_dept, acc_sent, f1_sent):
    """Genera visualizzazioni delle performance attuali."""
    try:
        from pathlib import Path
        plots_dir = Path('plots/sklearn')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Grafico performance attuale
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy
        models = ['Classificazione\nReparto', 'Analisi\nSentiment']
        accuracies = [acc_dept, acc_sent]
        colors = ['#2E86AB', '#A23B72']
        
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('Accuracy dei Modelli', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Aggiungi valori
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.02,
                     f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # F1-Score
        f1_scores = [f1_dept, f1_sent]
        bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8)
        ax2.set_title('F1-Score Macro dei Modelli', fontweight='bold', fontsize=14)
        ax2.set_ylabel('F1-Score', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., f1 + 0.02,
                     f'{f1:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.suptitle('SmartHotels: Performance Modelli ML', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'current_performance_sklearn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizzazione performance salvata in: {plots_dir / 'current_performance_sklearn.png'}")
        print(f"   (Directory: {plots_dir.absolute()})")
        
    except ImportError:
        print("⚠️  Matplotlib non disponibile per le visualizzazioni")
    except Exception as e:
        print(f"⚠️  Errore nella generazione visualizzazioni: {e}")

# Genera visualizzazioni delle performance attuali
generate_performance_visualizations(acc_dept, f1_dept, acc_sent, f1_sent)