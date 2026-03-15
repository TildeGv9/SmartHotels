import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurazione dei Percorsi ---
# Supporta variabili d'ambiente per integrazione con run_experiments.py
DATASET_PATH = os.environ.get('DATASET_PATH', 'data/pytorch/reviews/hotel_reviews_synthetic_0.9.csv')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', 'data/pytorch/predictions')
MODELS_DIR = 'models'

# Estrae il valore di RANDOMNESS dal nome del file dataset (es. hotel_reviews_synthetic_0.9.csv → 0.9)
_randomness_match = re.search(r'_(\d+\.\d+)\.csv$', DATASET_PATH)
RANDOMNESS_VALUE = _randomness_match.group(1) if _randomness_match else 'unknown'

# Crea le directory se non esistono
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Riproducibilità
torch.manual_seed(42)
np.random.seed(42)

# Controlla se GPU è disponibile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Usando dispositivo: {device}")

# Caricamento del dataset
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Errore: File {DATASET_PATH} non trovato.")
    exit()

# 1. Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['title'] + ' ' + df['body']
df['processed_text'] = df['text'].apply(preprocess)

# 2. Preparazione dati
X = df['processed_text']
y_dept = df['department']
y_sent = df['sentiment']

# Encoding delle label
from sklearn.preprocessing import LabelEncoder
dept_encoder = LabelEncoder()
sent_encoder = LabelEncoder()

y_dept_encoded = dept_encoder.fit_transform(y_dept)
y_sent_encoded = sent_encoder.fit_transform(y_sent)

# Split
X_train, X_test, y_dept_train, y_dept_test, y_sent_train, y_sent_test = train_test_split(
    X, y_dept_encoded, y_sent_encoded, test_size=0.2, random_state=42,
    stratify=np.column_stack([y_dept_encoded, y_sent_encoded])
)

# 3. Feature Extraction (TF-IDF)
dept_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
sent_vectorizer = TfidfVectorizer(max_features=5000)

X_train_dept_tfidf = dept_vectorizer.fit_transform(X_train).toarray()
X_test_dept_tfidf = dept_vectorizer.transform(X_test).toarray()

X_train_sent_tfidf = sent_vectorizer.fit_transform(X_train).toarray()
X_test_sent_tfidf = sent_vectorizer.transform(X_test).toarray()

# 4. Dataset PyTorch
class ReviewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 5. Modello Neural Network
class TextClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# 6. Funzione di Training
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoca [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 7. Funzione di Valutazione
def evaluate_model(model, test_loader, y_true, name):
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(y_true, all_preds)
    f1_macro = f1_score(y_true, all_preds, average='macro', zero_division=0)
    report = classification_report(y_true, all_preds, zero_division=0)
    
    print(f"\n### {name} - Risultati")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1_macro:.4f}")
    print("\nReport di Classificazione:\n", report)
    
    return confusion_matrix(y_true, all_preds), acc, f1_macro, all_preds, np.array(all_probs)

# --- Training Modello Reparto ---
print("\n--- Addestramento Modello Reparto (PyTorch) ---")
dept_train_dataset = ReviewDataset(X_train_dept_tfidf, y_dept_train)
dept_test_dataset = ReviewDataset(X_test_dept_tfidf, y_dept_test)

dept_train_loader = DataLoader(dept_train_dataset, batch_size=32, shuffle=True)
dept_test_loader = DataLoader(dept_test_dataset, batch_size=32, shuffle=False)

num_dept_classes = len(dept_encoder.classes_)
dept_model = TextClassifier(X_train_dept_tfidf.shape[1], num_dept_classes).to(device)
dept_criterion = nn.CrossEntropyLoss()
dept_optimizer = optim.Adam(dept_model.parameters(), lr=0.001)

train_model(dept_model, dept_train_loader, dept_criterion, dept_optimizer, epochs=50)

# --- Training Modello Sentiment ---
print("\n--- Addestramento Modello Sentiment (PyTorch) ---")
sent_train_dataset = ReviewDataset(X_train_sent_tfidf, y_sent_train)
sent_test_dataset = ReviewDataset(X_test_sent_tfidf, y_sent_test)

sent_train_loader = DataLoader(sent_train_dataset, batch_size=32, shuffle=True)
sent_test_loader = DataLoader(sent_test_dataset, batch_size=32, shuffle=False)

num_sent_classes = len(sent_encoder.classes_)
sent_model = TextClassifier(X_train_sent_tfidf.shape[1], num_sent_classes).to(device)
sent_criterion = nn.CrossEntropyLoss()
sent_optimizer = optim.Adam(sent_model.parameters(), lr=0.001)

train_model(sent_model, sent_train_loader, sent_criterion, sent_optimizer, epochs=50)

# --- Valutazione ---
print("\n## 📊 Valutazione Modelli (PyTorch)\n")

cm_dept, acc_dept, f1_dept, y_dept_pred, _ = evaluate_model(
    dept_model, dept_test_loader, y_dept_test, "Classificazione Reparto"
)

cm_sent, acc_sent, f1_sent, y_sent_pred, y_sent_proba = evaluate_model(
    sent_model, sent_test_loader, y_sent_test, "Analisi Sentiment"
)

# --- Salvataggio Modelli ---
torch.save({
    'model_state_dict': dept_model.state_dict(),
    'vectorizer': dept_vectorizer,
    'encoder': dept_encoder
}, os.path.join(MODELS_DIR, 'department_pytorch.pth'))

torch.save({
    'model_state_dict': sent_model.state_dict(),
    'vectorizer': sent_vectorizer,
    'encoder': sent_encoder
}, os.path.join(MODELS_DIR, 'sentiment_pytorch.pth'))

print(f"\n✅ Modelli PyTorch salvati in: {MODELS_DIR}/")

# --- Generazione Visualizzazioni Avanzate (opzionale) ---
def generate_performance_visualizations(acc_dept, f1_dept, acc_sent, f1_sent):
    """Genera visualizzazioni delle performance attuali."""
    try:
        from pathlib import Path
        plots_dir = Path('plots/pytorch')
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

        plt.suptitle('SmartHotels: Performance Modelli PyTorch', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'current_performance_pytorch.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Visualizzazione performance salvata in: {plots_dir / 'current_performance_pytorch.png'}")

    except ImportError:
        print("⚠️  Matplotlib non disponibile per le visualizzazioni")
    except Exception as e:
        print(f"⚠️  Errore nella generazione visualizzazioni: {e}")

# Genera visualizzazioni delle performance attuali
generate_performance_visualizations(acc_dept, f1_dept, acc_sent, f1_sent)

# --- Visualizzazioni ---
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Vero Etichetta')
    plt.xlabel('Predetto Etichetta')
    plt.show()

plot_confusion_matrix(cm_dept, dept_encoder.classes_, 'Matrice di Confusione - Reparto (PyTorch)')
plot_confusion_matrix(cm_sent, sent_encoder.classes_, 'Matrice di Confusione - Sentiment (PyTorch)')

# --- Salvataggio Predizioni ---
y_dept_pred_decoded = dept_encoder.inverse_transform(y_dept_pred)
y_sent_pred_decoded = sent_encoder.inverse_transform(y_sent_pred)

test_results = pd.DataFrame({
    'id': df.loc[X_test.index, 'id'],
    'title': df.loc[X_test.index, 'title'],
    'body': df.loc[X_test.index, 'body'],
    'true_department': dept_encoder.inverse_transform(y_dept_test),
    'predicted_department': y_dept_pred_decoded,
    'true_sentiment': sent_encoder.inverse_transform(y_sent_test),
    'predicted_sentiment': y_sent_pred_decoded,
    'sentiment_confidence': y_sent_proba.max(axis=1)
})

OUTPUT_FILENAME = f'predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}_R{RANDOMNESS_VALUE}_pytorch.csv'
OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, OUTPUT_FILENAME)
test_results.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Risultati predizioni salvati in: {OUTPUT_PATH}")