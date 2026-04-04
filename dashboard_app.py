import streamlit as st
import pandas as pd
from joblib import load
import time # Per l'export con timestamp
import os # Necessario per path
import re

# Definisci le directory all'inizio della dashboard
MODELS_DIR = 'models'

# --- Preprocessing (comune a entrambe le pipeline) ---
def preprocess(text):
    """Lowercasing, rimozione punteggiatura/numeri e spazi extra."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Caricamento modelli sklearn ---
@st.cache_resource
def load_sklearn_models():
    try:
        dept_path = os.path.join(MODELS_DIR, 'department_classifier_sklearn.joblib')
        sent_path = os.path.join(MODELS_DIR, 'sentiment_classifier_sklearn.joblib')
        dept_clf = load(dept_path)
        sent_clf = load(sent_path)
        return dept_clf, sent_clf
    except FileNotFoundError:
        return None, None

# --- Caricamento modelli PyTorch ---
@st.cache_resource
def load_pytorch_models():
    try:
        import torch
        import torch.nn as nn

        # Definizione architettura (deve corrispondere a ml_pipeline_pytorch.py)
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

        dept_ckpt = torch.load(os.path.join(MODELS_DIR, 'department_pytorch.pth'),
                               map_location='cpu', weights_only=False)
        sent_ckpt = torch.load(os.path.join(MODELS_DIR, 'sentiment_pytorch.pth'),
                               map_location='cpu', weights_only=False)

        # Ricostruisci i modelli dall'architettura e pesi salvati
        dept_vectorizer = dept_ckpt['vectorizer']
        dept_encoder = dept_ckpt['encoder']
        input_size_dept = len(dept_vectorizer.vocabulary_)
        dept_model = TextClassifier(input_size_dept, len(dept_encoder.classes_))
        dept_model.load_state_dict(dept_ckpt['model_state_dict'])
        dept_model.eval()

        sent_vectorizer = sent_ckpt['vectorizer']
        sent_encoder = sent_ckpt['encoder']
        input_size_sent = len(sent_vectorizer.vocabulary_)
        sent_model = TextClassifier(input_size_sent, len(sent_encoder.classes_))
        sent_model.load_state_dict(sent_ckpt['model_state_dict'])
        sent_model.eval()

        return dept_model, sent_model, dept_vectorizer, sent_vectorizer, dept_encoder, sent_encoder
    except (FileNotFoundError, ImportError):
        return None, None, None, None, None, None

# --- Funzioni di predizione ---
def predict_sklearn(texts, dept_clf, sent_clf):
    """Predizione con modelli sklearn. Accetta lista di testi preprocessati."""
    dept_preds = dept_clf.predict(texts)
    sent_preds = sent_clf.predict(texts)
    sent_proba = sent_clf.predict_proba(texts)
    confidences = sent_proba.max(axis=1)
    proba_dicts = [dict(zip(sent_clf.classes_, row)) for row in sent_proba]
    return dept_preds, sent_preds, confidences, proba_dicts

def predict_pytorch(texts, dept_model, sent_model, dept_vec, sent_vec, dept_enc, sent_enc):
    """Predizione con modelli PyTorch. Accetta lista di testi preprocessati."""
    import torch

    # TF-IDF + forward pass per reparto
    X_dept = torch.FloatTensor(dept_vec.transform(texts).toarray())
    with torch.no_grad():
        dept_out = dept_model(X_dept)
        dept_preds = dept_enc.inverse_transform(torch.argmax(dept_out, dim=1).numpy())

    # TF-IDF + forward pass per sentiment
    X_sent = torch.FloatTensor(sent_vec.transform(texts).toarray())
    with torch.no_grad():
        sent_out = sent_model(X_sent)
        sent_proba = torch.softmax(sent_out, dim=1).numpy()
        sent_preds = sent_enc.inverse_transform(torch.argmax(sent_out, dim=1).numpy())

    confidences = sent_proba.max(axis=1)
    proba_dicts = [dict(zip(sent_enc.classes_, row)) for row in sent_proba]
    return dept_preds, sent_preds, confidences, proba_dicts

# --- Interfaccia Streamlit ---
st.set_page_config(layout="wide")

# CSS personalizzato
st.markdown("""
<style>
    /* Bordi e accenti azzurri su card/form */
    div[data-testid="stForm"] {
        border: 1px solid #00AEEF33;
        border-radius: 8px;
        padding: 16px;
    }
    /* Separatore orizzontale */
    hr {
        border-color: #00AEEF44;
    }
    /* Intestazioni sezione */
    h2, h3 {
        color: #00AEEF !important;
    }
    /* Badge sidebar (success/error) */
    div[data-testid="stSidebar"] .stAlert {
        border-radius: 6px;
    }
    /* Pulsanti: bordo azzurro al hover */
    div.stButton > button:hover {
        border-color: #00AEEF;
        color: #00AEEF;
    }
    /* Dataframe header */
    thead tr th {
        background-color: #0D1B2A !important;
        color: #00AEEF !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏨 Smistamento Recensioni e Analisi Sentiment con ML")

# Sidebar: selezione pipeline
st.sidebar.header("⚙️ Configurazione")
pipeline_choice = st.sidebar.radio(
    "Pipeline ML da utilizzare:",
    ["scikit-learn", "PyTorch"],
    index=0,
    help="Seleziona il framework dei modelli da caricare per le predizioni."
)

# Caricamento modelli in base alla scelta
if pipeline_choice == "scikit-learn":
    dept_clf, sent_clf = load_sklearn_models()
    models_loaded = dept_clf is not None
    if not models_loaded:
        st.sidebar.error("Modelli sklearn non trovati. Esegui prima `ml_pipeline_sklearn.py`.")
    else:
        st.sidebar.success("Modelli scikit-learn caricati.")
else:
    pytorch_result = load_pytorch_models()
    dept_model, sent_model, dept_vec, sent_vec, dept_enc, sent_enc = pytorch_result
    models_loaded = dept_model is not None
    if not models_loaded:
        st.sidebar.error("Modelli PyTorch non trovati. Esegui prima `ml_pipeline_pytorch.py`.")
    else:
        st.sidebar.success("Modelli PyTorch caricati.")

st.markdown("---")

# Sezione 1: Predizione Singola
st.header("1. Analizza una Recensione Singola 💬")
with st.form("single_review_form"):
    title = st.text_input("Titolo della Recensione (es. 'Servizio veloce')", max_chars=50)
    body = st.text_area("Testo della Recensione (es. 'Camera pulita ma check-out lento')", height=150)
    submitted = st.form_submit_button("Analizza")

    if submitted:
        if not title and not body:
            st.error("Titolo e testo della recensione sono obbligatori.")
        elif not title:
            st.error("Il titolo della recensione è obbligatorio.")
        elif not body:
            st.error("Il testo della recensione è obbligatorio.")
        else:
            if not models_loaded:
                st.error("Nessun modello disponibile. Controlla la configurazione nella sidebar.")
            else:
                full_text = preprocess(title + ' ' + body)

                if pipeline_choice == "scikit-learn":
                    dept_preds, sent_preds, confidences, proba_dicts = predict_sklearn(
                        [full_text], dept_clf, sent_clf)
                else:
                    dept_preds, sent_preds, confidences, proba_dicts = predict_pytorch(
                        [full_text], dept_model, sent_model, dept_vec, sent_vec, dept_enc, sent_enc)

                dept = dept_preds[0]
                sent = sent_preds[0]
                confidence = confidences[0]
                proba_dict = proba_dicts[0]

                # Soglia minima di confidenza: con 3 classi il caso random è ~0.33
                CONFIDENCE_THRESHOLD = 0.45

                # Icone per il sentiment
                sent_map = {'pos': ('🟢 Positivo', 'green'), 'neg': ('🔴 Negativo', 'red'), 'neu': ('🟡 Neutro', 'orange')}
                sent_emoji, color = sent_map.get(sent, ('⚪ Sconosciuto', 'gray'))

                # Visualizzazione dei risultati
                st.subheader("Risultato dell'Analisi")

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"⚠️ Testo non riconosciuto — confidenza troppo bassa ({confidence:.2f}). "
                        "Inserisci una recensione più descrittiva."
                    )
                else:
                    st.markdown(f"**Reparto Consigliato:** **<span style='font-size: 24px;'>{dept}</span>**", unsafe_allow_html=True)
                    st.markdown(f"**Sentiment Stimato:** **<span style='color:{color}; font-size: 24px;'>{sent_emoji}</span>**", unsafe_allow_html=True)
                    st.info(f"Confidenza predizione: **{confidence:.2f}**")

                st.caption(f"Pipeline: **{pipeline_choice}**")
                # Dettaglio probabilità per classe
                proba_text = " | ".join([f"P({cls}): {p:.2f}" for cls, p in sorted(proba_dict.items())])
                st.caption(proba_text)

st.markdown("---")

# Sezione 2: Predizione Batch (Upload CSV)
st.header("2. Predizione Batch (Carica CSV) 🔗")
uploaded_file = st.file_uploader("Carica un file CSV (deve contenere colonne 'title' e 'body')", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Verifica colonne
        if 'title' not in batch_df.columns or 'body' not in batch_df.columns:
            st.error("Il file CSV deve contenere le colonne 'title' e 'body'.")
        elif not models_loaded:
            st.error("Nessun modello disponibile. Controlla la configurazione nella sidebar.")
        else:
            st.success(f"File caricato correttamente. Trovate {len(batch_df)} recensioni.")

            if st.button("Esegui Predizione Batch"):
                with st.spinner('Analisi in corso...'):
                    # Preprocessing batch
                    batch_df['processed_text'] = (batch_df['title'] + ' ' + batch_df['body']).apply(preprocess)
                    texts = batch_df['processed_text'].tolist()

                    if pipeline_choice == "scikit-learn":
                        dept_preds, sent_preds, confidences, _ = predict_sklearn(
                            texts, dept_clf, sent_clf)
                    else:
                        dept_preds, sent_preds, confidences, _ = predict_pytorch(
                            texts, dept_model, sent_model, dept_vec, sent_vec, dept_enc, sent_enc)

                    batch_df['predicted_department'] = dept_preds
                    batch_df['predicted_sentiment'] = sent_preds
                    batch_df['sentiment_confidence'] = confidences

                    st.subheader("Risultati del Batch")
                    st.dataframe(batch_df[['title', 'body', 'predicted_department', 'predicted_sentiment', 'sentiment_confidence']].head())

                    # Esporta risultati con timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    csv_output = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Scarica Risultati Predizione CSV",
                        data=csv_output,
                        file_name=f'batch_predictions_{timestamp}.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"Si è verificato un errore durante la lettura del file: {e}")
