import streamlit as st
import pandas as pd
from joblib import load
import time # Per l'export con timestamp
import os # Necessario per path

# Definisci le directory all'inizio della dashboard
MODELS_DIR = 'models'

# Carica i modelli e la funzione di preprocessing
@st.cache_resource
def load_models():
    # Deve corrispondere alla funzione di preprocessing in ml_pipeline.py
    def preprocess(text):
        # Implementa qui la stessa logica di preprocess del modello
        import re
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    try:
        # Percorsi aggiornati
        dept_path = os.path.join(MODELS_DIR, 'department_classifier.joblib')
        sent_path = os.path.join(MODELS_DIR, 'sentiment_classifier.joblib')
        
        dept_clf = load(dept_path)
        sent_clf = load(sent_path)
        return dept_clf, sent_clf, preprocess
    except FileNotFoundError:
        st.error(f"Errore: I file dei modelli non sono stati trovati nella cartella '{MODELS_DIR}'. Esegui prima 'ml_pipeline.py'.")
        return None, None, None

dept_clf, sent_clf, preprocess = load_models()

# --- Funzione di Predizione Singola ---
def predict_review(title, body):
    if dept_clf is None or sent_clf is None:
        return None, None, None, None
    
    full_text = preprocess(title + ' ' + body)
    
    # Previsione Reparto
    predicted_dept = dept_clf.predict([full_text])[0]
    
    # Previsione Sentiment
    predicted_sent = sent_clf.predict([full_text])[0]
    
    # Probabilità Sentiment (probabilità di 'pos' - è il secondo elemento dell'array se le classi sono ['neg', 'pos'])
    proba_array = sent_clf.predict_proba([full_text])[0]
    
    # Assumendo che le classi di sentiment siano ['neg', 'pos']
    if sent_clf.classes_[1] == 'pos':
        pos_proba = proba_array[1]
    else: # Gestione se l'ordine fosse diverso
        pos_proba = proba_array[0] 
        
    return predicted_dept, predicted_sent, pos_proba, full_text

# --- Interfaccia Streamlit ---
st.set_page_config(layout="wide")
st.title("🏨 Smistamento Recensioni e Analisi Sentiment con ML")
st.markdown("---")

# Sezione 1: Predizione Singola
st.header("1. Analizza una Recensione Singola")
with st.form("single_review_form"):
    title = st.text_input("Titolo della Recensione (es. 'Servizio veloce')", max_chars=50)
    body = st.text_area("Testo della Recensione (es. 'Camera pulita ma check-out lento')", height=150)
    submitted = st.form_submit_button("Analizza")

    if submitted:
        if body:
            dept, sent, proba, processed = predict_review(title, body)
            
            if dept:
                # Icone per il sentiment
                sent_emoji = "🟢 Positivo" if sent == 'pos' else "🔴 Negativo"
                color = "green" if sent == 'pos' else "red"
                
                # Visualizzazione dei risultati
                st.subheader("Risultato dell'Analisi")
                st.markdown(f"**Reparto Consigliato:** **<span style='font-size: 24px;'>{dept}</span>**", unsafe_allow_html=True)
                st.markdown(f"**Sentiment Stimato:** **<span style='color:{color}; font-size: 24px;'>{sent_emoji}</span>**", unsafe_allow_html=True)
                st.info(f"Probabilità che sia Positivo (P(pos)): **{proba:.2f}**")
                # st.caption(f"Testo preprocessato: {processed}")

        else:
            st.warning("Per favore, inserisci il testo della recensione.")

st.markdown("---")

# Sezione 2: Predizione Batch (Upload CSV)
st.header("2. Predizione Batch (Carica CSV)")
uploaded_file = st.file_uploader("Carica un file CSV (deve contenere colonne 'title' e 'body')", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        
        # Verifica colonne
        if 'title' not in batch_df.columns or 'body' not in batch_df.columns:
            st.error("Il file CSV deve contenere le colonne 'title' e 'body'.")
        else:
            st.success(f"File caricato correttamente. Trovate {len(batch_df)} recensioni.")
            
            if st.button("Esegui Predizione Batch"):
                with st.spinner('Analisi in corso...'):
                    # Preprocessing batch
                    batch_df['processed_text'] = (batch_df['title'] + ' ' + batch_df['body']).apply(preprocess)
                    
                    # Predizioni
                    batch_df['predicted_department'] = dept_clf.predict(batch_df['processed_text'])
                    batch_df['predicted_sentiment'] = sent_clf.predict(batch_df['processed_text'])
                    
                    # Probabilità Sentiment
                    proba_array = sent_clf.predict_proba(batch_df['processed_text'])
                    # Assumendo che le classi di sentiment siano ['neg', 'pos']
                    pos_proba_index = 1 if sent_clf.classes_[1] == 'pos' else 0
                    batch_df['positive_proba'] = proba_array[:, pos_proba_index]
                    
                    st.subheader("Risultati del Batch")
                    st.dataframe(batch_df[['title', 'body', 'predicted_department', 'predicted_sentiment', 'positive_proba']].head())
                    
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
