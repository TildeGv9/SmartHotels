import pandas as pd
import numpy as np
import os
import random
from sklearn.utils import shuffle

# --- Configurazione e Lessico ---
RANDOMNESS = 0.9
NUM_REVIEWS = 450  # Dataset più piccolo ma molto più challenging
DEPARTMENTS = ['Housekeeping', 'Reception', 'F&B']
SENTIMENTS = ['pos', 'neg', 'neu']

# Lessico specifico per reparto/sentiment
LEXICON = {
    'Housekeeping': {
        'pos': [
            "Camera **pulitissima** e profumata, ottimo lavoro.",
            "**Staff** molto cortese, ho apprezzato l'attenzione ai dettagli.",
            "Mi hanno aiutato con grande disponibilità, tutto risolto rapidamente.",
            "Personale gentile e competente, hanno gestito tutto perfettamente.",
            "Servizio rapido ed efficiente, sono rimasto molto soddisfatto.",
            "Grande professionalità dimostrata, esperienza davvero positiva.",
            "Mi hanno seguito con cura, risultato finale eccellente.",
            "Ottima organizzazione, tutto funziona come dovrebbe funzionare.",
        ],
        'neg': [
            "**Staff** scortese, mi hanno fatto sentire a disagio.",
            "Ho dovuto aspettare molto tempo, nessuno si è scusato.",
            "Servizio lento, non sono riusciti a risolvere il problema.",
            "Personale poco disponibile, esperienza frustrante complessivamente.",
            "Organizzazione scarsa, tutto gestito male e con superficialità.",
            "Mi hanno trattato con indifferenza, molto deluso dal risultato.",
            "Poca professionalità dimostrata, standard davvero bassi.",
            "Esperienza negativa, non tornerò più in questo posto.",
        ],
        'neu': [
            "Camera pulita, niente di particolare da segnalare.",
            "Standard di pulizia accettabile, nella media.",
            "Servizio di pulizia regolare, senza particolari note.",
            "Camera in ordine, tutto funzionale.",
            "Pulizia adeguata, niente da eccepire.",
            "Stanza pulita, servizio nella norma.",
            "Pulizia sufficiente, come ci si aspetta.",
            "Camera ordinata, standard normale.",
        ]
    },
    'Reception': {
        'pos': [
            "**Personale** molto gentile, mi hanno aiutato con grande cortesia.",
            "Servizio veloce ed efficiente, tutto risolto senza problemi.",
            "**Staff** preparato e disponibile, esperienza molto positiva.",
            "Mi hanno assistito con professionalità, sono rimasto molto contento.",
            "Organizzazione perfetta, hanno gestito tutto con grande competenza.",
            "Personale cordiale e attento, servizio davvero di qualità.",
            "Esperienza eccellente, **staff** sempre pronto ad aiutare.",
            "Mi hanno seguito passo passo, risultato finale soddisfacente.",
        ],
        'neg': [
            "**Personale** poco gentile, mi hanno fatto aspettare inutilmente.",
            "Servizio lento e disorganizzato, nessuno sapeva cosa fare.",
            "**Staff** impreparato, non sono riusciti a darmi informazioni.",
            "Mi hanno trattato male, esperienza davvero spiacevole.",
            "Organizzazione pessima, tutto gestito con superficialità evidente.",
            "Personale scortese e poco professionale, sono rimasto deluso.",
            "Esperienza negativa, **staff** indifferente ai miei problemi.",
            "Mi hanno ignorato completamente, servizio davvero scadente.",
        ],
        'neu': [
            "Check-in regolare, senza particolari problemi.",
            "Personale corretto, servizio nella norma.",
            "Reception funzionale, tempi di attesa accettabili.",
            "Servizio standard, tutto ok.",
            "Personale disponibile quando necessario, nella media.",
            "Check-out veloce, nulla da segnalare.",
            "Accoglienza normale, come da aspettative.",
            "Servizio adeguato, senza particolari note.",
        ]
    },
    'F&B': {
        'pos': [
            "**Personale** molto cortese, mi hanno servito con grande attenzione.",
            "Servizio veloce e **staff** sempre disponibile, esperienza piacevole.",
            "Mi hanno consigliato bene, qualità ottima e prezzi giusti.",
            "**Staff** preparato e gentile, tutto servito con grande cura.",
            "Organizzazione perfetta, hanno gestito tutto con professionalità.",
            "Personale attento ai dettagli, servizio davvero di livello.",
            "Esperienza molto positiva, **staff** competente e disponibile.",
            "Mi hanno seguito con attenzione, risultato finale eccellente.",
            "Tutto ok, esperienza nella media senza particolari problemi.",
            "Va bene così, niente di eccezionale ma nemmeno male.",
            "Esperienza standard, quello che ci si aspetta normalmente.",
        ],
        'neg': [
            "**Personale** poco attento, mi hanno servito con superficialità.",
            "Servizio lento e **staff** disorganizzato, esperienza deludente.",
            "Mi hanno fatto aspettare troppo, qualità sotto le aspettative.",
            "**Staff** impreparato, non sapevano consigliarmi adeguatamente.",
            "Organizzazione scarsa, tutto gestito male e con poca cura.",
            "Personale poco disponibile, servizio davvero scadente complessivamente.",
            "Esperienza negativa, **staff** indifferente ai clienti.",
            "Mi hanno trattato con superficialità, sono rimasto molto deluso.",
        ],
        'neu': [
            "Colazione standard, varietà sufficiente.",
            "Cibo nella media, prezzi normali per la categoria.",
            "Servizio regolare, niente di particolare.",
            "Qualità accettabile, scelta adeguata.",
            "Ristorante nella norma, prezzi in linea.",
            "Colazione decente, tutto sommato ok.",
            "Cibo discreto, servizio standard.",
            "Pasti nella media, nulla di eccezionale o negativo.",
        ]
    }
}

# Modelli per i titoli
TITLES = [
    "Servizio eccellente!", "Problemi al check-in", "Colazione da dimenticare",
    "Camera impeccabile", "Staff professionale", "Un disastro",
    "Torneremo sicuramente", "Delusione totale", "Prezzo e qualità non si incontrano"
]

def generate_review(department, sentiment):
    """Genera una recensione per reparto e sentiment specifici."""
    body_templates = LEXICON[department][sentiment]
    body = random.choice(body_templates)
    # Aggiunge un po' di rumore/ambiguità:
    noise = random.choice([
        " La posizione è centrale.",
        " Il Wi-Fi era lento.",
        " L'arredamento è datato.",
        " Buona la piscina.",
        " Lo staff era giovane e inesperto.",
        " I prezzi sono nella media.",
        " Parcheggio difficile da trovare.",
        " Vista mare molto bella.",
        " Struttura un po' rumorosa.",
        ""  # Senza rumore
    ])
    # Togli il grassetto se presente
    body = body.replace('**', '')
    
    # Aggiunge il rumore più frequentemente
    if len(body) > 30:  # Soglia più bassa
        if random.random() < RANDOMNESS:
            body += noise
        
    # Sceglie un titolo, magari relazionato al sentiment
    if sentiment == 'pos':
        title = random.choice([t for t in TITLES if 'eccellente' in t or 'impeccabile' in t or 'professionale' in t or 'sicuramente' in t])
    else:
        title = random.choice([t for t in TITLES if 'Problemi' in t or 'dimenticare' in t or 'disastro' in t or 'Delusione' in t])
        
    return title, body.strip()

def create_mixed_review(target_dept, target_sent):
    """Crea una recensione che parla genuinamente di più reparti."""
    # Template che mischiano veramente i reparti
    mixed_templates = {
        'pos': [
            "Esperienza complessivamente positiva. Il personale è stato gentile e disponibile in ogni situazione.",
            "Molto soddisfatto del soggiorno. Staff competente, tutto organizzato bene e servizio di qualità.",
            "Bella esperienza, personale cordiale e attento. Tutto gestito con professionalità.",
            "Consiglio questo posto. Servizio efficiente, staff preparato e prezzi onesti.",
            "Esperienza piacevole, tutto funziona bene e il personale è sempre disponibile.",
        ],
        'neg': [
            "Esperienza deludente nel complesso. Il personale sembra poco motivato e il servizio è carente.",
            "Non sono rimasto soddisfatto. Staff disorganizzato, servizio lento e poca attenzione ai dettagli.",
            "Esperienza al di sotto delle aspettative. Personale poco disponibile e organizzazione scarsa.",
            "Non lo consiglio. Servizio scadente, staff poco preparato e prezzi troppo alti.",
            "Esperienza negativa, tutto gestito male e il personale sembra indifferente.",
        ],
        'neu': [
            "Soggiorno nella norma. Tutto funziona come dovrebbe, senza particolari sorprese.",
            "Esperienza standard. Servizio regolare e prezzi in linea con la categoria.",
            "Tutto ok nel complesso. Niente di eccezionale ma nemmeno problemi particolari.",
            "Soggiorno normale. Struttura adeguata, servizi nella media.",
            "Esperienza accettabile. Qualche aspetto positivo, qualche aspetto migliorabile.",
        ]
    }
    
    title = random.choice(TITLES)
    body = random.choice(mixed_templates[target_sent])
    
    return title, body

def add_realistic_errors(df, error_rate=0.15):
    """Aggiunge errori realistici al dataset per simulare classificazioni difficili."""
    df_copy = df.copy()
    n_errors = int(len(df) * error_rate)
    
    # Seleziona casualmente le righe da "corrompere"
    error_indices = np.random.choice(df.index, n_errors, replace=False)
    
    for idx in error_indices:
        current_dept = df_copy.loc[idx, 'department']
        current_sent = df_copy.loc[idx, 'sentiment']
        
        if random.random() < RANDOMNESS:
            # Cambia reparto mantenendo sentiment
            available_depts = [d for d in DEPARTMENTS if d != current_dept]
            df_copy.loc[idx, 'department'] = random.choice(available_depts)
        else:
            # Cambia sentiment mantenendo reparto  
            df_copy.loc[idx, 'sentiment'] = 'pos' if current_sent == 'neg' else 'neg'
    
    return df_copy

# --- Generazione del Dataset ---
data = []
# Calcola quante recensioni per combinazione (es. HK/pos)
reviews_per_combination = NUM_REVIEWS // (len(DEPARTMENTS) * len(SENTIMENTS))

review_id = 1
for dept in DEPARTMENTS:
    for sent in SENTIMENTS:
        for i in range(reviews_per_combination):
            if random.random() < RANDOMNESS:
                title, body = create_mixed_review(dept, sent)
            else:
                title, body = generate_review(dept, sent)
                
            data.append({
                'id': review_id,
                'title': title,
                'body': body,
                'department': dept,
                'sentiment': sent
            })
            review_id += 1

# Crea DataFrame, lo mescola e aggiunge errori realistici
df = pd.DataFrame(data)
df = shuffle(df, random_state=42).reset_index(drop=True)

# ⚠️ Aggiunge errori realistici al dataset
print("🔄 Aggiunta di errori realistici al dataset...")
df = add_realistic_errors(df, error_rate=RANDOMNESS) 

df['id'] = df.index + 1 # Ri-assegna ID sequenziale dopo lo shuffle

# Esporta il dataset (il nome include il valore di RANDOMNESS come suffisso)
DATASET_PATH = f'data/hotel_reviews_synthetic_{RANDOMNESS}.csv'
os.makedirs('data', exist_ok=True)
df.to_csv(DATASET_PATH, index=False)
print(f"✅ Dataset generato e salvato in: {DATASET_PATH}")
print(f"Totale recensioni: {len(df)}")
print("\nDistribuzione Reparto:")
print(df['department'].value_counts())
print("\nDistribuzione Sentiment:")
print(df['sentiment'].value_counts())