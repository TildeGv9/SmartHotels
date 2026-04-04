#!/usr/bin/env python3
"""
Script per eseguire esperimenti automatici con diversi valori di RANDOMNESS
e generare visualizzazioni comparative per SmartHotels.
"""

import argparse
import subprocess
import sys
import os
import shutil
import time
import json
from pathlib import Path
from datetime import datetime

PIPELINE_SCRIPTS = {
    'sklearn': 'ml_pipeline_sklearn.py',
    'pytorch': 'ml_pipeline_pytorch.py',
}

def run_experiment(randomness_value, pipeline_script, pipeline_name):
    """Esegue un esperimento con un valore specifico di RANDOMNESS."""
    print(f"\n Esperimento con RANDOMNESS = {randomness_value}")

    generator_path = Path('dataset_generator.py')

    with open(generator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    try:
        # Modifica il valore di RANDOMNESS nel file del generatore
        import re
        new_content = re.sub(r'RANDOMNESS = [0-9.]+', f'RANDOMNESS = {randomness_value}', content)

        # Riscrive il file modificato
        with open(generator_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Esegui il generatore
        print("   Generando dataset...")
        result_gen = subprocess.run([sys.executable, 'dataset_generator.py'],
                                   capture_output=True, text=True)

        if result_gen.returncode != 0:
            print(f"   ❌ Errore nel generatore: {result_gen.stderr}")
            return None

        # Copia il dataset nella sottocartella della pipeline
        dataset_src = f'data/hotel_reviews_synthetic_{randomness_value}.csv'
        reviews_dir = Path(f'data/{pipeline_name}/reviews')
        predictions_dir = Path(f'data/{pipeline_name}/predictions')
        reviews_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        dataset_dst = reviews_dir / f'hotel_reviews_synthetic_{randomness_value}.csv'
        shutil.copy2(dataset_src, dataset_dst)
        os.remove(dataset_src)
        print(f"   📁 Dataset spostato in: {dataset_dst}")

        # Prepara variabili d'ambiente per la pipeline
        env = os.environ.copy()
        env['DATASET_PATH'] = str(dataset_dst)
        env['PREDICTIONS_DIR'] = str(predictions_dir)

        # Esegui la pipeline ML
        print(f"   Addestrando modelli con {pipeline_script}...")
        result_ml = subprocess.run([sys.executable, pipeline_script],
                                  capture_output=True, text=True, env=env)
        
        if result_ml.returncode != 0:
            print(f"   ❌ Errore nella pipeline ML: {result_ml.stderr}")
            return None
        
        # Estrai le metriche dall'output (parsing semplificato)
        output = result_ml.stdout
        
        # Parsing delle metriche (implementazione semplificata)
        dept_accuracy = extract_metric(output, "Classificazione Reparto", "Accuracy:")
        sent_accuracy = extract_metric(output, "Analisi Sentiment", "Accuracy:")
        dept_f1 = extract_metric(output, "Classificazione Reparto", "F1-Macro:")
        sent_f1 = extract_metric(output, "Analisi Sentiment", "F1-Macro:")
        
        result = {
            'randomness': randomness_value,
            'dept_accuracy': dept_accuracy,
            'sent_accuracy': sent_accuracy,
            'dept_f1': dept_f1,
            'sent_f1': sent_f1,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ Reparto: {dept_accuracy:.1%} | Sentiment: {sent_accuracy:.1%}")
        return result
        
    finally:
        # Ripristina il file originale
        with open(generator_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

def extract_metric(output, section, metric):
    """Estrae una metrica specifica dall'output della pipeline."""
    try:
        lines = output.split('\n')
        in_section = False
        
        for line in lines:
            if section in line:
                in_section = True
            elif in_section and metric in line:
                # Estrai il valore numerico
                import re
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    return float(match.group(1))
        return 0.0
    except:
        return 0.0

def run_full_experiment(pipeline_script, pipeline_name):
    """Esegue l'esperimento completo con diversi valori di RANDOMNESS."""
    print("SmartHotels: Esperimento RANDOMNESS Completo")
    print(f"Pipeline selezionata: {pipeline_script}")
    print("=" * 50)
    
    # Valori di RANDOMNESS da testare
    randomness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    # Crea cartella per i risultati
    results_dir = Path('experiments')
    results_dir.mkdir(exist_ok=True)
    
    # Esegui esperimenti
    for randomness in randomness_values:
        result = run_experiment(randomness, pipeline_script, pipeline_name)
        if result:
            results.append(result)
        time.sleep(1)  # Pausa tra esperimenti
    
    # Salva risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'randomness_experiment_{pipeline_name}_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Risultati salvati in: {results_file}")
    
    # Genera visualizzazioni
    try:
        from visualization_analysis import SmartHotelsVisualizer
        
        # Il visualizer carica automaticamente il JSON appena salvato
        visualizer = SmartHotelsVisualizer(pipeline_name=pipeline_name)

        print("\nGenerando visualizzazioni con dati sperimentali...")
        visualizer.generate_all_plots()
        
    except ImportError:
        print("⚠️  Modulo visualization_analysis non trovato")
    except Exception as e:
        print(f"⚠️  Errore nella generazione visualizzazioni: {e}")
    
    return results

def print_summary(results):
    """Stampa un riassunto dei risultati."""
    print("\nTABELLA RIEPILOGATIVA")
    print("=" * 50)
    print(f"{'RANDOMNESS':<12} {'Reparto':<10} {'Sentiment':<12} {'Gap':<8}")
    print("-" * 50)
    
    for result in results:
        gap = result['sent_accuracy'] - result['dept_accuracy']
        print(f"{result['randomness']:<12} {result['dept_accuracy']:<10.1%} "
              f"{result['sent_accuracy']:<12.1%} {gap:<8.1%}")
    
    # Trova il valore ottimale
    best_overall = max(results, key=lambda x: (x['dept_accuracy'] + x['sent_accuracy']) / 2)
    print(f"\n Miglior performance complessiva: RANDOMNESS = {best_overall['randomness']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Esperimenti SmartHotels con diversi valori di RANDOMNESS')
    parser.add_argument('--pipeline', choices=['sklearn', 'pytorch'], default='sklearn',
                        help='Pipeline ML da utilizzare (default: sklearn)')
    args = parser.parse_args()

    pipeline_script = PIPELINE_SCRIPTS[args.pipeline]

    try:
        results = run_full_experiment(pipeline_script, args.pipeline)
        if results:
            print_summary(results)
    except KeyboardInterrupt:
        print("\n Esperimento interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore nell'esperimento: {e}")