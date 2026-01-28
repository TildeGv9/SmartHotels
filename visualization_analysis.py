import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Configurazione stile
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SmartHotelsVisualizer:
    """Classe per generare visualizzazioni dell'analisi RANDOMNESS per SmartHotels."""
    
    def __init__(self):
        # Dati sperimentali raccolti
        # Valori statici per l'esempio, verrano sostituiti con dati provenienti dagli esperimenti desiderati (run_experiments.py)
        self.randomness_values = [0.2, 0.4, 0.9]
        self.reparto_accuracy = [0.76, 0.57, 0.31]
        self.sentiment_accuracy = [0.81, 0.70, 0.93]
        
        # F1-scores dettagliati per reparto
        self.f1_data = {
            0.2: {'F&B': 0.81, 'Housekeeping': 0.75, 'Reception': 0.71},
            0.4: {'F&B': 0.62, 'Housekeeping': 0.52, 'Reception': 0.55},
            0.9: {'F&B': 0.35, 'Housekeeping': 0.30, 'Reception': 0.28}
        }
        
        # Crea cartella per i grafici
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
    
    def plot_accuracy_comparison(self, save=True):
        """Grafico 1: Confronto Accuracy tra Reparto e Sentiment."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.randomness_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, self.reparto_accuracy, width, 
                      label='Classificazione Reparto', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, self.sentiment_accuracy, width, 
                      label='Analisi Sentiment', color='#A23B72', alpha=0.8)
        
        # Aggiungi valori sulle barre
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.0%}', ha='center', va='bottom', fontweight='bold')
                    
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.0%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Parametro RANDOMNESS', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('SmartHotels: Impatto del RANDOMNESS sulle Performance', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.randomness_values)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_paradox_visualization(self, save=True):
        """Grafico 2: Visualizzazione del Paradosso RANDOMNESS."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Linee principali
        ax.plot(self.randomness_values, self.reparto_accuracy, 'o-', 
                linewidth=4, markersize=12, label='Classificazione Reparto', 
                color='#2E86AB', alpha=0.9)
        ax.plot(self.randomness_values, self.sentiment_accuracy, '^-', 
                linewidth=4, markersize=12, label='Analisi Sentiment', 
                color='#A23B72', alpha=0.9)
        
        # Zona del paradosso
        ax.axvspan(0.8, 1.0, alpha=0.2, color='gold', label='Zona Paradosso')
        
        # Annotazioni del paradosso
        ax.annotate('PARADOSSO!\nSentiment migliora\ncon più rumore', 
                    xy=(0.9, 0.93), xytext=(0.65, 0.85),
                    arrowprops=dict(arrowstyle='->', color='#A23B72', lw=3),
                    fontsize=12, ha='center', color='#A23B72', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        ax.annotate('Reparto collassa\ncon troppo rumore', 
                    xy=(0.9, 0.31), xytext=(0.65, 0.45),
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=3),
                    fontsize=12, ha='center', color='#2E86AB', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Parametro RANDOMNESS', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Il Paradosso del RANDOMNESS\n"Più Rumore = Sentiment Migliore"', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        ax.set_ylim(0.2, 1.0)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.plots_dir / 'paradox_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_f1_heatmap(self, save=True):
        """Grafico 3: Heatmap F1-Score per Reparto."""
        # Prepara dati per heatmap
        pivot_data = np.array([[self.f1_data[0.2]['F&B'], self.f1_data[0.2]['Housekeeping'], self.f1_data[0.2]['Reception']],
                               [self.f1_data[0.4]['F&B'], self.f1_data[0.4]['Housekeeping'], self.f1_data[0.4]['Reception']],
                               [self.f1_data[0.9]['F&B'], self.f1_data[0.9]['Housekeeping'], self.f1_data[0.9]['Reception']]])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_data, 
                    xticklabels=['F&B', 'Housekeeping', 'Receptionh'],
                    yticklabels=['RANDOMNESS = 0.2', 'RANDOMNESS = 0.4', 'RANDOMNESS = 0.9'],
                    annot=True, cmap='RdYlBu_r', center=0.5,
                    fmt='.2f', cbar_kws={'label': 'F1-Score'},
                    linewidths=0.5, ax=ax)
        
        ax.set_title('F1-Score per Reparto al Variare del RANDOMNESS', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Reparto', fontsize=14, fontweight='bold')
        ax.set_ylabel('Configurazione RANDOMNESS', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.plots_dir / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dashboard(self, save=True):
        """Grafico 4: Dashboard completo con 4 subplots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy Comparison
        x = np.arange(len(self.randomness_values))
        width = 0.35
        ax1.bar(x - width/2, self.reparto_accuracy, width, label='Reparto', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, self.sentiment_accuracy, width, label='Sentiment', color='#A23B72', alpha=0.8)
        ax1.set_title('Accuracy per RANDOMNESS', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.randomness_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Gap Sentiment-Reparto
        diff = np.array(self.sentiment_accuracy) - np.array(self.reparto_accuracy)
        colors = ['green' if d > 0 else 'red' for d in diff]
        bars = ax2.bar(self.randomness_values, diff, color=colors, alpha=0.7)
        ax2.set_title('Gap Sentiment - Reparto', fontweight='bold')
        ax2.set_xlabel('RANDOMNESS')
        ax2.set_ylabel('Differenza Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar, val in zip(bars, diff):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (0.01 if val > 0 else -0.03),
                     f'{val:+.0%}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        
        # 3. Trend Lines
        ax3.plot(self.randomness_values, self.reparto_accuracy, 'o-', 
                label='Reparto', linewidth=3, markersize=8, color='#2E86AB')
        ax3.plot(self.randomness_values, self.sentiment_accuracy, '^-', 
                label='Sentiment', linewidth=3, markersize=8, color='#A23B72')
        ax3.set_title('Trend Performance', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Robustezza (inverso della varianza)
        robustezza_reparto = 1 - np.std(self.reparto_accuracy)
        robustezza_sentiment = 1 - np.std(self.sentiment_accuracy)
        bars = ax4.bar(['Reparto', 'Sentiment'], 
                      [robustezza_reparto, robustezza_sentiment], 
                      color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax4.set_title('Robustezza al Rumore', fontweight='bold')
        ax4.set_ylabel('Indice di Robustezza')
        ax4.grid(True, alpha=0.3)
        
        # Aggiungi valori
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('SmartHotels: Dashboard Analisi RANDOMNESS', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'dashboard_complete.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self):
        """Genera tutti i grafici."""
        print("Generazione visualizzazioni SmartHotels...")
        print(f"Salvando in: {self.plots_dir.absolute()}")
        
        self.plot_accuracy_comparison()
        print("✅ Grafico accuracy comparison salvato")
        
        self.plot_paradox_visualization()
        print("✅ Grafico paradosso salvato")
        
        self.plot_f1_heatmap()
        print("✅ Heatmap F1-score salvata")
        
        self.plot_dashboard()
        print("✅ Dashboard completo salvato")
        
        print(f"\nTutti i grafici salvati in: {self.plots_dir.absolute()}")


def main():
    """Funzione principale per generare tutte le visualizzazioni."""
    visualizer = SmartHotelsVisualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()