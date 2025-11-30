import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Colocar todos os plots aqui!

def plot_matriz_confusao(teste, predicao, classes):
    cm = confusion_matrix(teste, predicao)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(classes), 
                yticklabels=sorted(classes),
                cbar_kws={'label': 'Quantidade de Amostras'})
    plt.xlabel('Local Predito', fontsize=12)
    plt.ylabel('Local Real', fontsize=12)
    plt.title('Matriz de Confusão - Classificação de Localização Indoor', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    print('\nGráfico salvo em outputs/confusion_matrix.png')
    plt.close()