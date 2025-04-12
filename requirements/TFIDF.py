import numpy as np
from math import log
from requirements.corpy import Corpy

class TF_IDF:
    def __init__(self, corpus: 'Corpy'):
        # Convertir a arrays de NumPy
        self.tf = np.array([
            [corpus.data[word][f'doc{i+1}'] / len(corpus.corpus[i].split()) 
             for word in corpus.vocabulary] 
            for i in range(len(corpus.corpus))
        ])
        
        self.idf = np.array([
            log(len(corpus.corpus) / (1 + corpus.data[word]['doc_freq'])) 
            for word in corpus.vocabulary
        ])
        
        # Multiplicación elemento por elemento
        self.matrix = self.tf * self.idf  # Broadcasting automático