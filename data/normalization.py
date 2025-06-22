"""Eksploracja danych >> Normalizacja danych
   Wymagane pakiety:
   pip install scikit-learn
"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.0.0"

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize(df):
    """Normalizacja danych z wykorzystaniem skalowania Min-Max"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled
