"""Eksploracja danych >> Odkrywanie predykcyjne >> Predykcja
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install seaborn
   pip install prophet
   pip install statsmodels"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.5.0"

from data.mining.predictive import *
from data.datasources import *
from prophet import Prophet
from scipy import stats
from statsmodels.tools.eval_measures import rmse

import pandas as pd
import numpy as np
import datetime as dt
import warnings
import os

"""Zapytanie do analizy sprzedaży wg lat"""
analysis_sale_by_year_query = "SELECT rok, miesiac, CAST(COUNT(id_uslugi) AS integer) AS liczba, CAST(SUM(cena) AS integer) AS wartosc " \
                              "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                              "INNER JOIN poczta_olap.czas ON czas.id_czasu = sprzedaz.id_czasu_nadania " \
                              "WHERE rok BETWEEN 2008 AND 2019 " \
                              "GROUP BY rok, miesiac " \
                              "ORDER BY rok, miesiac"

def make_experiment_prediction(query):
    """Eksperyment predykcji"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    # Utworzenie ramki danych
    df = pd.DataFrame(rs, columns=["rok", "miesiac", "liczba", "wartosc"])
    print(df, os.linesep)

    x_feature = "liczba"
    y_feature = "wartosc"
    xlabel = "Liczba usług"
    ylabel = "Wartość usług"
    title = "Sprzedaż usług"

    # Wykres punktowy kompletnego zbioru pobranych danych
    visualize_plot_scatter(x_plot=None, y_plot=None, x_scatter=df.loc[:, x_feature], y_scatter=df.loc[:, y_feature],
                           title=title, xlabel=xlabel, ylabel=ylabel, plot_color="orange", scatter_color="green")

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR UCZĄCEGO
    df_train = df[df.rok <= 2016]
    print(df_train.head, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR TESTOWEGO
    df_test = df[df.rok > 2016]
    print(df_test.head, os.linesep)

    print("Rozmiar ZBIORU UCZĄCEGO: ", df_train.loc[:, y_feature].count() / df.loc[:, y_feature].count())
    print("Rozmiar ZBIORU TESTOWEGO: ", df_test.loc[:, y_feature].count() / df.loc[:, y_feature].count(), os.linesep)

    # Wykres punktowy z oznaczeniem lat ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + "\n{Zbiór UCZĄCY}", grouping="rok")

    # Wykres punktowy z oznaczeniem lat ZBIÓR TESTOWY
    visualize(df=df_test, x=x_feature, y=y_feature, title=title + "\n{Zbiór TESTOWY}", grouping="rok")

    # Wykres punktowy ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + "\n{Zbiór UCZĄCY}", regression=False)

    # Wykres punktowy z regresją liniową ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + " & " + "Regresja" + "\n{Zbiór UCZĄCY}",
              regression=True)

    # Współczynnik korelacji liniowej Pearsona
    pc = np.corrcoef(df_train[x_feature], df_train[y_feature])
    print("Współczynnik korelacji liniowej Pearsona:\n", pc, os.linesep)

    # Utworzenie modelu predykcji tj. predyktora z wykorzystaniem REGRESJI LINIOWEJ
    x_train = np.array(df_train[x_feature]).reshape((-1, 1))
    y_train = np.array(df_train[y_feature])
    model, description = predictor(x_train, y_train)
    print(description, os.linesep)

    # Wizualizacja modelu predykcji
    x = np.linspace(np.min(x_train), max(x_train), 5000)
    y = model.coef_ * x + model.intercept_
    chart_title = "MODEL PREDYKCJI\n" + title + " & " + "Regresja"
    visualize_plot_scatter(x_plot=x, y_plot=y, x_scatter=x_train, y_scatter=y_train,
                           title=chart_title, xlabel=xlabel, ylabel=ylabel, plot_color="orange", scatter_color="green")

    # Predykcja
    x_test = np.array(df_test[x_feature]).reshape((-1, 1))
    y_test = np.array(df_test[y_feature])
    df_prediction, description = prediction(model, x_test, y_test)
    print("Rezultat predykcji:")
    print(df_prediction, os.linesep)

    # Ocena [modelu] predykcji
    print(description, os.linesep)

    # Wizualizacja predykcji
    chart_title = "PREDYKCJA\n" + title + " & " + "Regresja"
    visualize_plot_scatter(x_plot=x_test, y_plot=df_prediction["y_pred"], x_scatter=x_test, y_scatter=y_test,
                           title=chart_title, xlabel=xlabel, ylabel=ylabel, plot_color="orange", scatter_color="green")


"""Uruchamienie eksperymentów"""
make_experiment_prediction(analysis_sale_by_year_query)