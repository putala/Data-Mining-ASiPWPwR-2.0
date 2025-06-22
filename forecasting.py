
__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from data.datasources import connect
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Dodatkowe importy do walidacji krzyżowej Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

# Zapytanie pobierające sumę przesyłek miesięcznie w Polsce (lata 2010-2023)
MONTHLY_VOLUME_QUERY = """
SELECT
    c.rok,
    c.miesiac,
    COUNT(*) AS liczba_przesylek
FROM poczta_olap.sprzedaz s
JOIN poczta_olap.czas c ON s.id_czasu_nadania = c.id_czasu
WHERE c.rok BETWEEN 2010 AND 2023
GROUP BY c.rok, c.miesiac
ORDER BY c.rok, c.miesiac;
"""

def forecast_monthly_volume():
    rs = connect(MONTHLY_VOLUME_QUERY)
    df = pd.DataFrame(rs, columns=['rok', 'miesiac', 'liczba_przesylek'])

    df['ds'] = pd.to_datetime(df['rok'].astype(str) + '-' + df['miesiac'].astype(str) + '-01')
    df['y'] = df['liczba_przesylek']
    df = df[['ds', 'y']]

    train = df[df['ds'] < '2023-01-01']
    test = df[df['ds'] >= '2023-01-01']

    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(train)

    future = model.make_future_dataframe(periods=12, freq='MS')  # freq='MS' = Month Start
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    plt.title("Prognoza miesięcznej liczby przesyłek")
    plt.xlabel("Data")
    plt.ylabel("Liczba przesyłek")
    plt.show()

    forecast_indexed = forecast.set_index('ds')
    test_dates = test['ds']

    # Wyciągamy prognozy dla dat z testu
    test_forecast = forecast_indexed.loc[forecast_indexed.index.isin(test_dates)].reset_index()

    y_true = test['y'].values
    y_pred = test_forecast['yhat'].values

    mape = (np.abs(y_true - y_pred) / y_true).mean() * 100

    logger.info(f"MAPE (średni błąd procentowy) na zbiorze testowym: {mape:.2f}%")

    # Wywołanie walidacji krzyżowej
    df_cv, df_metrics = forecast_cross_validation(model)

    return model, forecast, mape, train, test, test_forecast, df_cv, df_metrics


def evaluate_and_plot_forecast(train, test, test_forecast):
    # Oblicz metryki błędu
    y_true = test['y'].values
    y_pred = test_forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    logger.info(f"MAE (średni błąd bezwzględny): {mae:.2f}")
    logger.info(f"RMSE (pierwiastek błędu średniokwadratowego): {rmse:.2f}")

    # Wykres porównawczy
    plt.figure(figsize=(10,6))
    plt.plot(train['ds'], train['y'], label='Dane treningowe')
    plt.plot(test['ds'], y_true, 'o-', label='Dane testowe (rzeczywiste)')
    plt.plot(test_forecast['ds'], y_pred, 'x--', label='Prognoza')
    plt.title("Porównanie danych rzeczywistych i prognozowanych")
    plt.xlabel("Data")
    plt.ylabel("Liczba przesyłek")
    plt.legend()
    plt.grid(True)
    plt.show()

    return mae, rmse


def forecast_cross_validation(model, initial='730 days', period='180 days', horizon='365 days'):
    """
    Wykonuje walidację krzyżową modelu Prophet.

    Args:
        model: wytrenowany model Prophet
        initial: początkowy okres treningowy (domyślnie 2 lata)
        period: odstęp pomiędzy kolejnymi prognozami (domyślnie 6 miesięcy)
        horizon: okres prognozy do oceny (domyślnie 1 rok)

    Zwraca:
        df_cv - dataframe z prognozami walidacyjnymi
        df_metrics - dataframe z metrykami jakości
    """

    logger.info("[INFO] Rozpoczynam walidację krzyżową modelu Prophet...")
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel="processes")
    df_metrics = performance_metrics(df_cv)

    logger.info("[INFO] Wyniki walidacji krzyżowej (pierwsze 10 rekordów):")
    logger.info("\n" + df_metrics[['horizon', 'mape', 'rmse', 'mae']].head(10).to_string())

    # Wykres metryk w funkcji horyzontu prognozy
    plt.figure(figsize=(10,6))
    plt.plot(df_metrics['horizon'], df_metrics['mape'], label='MAPE')
    plt.plot(df_metrics['horizon'], df_metrics['rmse'], label='RMSE')
    plt.plot(df_metrics['horizon'], df_metrics['mae'], label='MAE')
    plt.xlabel('Horyzont prognozy')
    plt.ylabel('Błąd')
    plt.title('Walidacja krzyżowa modelu Prophet')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_cv, df_metrics



