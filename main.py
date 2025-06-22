
__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import pandas as pd
import warnings
import logging

from data.datasources import get_seasonality_data
from data.visualizations import plot_seasonality_heatmaps, plot_country_analysis
from clustering import run_affinity_propagation_clustering, analysis_purchase_by_sender_query
from forecasting import forecast_monthly_volume, evaluate_and_plot_forecast

# --- Uciszanie ostrzeżeń i nadmiernych logów ---
warnings.filterwarnings("ignore", category=FutureWarning)

noisy_modules = ['cmdstanpy', 'prophet', 'fbprophet', 'matplotlib']
for module in noisy_modules:
    logger = logging.getLogger(module)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# Globalny poziom logowania – tylko ostrzeżenia i błędy
logging.basicConfig(level=logging.WARNING)


def main():
    # --- 1. Analiza deskrypcyjna (sezonowość i przegląd ogólny) ---
    data = get_seasonality_data()
    df = pd.DataFrame(data, columns=['rok', 'miesiac', 'region', 'liczba_przesylek'])

    plot_seasonality_heatmaps(df)
    plot_country_analysis(df)

    # --- 2. Analiza odkrywcza — klasteryzacja nadawców (Affinity Propagation) ---
    print("\n[INFO] Rozpoczynanie analizy skupień z użyciem Affinity Propagation...\n")
    run_affinity_propagation_clustering(analysis_purchase_by_sender_query)

    # --- 3. Prognozowanie wolumenu przesyłek ---
    print("\n[INFO] Rozpoczynanie prognozowania wolumenu przesyłek...\n")
    model, forecast, mape, train, test, test_forecast, *_ = forecast_monthly_volume()
    print(f"[INFO] Prognozowanie zakończone, MAPE: {mape:.2f}%")

    # --- 4. Ocena jakości modelu i wykres porównawczy ---
    mae, rmse = evaluate_and_plot_forecast(train, test, test_forecast)
    print(f"[INFO] MAE: {mae:.2f}, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
