

__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

from data.datasources import connect
from data.normalization import normalize
from data.mining.descriptive import (
    central_clustering,
    hierarchical_clustering,
    visualize,
    visualize_dendrogram,
    visualize_dendrograms,
    determine_optimal_number_of_groups,
    evaluate,
    evaluate_on_silhouette
)

# --- Zapytania SQL ---
analysis_purchase_by_sender_query = """
WITH nadawcy AS (
    SELECT
        imie || ' ' || nazwisko AS nadawca,
        CASE plec WHEN 'M' THEN 1 WHEN 'K' THEN 2 ELSE NULL END AS plec,
        region,
        CAST(EXTRACT(YEAR FROM age(data_urodzenia)) AS integer) AS wiek,
        CAST(100.0 * COUNT(*) FILTER (WHERE szybkosc = 'priorytetowa') / COUNT(*) AS integer) AS priorytet,
        CAST(100.0 * COUNT(*) FILTER (WHERE szybkosc = 'ekonomiczna') / COUNT(*) AS integer) AS ekonomiczna,
        CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka listowa') / COUNT(*) AS integer) AS list,
        CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paczkowa') / COUNT(*) AS integer) AS paczka,
        CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paczkomatowa') / COUNT(*) AS integer) AS paczkomat,
        CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paletowa') / COUNT(*) AS integer) AS paleta,
        CAST(COUNT(*) AS integer) AS liczba,
        CAST(SUM(cena) AS integer) AS wartosc
    FROM poczta_olap.sprzedaz
    NATURAL JOIN poczta_olap.nadawca
    NATURAL JOIN poczta_olap.usluga
    GROUP BY 1, 2, 3, 4
)
SELECT nadawca, wiek, wartosc
FROM nadawcy
WHERE region = 'małopolskie';
"""

# --- Central Clustering Experiment ---
def make_experiment_central_clustering(algorithm, n_clusters, query, n_init=1):
    warnings.filterwarnings("ignore")

    rs = connect(query)
    df = pd.DataFrame(rs, columns=["nadawca", "wiek", "wartosc"])
    df = df.set_index("nadawca")
    print("Dane oryginalne:\n", df, os.linesep)

    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "wiek": scaled[:, 0], "wartosc": scaled[:, 1]})
    print("Dane znormalizowane:\n", df_scaled, os.linesep)

    model, df_grouped, inertia = central_clustering(n_clusters, df_scaled, algorithm, n_init)
    print("Wartość funkcji celu (inertia/dissimilarity):", inertia, os.linesep)
    print("Podział na grupy:\n", df_grouped, os.linesep)

    for i in range(model.n_clusters):
        print(f"Grupa {i}:", df_grouped[df_grouped.grupa == i], os.linesep)

    visualize(model, df_grouped)
    determine_optimal_number_of_groups(model, scaled, max_k=5)
    evaluate(model, scaled)
    evaluate_on_silhouette(model, scaled)


# --- Hierarchical Clustering Experiment ---
def make_experiment_hierarchical_clustering(n_clusters, query, linkage_method, metric):
    warnings.filterwarnings("ignore")

    rs = connect(query)
    df = pd.DataFrame(rs, columns=["nadawca", "wiek", "wartosc"])
    df = df.set_index("nadawca")
    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "wiek": scaled[:, 0], "wartosc": scaled[:, 1]})

    visualize_dendrogram(scaled, linkage_method=linkage_method, metric=metric)

    model, df_grouped, d = hierarchical_clustering(n_clusters, df_scaled, linkage_method, metric)
    print("Wartość funkcji celu:", d, os.linesep)
    print("Podział na grupy:\n", df_grouped, os.linesep)

    for i in range(model.n_clusters):
        print(f"Grupa {i}:", df_grouped[df_grouped.grupa == i], os.linesep)

    visualize_dendrograms(scaled, metric)


# --- Affinity Propagation Experiment ---


def run_affinity_propagation_clustering(query):
    rs = connect(query)
    df = pd.DataFrame(rs, columns=["nadawca", "wiek", "wartosc"])
    df = df.set_index("nadawca")

    # ✂️ ograniczamy dane do 100 nadawców, aby uniknąć 385 klastrów
    # df = df.sample(n=100, random_state=42)

    # 🔄 normalizacja danych
    scaled = normalize(df)

    # ⚙️ Affinity Propagation z dampingiem 0.9
    model = AffinityPropagation(damping=0.9)
    model.fit(scaled)

    labels = model.labels_
    df_grouped = df.copy()
    df_grouped["grupa"] = labels

    print(f"[INFO] Liczba wykrytych skupień: {len(set(labels))}")
    print(df_grouped.groupby("grupa").agg(["count", "mean", "std"]))
    print()

    # 📈 wykres scatter z klastrami
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=scaled[:, 0], y=scaled[:, 1], hue=labels, palette='tab10')
    plt.title("Affinity Propagation — Klasteryzacja nadawców")
    plt.xlabel("wiek (znormalizowany)")
    plt.ylabel("wartość (znormalizowana)")
    plt.grid(True)
    plt.legend(title='Grupa', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # 📊 histogram liczebności klastrów
    plt.figure(figsize=(8, 4))
    df_grouped['grupa'].value_counts().sort_index().plot(kind='bar')
    plt.title("Liczebność klastrów - Affinity Propagation")
    plt.xlabel("Grupa")
    plt.ylabel("Liczba nadawców")
    plt.grid(axis='y')
    plt.show()

    # 📊 Ewaluacja - silhouette score
    silhouette = silhouette_score(scaled, labels)
    print(f"[EVAL] Silhouette Score: {silhouette:.2f}")


# --- Wywołanie eksperymentu ---
if __name__ == "__main__":
    make_experiment_central_clustering("kmedoids", 3, analysis_purchase_by_sender_query)
    # make_experiment_central_clustering("kmeans", 3, analysis_purchase_by_sender_query)
    # make_experiment_hierarchical_clustering(3, analysis_purchase_by_sender_query, "ward", metric="euclidean")
    # run_affinity_propagation_clustering(analysis_purchase_by_sender_query)
