"""Eksploracja danych >> Odkrywanie deskrypcyjne
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install scikit-learn-extra
   pip install yellowbrick"""

__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer, InterclusterDistance


def normalize(df):
    """Normalizacja danych z wykorzystaniem standaryzacji statystycznej"""

    scaler = StandardScaler()
    return scaler.fit_transform(df)


def central_clustering(k, df, algorithm="kmeans", n_init=25):
    """Grupowanie algorytmem centroidalnym ALBO medoidalnym
       Parametr k określa liczbę skupień
       Parametr df określa zbiór danych w formie ramki danych
       Parametr algorithm określa algorytm grupowania środkowego
       Parametr n_init określa liczbę uruchomień algorytmu z różnymi początkowymi środkami skupień"""

    if algorithm == "kmedoids":
        model = KMedoids(init="random", n_clusters=k)
    else:
        model = KMeans(init="random", n_clusters=k, n_init=n_init)

    groups = model.fit_predict(df.iloc[0:, 1:].to_numpy())
    df_grouped = pd.concat([df, pd.DataFrame(data=groups, columns=["grupa"])], axis=1)

    ss = silhouette_score(df_grouped.iloc[0:, 1:], model.labels_, metric="euclidean")

    centers = []
    for index, center in enumerate(model.cluster_centers_.round(2)):
        centers.append(["Środek skupienia " + str(index), center.tolist()])
    values = [["Algorytm", model.__class__.__name__.upper()],
              ["Liczba skupień k", model.n_clusters],
              ["WCSS", model.inertia_],
              ["Liczba iteracji:", model.n_iter_],
              ["Wskaźnik Silhouette:", ss]] + centers
    description = pd.DataFrame(data=values, columns=["", ""])

    return model, df_grouped, description


def visualize(model, df_grouped):
    """Tworzenie wykresów.
       Parametr model określa model wyznaczony algorytmem grupowania środkowego
       Parametr df_grouped to pogrupowane dane w formie ramki danych."""

    colors = ["blue", "orange", "red", "green", "magenta", ]

    # Wyznaczenie środków skupień tj. centroidów albo medoidów oraz tworzenie wykresu
    # UWAGA: Wykres SCATTER wizualizuje tylko dwie cechy
    centers = model.cluster_centers_
    plt.figure(figsize=(15, 10))
    plt.title("Grouping plot with marked cluster centers")
    for i in range(model.n_clusters):
        cluster = df_grouped[df_grouped.grupa == i]
        plt.scatter(cluster.iloc[:, 1], cluster.iloc[:, 2], color=colors[i])
    plt.scatter(centers[:, 0], centers[:, 1], color="grey", s=160, marker="x")
    plt.show()

    # Tworzenie zarysu
    plt.figure(figsize=(15, 10))
    sv = SilhouetteVisualizer(model)
    sv.fit(df_grouped.iloc[0:, 1:-1].to_numpy())
    sv.show()

    # Tworzenie map odległości międzyskupieniowych
    plt.figure(figsize=(15, 10))
    v = InterclusterDistance(model, embedding="mds")
    v.fit(df_grouped.iloc[0:, 1:].to_numpy())
    v.show()


def determine_optimal_number_of_groups(model, scaled, k_max=8):
    """Wyznaczenie optymalnej liczby grup metodą łokcia (ang. elbow method):
       a) na podstawie błędu WCSS (ang. Within-Cluster Sum of Squared Errors)
       b) na podstawie współczynnika zarysu (ang. silhouette score)
       Analiza realizowana dla liczby skupień k = 2...k_max"""

    if model.__class__.__name__ == "KMedoids":
        model_evaluation = KMedoids(init=model.init)
    else:
        model_evaluation = KMeans(init=model.init, n_init=model.n_init)

    v = KElbowVisualizer(model_evaluation, k=(2, k_max + 1))
    plt.figure(figsize=(15, 10))
    plt.title("Optymalna liczba skupień określana metodą łokcia (ang. elbow method) na podstawie błędu WCSS")
    v.fit(scaled)
    plt.show()

    v = KElbowVisualizer(model_evaluation, k=(2, k_max + 1), metric="silhouette", timings=True)
    plt.figure(figsize=(15, 10))
    plt.title("Optymalna liczba skupień określana na podstawie współczynnika zarysu (ang. silhouette score)")
    v.fit(scaled)
    plt.show()


def evaluate_on_silhouette(model, scaled, k_max=5):
    """Ewaluacja grupowania z wykorzystaniem zarysu (ang. Silhouette)
       Grupowanie oraz wizualizacja jest realizowana dla liczby skupień k = 2...k_max"""

    figure, ax = plt.subplots(2, 2, figsize=(15, 10))

    sv = SilhouetteVisualizer(model)
    for k in range(2, k_max + 1):
        if model.__class__.__name__ == "KMedoids":
            model_evaluation = KMedoids(init=model.init, n_clusters=k, random_state=128)
        else:
            model_evaluation = KMeans(init=model.init, n_clusters=k, n_init=model.n_init, random_state=128)

        quotient, remainder = divmod(k, 2)
        sv = SilhouetteVisualizer(model_evaluation, ax=ax[quotient - 1][remainder])
        sv.fit(scaled)
    sv.show()


def evaluate_on_intercluster_distance_maps(model, scaled, k_max=5):
    """Ewaluacja grupowania z wykorzystaniem map odległości międzygrupowych (ang. Intercluster Distance Maps)
       Grupowanie oraz wizualizacja jest realizowana dla liczby skupień k = 2...k_max"""

    figure, ax = plt.subplots(2, 2, figsize=(15, 10))

    v = InterclusterDistance(model)
    for k in range(2, k_max + 1):
        if model.__class__.__name__ == "KMedoids":
            model_evaluation = KMedoids(init=model.init, n_clusters=k, random_state=128)
        else:
            model_evaluation = KMeans(init=model.init, n_clusters=k, n_init=model.n_init, random_state=128)

        quotient, remainder = divmod(k, 2)
        v = InterclusterDistance(model_evaluation, ax=ax[quotient - 1][remainder])
        v.fit(scaled)
    v.show()


def evaluate(model, scaled, k_max=5):
    """Ewaluacja grupowania z wykorzystaniem:
       a) zarysu (ang. Silhouette)
       b) map odległości międzygrupowych (ang. Intercluster Distance Maps)
       Grupowanie oraz wizualizacja jest realizowana dla liczby skupień k = 2...k_max"""

    figure, ax = plt.subplots(4, 2, figsize=(15, 20))

    v = InterclusterDistance(model)
    sv = SilhouetteVisualizer(model)
    for k in range(2, k_max + 1):
        if model.__class__.__name__ == "KMedoids":
            model_evaluation = KMedoids(init=model.init, n_clusters=k, random_state=128)
        else:
            model_evaluation = KMeans(init=model.init, n_clusters=k, n_init=model.n_init, random_state=128)

        quotient, remainder = divmod(k, 2)
        v = InterclusterDistance(model_evaluation, ax=ax[quotient - 1][remainder])
        v.fit(scaled)
        sv = SilhouetteVisualizer(model_evaluation, ax=ax[quotient - 1 + 2][remainder])
        sv.fit(scaled)

    v.show()
    sv.show()


def visualize_dendrogram(scaled, linkage_method="ward", metric="euclidean"):
    """Wizualizacja dendrogramem grupowania hierarchicznego
       Parametr scaled określa kolekcję numpy.ndarray"""

    plt.figure(figsize=(15, 10))
    plt.title("DENDROGRAM\nMetryka obliczania odległości " + metric.upper() + "\nMiara odległości między skupieniami " + linkage_method.upper())
    plt.ylabel("Odległość " + metric.upper())
    # plt.hlines(y=25000, xmin=0, xmax=2000, lw=2, linestyles='-', color="green")
    # plt.text(y=26000, x=62.5, s="Linia pozioma przecina 3 linie pionowe!", fontsize=14, color="green")
    plt.grid(True)
    hlink = hierarchy.linkage(scaled, method=linkage_method, metric=metric)
    dendrogram = hierarchy.dendrogram(hlink)
    plt.show()
    return dendrogram


def visualize_dendrograms(scaled, metric="euclidean"):
    """Wizualizacja dendrogramem grupowania hierarchicznego wg różnych miar odległości skupień/grup
       Parametr scaled określa kolekcję numpy.ndarray"""

    figure, ax = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle("DENDROGRAM\nMetryka obliczania odległości " + metric.upper())

    linkages = ["single", "complete", "average", "ward"]

    for k, linkage in enumerate(linkages):
        quotient, remainder = divmod(k, 2)
        hlink = hierarchy.linkage(scaled, method=linkage, metric=metric)
        subplot = ax[quotient][remainder]
        subplot.title.set_text("Miara odległości między skupieniami " + linkage.upper())
        subplot.set_ylabel("Odległość " + metric.upper())
        dendrogram = hierarchy.dendrogram(hlink, ax=subplot)

    plt.show()


def hierarchical_clustering(k, df, linkage_method="ward", metric="euclidean"):
    """Grupowanie hierarchiczne
       Parametr k określa liczbę skupień
       Parametr df określa zbiór danych w formie ramki danych
       Parametr linkage określa miarę odelgłości między skupieniami {single, complete, average, ward}"""

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, compute_distances=True, metric=metric)

    groups = model.fit_predict(df.iloc[0:, 1:].to_numpy())
    df_grouped = pd.concat([df, pd.DataFrame(data=groups, columns=["grupa"])], axis=1)

    values = [["Algorytm", model.__class__.__name__.upper()],
              ["Liczba skupień k", model.n_clusters],
              ["Miara odległości między skupieniami", model.linkage]]
    description = pd.DataFrame(data=values, columns=["", ""])

    return model, df_grouped, description
