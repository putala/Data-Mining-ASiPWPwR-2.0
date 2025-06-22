
__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_seasonality_heatmaps(df):
    """
    Generuje wykresy sezonowości przesyłek dla poszczególnych regionów oraz analizę ogólną.
    """
    if df.empty:
        logger.warning("DataFrame jest pusty. Brak danych do wizualizacji.")
        return

    regions = [
        'kujawsko-pomorskie', 'lubelskie', 'łódzkie', 'podkarpackie', 'śląskie',
        'wielkopolskie', 'zachodniopomorskie', 'dolnośląskie', 'lubuskie',
        'małopolskie', 'mazowieckie', 'opolskie', 'podlaskie', 'pomorskie',
        'świętokrzyskie', 'warmińsko-mazurskie'
    ]

    df_filtered = df[(df['rok'] >= 2010) & (df['rok'] <= 2023)]

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle('Sezonowość przesyłek w regionach (2010–2023)', fontsize=18)

    vmax = df_filtered['liczba_przesylek'].max()
    vmin = df_filtered['liczba_przesylek'].min()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

    for i, region in enumerate(regions):
        ax = axes[i // cols, i % cols]
        df_region = df_filtered[df_filtered['region'] == region]
        pivot_table = df_region.pivot(index='miesiac', columns='rok', values='liczba_przesylek')

        if pivot_table.empty:
            ax.set_title(f'{region}\nBrak danych')
            ax.axis('off')
            continue

        sns.heatmap(
            pivot_table,
            annot=False,
            fmt='g',
            cmap='YlGnBu',
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None
        )
        ax.set_title(region)
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05)
    plt.show()

def plot_country_analysis(df):
    """
    Generuje wykresy analizy przesyłek dla całej Polski.
    """
    if df.empty:
        logger.warning("DataFrame jest pusty. Brak danych do wizualizacji.")
        return

    df_filtered = df[(df['rok'] >= 2010) & (df['rok'] <= 2023)]

    # 1. Sezonowość w kraju
    df_country = df_filtered.groupby(['rok', 'miesiac'])['liczba_przesylek'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_country, x='miesiac', y='liczba_przesylek', hue='rok', palette='tab20', marker='o')
    plt.title('Sezonowość przesyłek w Polsce (2010–2023)')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Miesiąc')
    plt.ylabel('Liczba przesyłek')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Roczna liczba przesyłek w regionach
    df_region_year = df_filtered.groupby(['region', 'rok'])['liczba_przesylek'].sum().reset_index()
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_region_year, x='rok', y='liczba_przesylek', hue='region', palette='tab20')
    plt.title('Roczna liczba przesyłek w regionach (2010–2023)')
    plt.xlabel('Rok')
    plt.ylabel('Liczba przesyłek')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Suma roczna przesyłek w kraju
    df_yearly = df_filtered.groupby('rok')['liczba_przesylek'].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_yearly, x='rok', y='liczba_przesylek', marker='o', color='red')
    plt.title('Roczna liczba przesyłek w Polsce (2010–2023)')
    plt.xlabel('Rok')
    plt.ylabel('Liczba przesyłek')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Heatmapa region vs rok
    pivot_region_year = df_filtered.groupby(['region', 'rok'])['liczba_przesylek'].sum().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_region_year, annot=True, fmt='g', cmap='coolwarm')
    plt.title('Suma przesyłek w regionach w podziale na lata (2010–2023)')
    plt.xlabel('Rok')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.show()






