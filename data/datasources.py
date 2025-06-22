
__author__ = "Mateusz Putała"
__copyright__ = "Katedra Informatyki"
__version__ = "2.0.0"

import psycopg2 as db
import logging

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parametry połączenia
HOST = "195.150.230.208"
PORT = "5432"
DATABASE = "2024_student_e"
USER = "2024_student_e"
PASSWORD = ""

# Zapytanie do analizy sezonowości przesyłek
SEASONALITY_QUERY = """
SELECT
    c.rok,
    c.miesiac,
    n.region,
    COUNT(*) AS liczba_przesylek
FROM poczta_olap.sprzedaz s
JOIN poczta_olap.czas c ON s.id_czasu_nadania = c.id_czasu
JOIN poczta_olap.nadawca n ON s.id_nadawcy = n.id_nadawcy
GROUP BY c.rok, c.miesiac, n.region
ORDER BY c.rok, c.miesiac, n.region;
"""

# Zapytanie do analizy zakupów wg nadawcy (do klasteryzacji)
analysis_purchase_by_sender_query = """
SELECT
    n.id_nadawcy AS nadawca_id,
    n.region,
    COUNT(*) AS liczba_przesylek
FROM poczta_olap.sprzedaz s
JOIN poczta_olap.nadawca n ON s.id_nadawcy = n.id_nadawcy
GROUP BY n.id_nadawcy, n.region
ORDER BY n.id_nadawcy, n.region;
"""

def connect(query, host=HOST, port=PORT, database=DATABASE, user=USER, password=PASSWORD):
    """
    Wykonuje zapytanie SQL i zwraca wynik w postaci listy krotek.
    """
    try:
        con = db.connect(host=host, port=port, database=database, user=user, password=password)
        cursor = con.cursor()
        cursor.execute(query)
        rs = cursor.fetchall()
        cursor.close()
        con.close()
        logger.info("Zapytanie wykonane poprawnie.")
        return rs
    except Exception as e:
        logger.error(f"Błąd podczas połączenia z bazą danych: {e}")
        return []

# Aliasy do wygodnego wywołania
def get_seasonality_data():
    """Zwraca dane do analizy sezonowości przesyłek."""
    return connect(SEASONALITY_QUERY)

def run_query(query):
    """Alias do connect() — uniwersalne wykonanie zapytania."""
    return connect(query)
