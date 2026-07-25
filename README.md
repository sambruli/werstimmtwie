# 🏛️ Wer stimmt wie?

Ein persönliches Nebenprojekt, das Abstimmungsverhalten und Abweichungen im Schweizer
Parlament sichtbar macht — jenseits von Parteiprogrammen und Selbstdarstellung.

Die App zeigt, wie einzelne Parlamentsmitglieder und ganze Fraktionen bei jeder
einzelnen Abstimmung votiert haben, und wo jemand von der eigenen Fraktionslinie
abweicht.

## Was die App kann

Fünf Ansichten, erreichbar über die Sidebar:

- **Home** — Überblick über die Legislatur, Kurzanleitung und Methodik
- **Fraktion** — Kohäsionsvergleich der Fraktionen, Abweichungsraten der einzelnen
  Mitglieder, Richtungsanalyse (mit welchen anderen Fraktionen wird bei Abweichungen
  gestimmt?) und ein Kammernvergleich NR vs. SR
- **Mitglied** — Detailprofil eines Parlamentsmitglieds: Gesamtabweichung vom
  Fraktionskurs, Abweichung nach Thema und Session, Richtungsanalyse, vollständige
  Abstimmungshistorie und ein Vergleich mit den Fraktionskolleg:innen in der anderen
  Kammer
- **Thema** — Abstimmungsverhalten der Fraktionen zu einem politischen Thema
  (z.B. Klima, Migration, Soziales)
- **Geschäft** — Suche nach Geschäftsnummer oder Stichwort, mit dem Abstimmungsverhalten
  jeder Fraktion und jedes Mitglieds zu diesem Geschäft

### Methodik

Eine Abstimmung gilt als **Abweichung**, wenn ein Mitglied anders votierte als die
Mehrheit der *anwesenden* Fraktionsmitglieder bei derselben Abstimmung. Abwesenheit,
Entschuldigung und Nichtteilnahme zählen nicht als Abweichung — nur Ja, Nein und
Enthaltung fliessen in die Berechnung ein. Ein Geschäft (z.B. «23.456») kann mehrere
Einzelabstimmungen umfassen (Eintreten, Rückweisung, Detailberatung, Gesamtabstimmung);
jede wird separat ausgewertet.

## Datenquelle

Die Daten stammen aus der [Open-Data-API des Schweizer Parlaments](https://www.parlament.ch/de/services/open-data-webservices)
(parlament.ch, OData) und werden über [`swissparlpy`](https://pypi.org/project/swissparlpy/)
abgerufen. Abgedeckt ist Legislatur 52 (ab Dezember 2023).

## Tech-Stack

- [Streamlit](https://streamlit.io/) für das Frontend
- SQLite für das Scraping/lokale Arbeiten, [Parquet](https://parquet.apache.org/)-Dateien
  (`votes.parquet`, `members.parquet`, `sessions.parquet`, `biz_tags.parquet`) für das
  Deployment
- pandas für die Aggregation, Plotly für die Charts

## Projektstruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App mit allen fünf Seiten |
| `analysis.py` | Abweichungs- und Kohäsionsberechnungen |
| `scraper.py` | Holt Mitglieder, Sessionen und Abstimmungen von der Parlaments-API |
| `db.py` | SQLite-Schema und Verbindung |
| `export.py` | Exportiert die SQLite-DB nach Parquet fürs Deployment |
| `ci_scrape.py` | Skript für den wöchentlichen GitHub-Action-Scrape |
| `topics.py` | Themen-Klassifizierung (Klima, Migration, Soziales, …) |
| `xls_importer.py` | Import historischer Abstimmungsdaten aus Excel |

## Lokal starten

Voraussetzung: Python 3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Unter Windows kann alternativ `run.bat` verwendet werden — das Skript erstellt bei
Bedarf ein virtuelles Environment, installiert die Abhängigkeiten und startet die App.

Ohne lokale `swiss_parl.db` greift die App automatisch auf die mitgelieferten
Parquet-Dateien zurück (praktisch für Streamlit Cloud oder einen schnellen lokalen Test
ohne eigenes Scraping).

## Daten aktualisieren

```bash
python -c "from scraper import full_scrape; full_scrape()"   # oder update_scrape() für Deltas
python export.py                                              # SQLite → Parquet
```

Anschliessend die aktualisierten `*.parquet`-Dateien committen. Ein
[GitHub-Action-Workflow](.github/workflows/scrape.yml) erledigt das automatisch: er
läuft jeden Montag um 05:00 UTC, scrapt neue Sessionen und committet die aktualisierten
Parquet-Dateien.

## Hintergrund

Diese App ist ein persönliches Nebenprojekt — kein professionelles Produkt, keine
Organisation dahinter. Infrastrukturkosten werden aus dem persönlichen Budget gedeckt;
bitte die App nur mit Personen teilen, denen du vertraust.
