import re

TOPICS = {
    "Klima & Energie": [
        "klima", "climat", "climate", "co2", "treibhausgas", "emission",
        "erneuerbar", "renouvelable", "renewable", "solar", "wind", "photovoltaik",
        "netto-null", "net zero", "energiewende", "stromversorgung", "fossi",
        "wärmepumpe", "gebäudesanierung", "energieeffizienz", "atomkraft",
        "kernenergie", "wasserkraft", "strommarkt", "energiegesetz",
    ],
    "Umwelt & Biodiversität": [
        "umwelt", "environnement", "environment", "biodiversit", "ökologi",
        "pestizid", "pesticide", "gewässer", "wald", "forest", "abfall", "waste",
        "gletscher", "naturgefahren", "hochwasser", "artenschutz", "naturschutz",
        "luftqualität", "boden", "landschaft", "moor", "schutzgebiet",
        "ökosystem", "artensterben", "nitrat", "mikroplastik",
    ],
    "Landwirtschaft & Ernährung": [
        "landwirtschaft", "agriculture", "bäuerlich", "bauern", "tierwohl",
        "tierhaltung", "dünger", "biobauer", "bio", "trinkwasser",
        "direktzahlungen", "pflanzenschutz", "lebensmittel", "ernährung",
        "agrar", "bauernhof", "nutztier", "massentierhaltung",
    ],
    "Wirtschaft & Arbeit": [
        "wirtschaft", "économie", "economy", "arbeit", "travail", "work",
        "gewerbe", "industrie", "kmu", "unternehmen", "mindestlohn",
        "lohngleichheit", "lohn", "arbeitszeit", "arbeitslosigkeit",
        "konjunktur", "wettbewerb", "markt", "freihandel",
    ],
    "Soziales & Gesundheit": [
        "sozial", "soziale", "social", "gesundheit", "santé", "health",
        "krankenkasse", "prämie", "altersvorsorge", "rente", "ahv", "iv",
        "sozialhilfe", "invalidität", "pflege", "spital", "medikament",
        "krankenversicherung", "ergänzungsleistung", "familienleistung",
        "kinderbetreuung", "mutterschaft", "armut",
    ],
    "Bildung & Forschung": [
        "bildung", "éducation", "education", "schule", "université", "universität",
        "hochschule", "forschung", "recherche", "research", "innovation",
        "berufsbildung", "stipendium", "lehrplan", "wissenschaft",
    ],
    "Finanzen & Steuern": [
        "steuer", "impôt", "tax", "finanzen", "budget", "haushalt",
        "schulden", "defizit", "mehrwertsteuer", "verrechnungssteuer",
        "erbschaftssteuer", "unternehmenssteuer", "finanzausgleich",
        "staatsausgaben", "sparmassnahmen", "verschuldung", "bundeshaushalt",
    ],
    "Migration & Integration": [
        "migration", "asyl", "asile", "asylum", "flüchtling", "réfugié",
        "ausländer", "einbürgerung", "integration", "aufenthaltsrecht",
        "grenzschutz", "schengen", "dublin", "rückkehr", "wegweisung",
    ],
    "Sicherheit & Justiz": [
        "sicherheit", "sécurité", "security", "armee", "armée", "army",
        "polizei", "police", "justiz", "strafrecht", "gericht", "datenschutz",
        "überwachung", "nachrichtendienst", "terrorismus", "kriminalität",
        "strafvollzug", "cybersicherheit", "verteidigung",
    ],
    "Infrastruktur & Verkehr": [
        "infrastruktur", "verkehr", "transport", "bahn", "strasse",
        "autobahn", "öffentlicher verkehr", "öv", "velo", "velowegnetz",
        "flughafen", "luftfahrt", "schifffahrt", "elektroauto",
        "elektromobil", "parkplatz", "stau", "bahnausbau",
    ],
    "Digitalisierung & Medien": [
        "digital", "digitalisierung", "künstliche intelligenz", "ki", "ai",
        "internet", "daten", "plattform", "medien", "medienförderung",
        "social media", "e-id", "blockchain", "cybersicherheit", "algorithmus",
        "datenschutz", "fernmeldegesetz",
    ],
    "Aussenpolitik & Entwicklung": [
        "aussenpolitik", "politique étrangère", "foreign policy",
        "eu", "europa", "bilaterale", "uno", "sanktion", "entwicklungshilfe",
        "humanitär", "menschenrechte", "diplomatie", "völkerrecht",
        "freihandelsabkommen", "aussenwirtschaft", "neutralität",
    ],
    "Nachhaltigkeit & ESG": [
        "nachhaltig", "durable", "sustainab", "esg", "taxonomie",
        "kreislaufwirtschaft", "circular", "unternehmensverantwortung",
        "sorgfaltspflicht", "lieferkette", "supply chain", "diligence",
        "berichterstattung", "purpose", "gemeinwohl", "gemeinnützig",
        "sozialunternehmen", "social enterprise",
    ],
    "Demokratie & Institutionen": [
        "volksinitiative", "referendum", "volksrechte", "demokratie",
        "föderalismus", "parlament", "bundesrat", "verfassung",
        "transparenz", "lobbyismus", "politikfinanzierung", "wahlrecht",
        "abstimmung", "initiative", "parlamentsrecht",
    ],
}

_patterns = {
    topic: re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE)
    for topic, kws in TOPICS.items()
}

def classify(text: str) -> list[str]:
    return [t for t, pat in _patterns.items() if pat.search(text)]

def any_topic(text: str) -> bool:
    return any(pat.search(text) for pat in _patterns.values())
