"""
Team Name Mappings: Understat ↔ football-data.co.uk
====================================================
Hardcoded mappings for reliable dataset merging.
Understat names (keys) → football-data.co.uk names (values).
"""

# ---------------------------------------------------------------------------
# England — Premier League
# ---------------------------------------------------------------------------
EPL_MAPPING = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Tottenham": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton": "Brighton",
    "Aston Villa": "Aston Villa",
    "West Ham": "West Ham",
    "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace",
    "Wolverhampton Wanderers": "Wolves",
    "Fulham": "Fulham",
    "Everton": "Everton",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
    "Bournemouth": "Bournemouth",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield United",
    "Luton": "Luton",
    "Leicester": "Leicester",
    "Leeds": "Leeds",
    "Southampton": "Southampton",
    "Watford": "Watford",
    "Norwich": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Ipswich": "Ipswich",
    "Huddersfield": "Huddersfield",
    "Cardiff": "Cardiff",
    "Stoke City": "Stoke",
    "Swansea": "Swansea",
}

# ---------------------------------------------------------------------------
# Spain — La Liga
# ---------------------------------------------------------------------------
LA_LIGA_MAPPING = {
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Villarreal": "Villarreal",
    "Athletic Club": "Ath Bilbao",
    "Sevilla": "Sevilla",
    "Osasuna": "Osasuna",
    "Mallorca": "Mallorca",
    "Girona": "Girona",
    "Getafe": "Getafe",
    "Celta Vigo": "Celta",
    "Cadiz": "Cadiz",
    "Valencia": "Valencia",
    "Alaves": "Alaves",
    "Las Palmas": "Las Palmas",
    "Granada": "Granada",
    "Almeria": "Almeria",
    "Rayo Vallecano": "Vallecano",
    "Espanyol": "Espanol",
    "Real Valladolid": "Valladolid",
    "Leganes": "Leganes",
    "Levante": "Levante",
    "Eibar": "Eibar",
    "Huesca": "Huesca",
    "Elche": "Elche",
}

# ---------------------------------------------------------------------------
# Germany — Bundesliga
# ---------------------------------------------------------------------------
BUNDESLIGA_MAPPING = {
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "Borussia M.Gladbach": "M'gladbach",
    "SC Freiburg": "Freiburg",
    "VfL Wolfsburg": "Wolfsburg",
    "1899 Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "FC Augsburg": "Augsburg",
    "Werder Bremen": "Werder Bremen",
    "FSV Mainz 05": "Mainz",
    "1. FC Koeln": "FC Koln",
    "VfL Bochum": "Bochum",
    "FC Heidenheim": "Heidenheim",
    "SV Darmstadt 98": "Darmstadt",
    "Hertha Berlin": "Hertha",
    "Arminia Bielefeld": "Bielefeld",
    "SpVgg Greuther Fuerth": "Greuther Furth",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "SC Paderborn 07": "Paderborn",
    "FC Schalke 04": "Schalke 04",
    "Holstein Kiel": "Holstein Kiel",
    "FC St. Pauli": "St Pauli",
}

# ---------------------------------------------------------------------------
# Italy — Serie A
# ---------------------------------------------------------------------------
SERIE_A_MAPPING = {
    "Inter": "Inter",
    "AC Milan": "Milan",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Atalanta": "Atalanta",
    "AS Roma": "Roma",
    "Lazio": "Lazio",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Monza": "Monza",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Empoli": "Empoli",
    "Cagliari": "Cagliari",
    "Verona": "Verona",
    "Lecce": "Lecce",
    "Genoa": "Genoa",
    "Frosinone": "Frosinone",
    "Salernitana": "Salernitana",
    "Sampdoria": "Sampdoria",
    "Spezia": "Spezia",
    "Venezia": "Venezia",
    "Cremonese": "Cremonese",
    "Benevento": "Benevento",
    "Crotone": "Crotone",
    "Parma Calcio 1913": "Parma",
    "SPAL 2013": "Spal",
    "Brescia": "Brescia",
    "Como": "Como",
}

# ---------------------------------------------------------------------------
# France — Ligue 1
# ---------------------------------------------------------------------------
LIGUE_1_MAPPING = {
    "Paris Saint Germain": "Paris SG",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Lyon": "Lyon",
    "Nice": "Nice",
    "Lens": "Lens",
    "Rennes": "Rennes",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Reims": "Reims",
    "Brest": "Brest",
    "Le Havre": "Le Havre",
    "Lorient": "Lorient",
    "Metz": "Metz",
    "Clermont Foot": "Clermont",
    "Angers": "Angers",
    "Auxerre": "Auxerre",
    "Ajaccio": "Ajaccio",
    "Troyes": "Troyes",
    "Bordeaux": "Bordeaux",
    "Saint-Etienne": "St Etienne",
    "Dijon": "Dijon",
    "Nimes": "Nimes",
    "Amiens": "Amiens",
}

# ---------------------------------------------------------------------------
# Master mapping (all leagues combined)
# ---------------------------------------------------------------------------
ALL_MAPPINGS = {
    "EPL": EPL_MAPPING,
    "La_liga": LA_LIGA_MAPPING,
    "Bundesliga": BUNDESLIGA_MAPPING,
    "Serie_A": SERIE_A_MAPPING,
    "Ligue_1": LIGUE_1_MAPPING,
}


def get_fd_name(understat_name: str, league: str) -> str:
    """
    Convert an Understat team name to the football-data.co.uk equivalent.

    Parameters
    ----------
    understat_name : str
        Team name as it appears on Understat.
    league : str
        Understat league identifier (e.g., 'EPL', 'La_liga').

    Returns
    -------
    str
        The football-data.co.uk team name, or the original name if no mapping exists.
    """
    mapping = ALL_MAPPINGS.get(league, {})
    return mapping.get(understat_name, understat_name)


def get_understat_name(fd_name: str, league: str) -> str:
    """
    Reverse lookup: football-data.co.uk name → Understat name.

    Parameters
    ----------
    fd_name : str
        Team name as it appears in football-data.co.uk CSVs.
    league : str
        Understat league identifier.

    Returns
    -------
    str
        The Understat team name, or the original name if no mapping exists.
    """
    mapping = ALL_MAPPINGS.get(league, {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(fd_name, fd_name)
