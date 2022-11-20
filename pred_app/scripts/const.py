from sqlalchemy import create_engine
from datetime import date
from xgboost.sklearn import XGBClassifier

""" SQLAlchemy Database Engine """

ENGINE = create_engine("sqlite:///pred.db")

""" Headers and URL for retrieving daily schedule """

SCH_HEADER = {
    "user-agent": "Mozilla/5.0 (Windows NT 6.2; WOW64)"
    "Chrome/57.0.2987.133 Safari/537.36",
    "Accept-Language": "en",
    "origin": "http://stats.nba.com",
    "Referer": "https://google.com",
}

YEAR = "20" + date.today().strftime("%y")
SCH_JSON_URL = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{YEAR}/scores/00_todays_scores.json"  # pylint: disable=line-too-long
ODDS_UPDATE_URL = f"https://sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202022-23.xlsx"

""" Curent epoch settings for model """

NET_EPOCHS = 918
MASSEY_EPOCHS = 5000
DEF_CLASSIFIER = XGBClassifier(num_class=2)
FILE_NAME = "model.pk"

"""
Below section contains const lists of months used for a given year
"""

MONTHS_1999 = ["february", "march", "april", "may", "june"]
MONTHS_2012 = ["december", "january", "february", "march", "april", "may", "june"]

MONTHS_2020 = [
    "october-2019",
    "november",
    "december",
    "january",
    "february",
    "march",
    "july",
    "august",
    "september",
    "october-2020",
]

MONTHS_2021 = [
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
]

MONTHS_REG = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]

MONTHS_NO_OCT = [
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]

"""
List of various feature sets used for model
"""

NET_FULL = [
    "W_PCT",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "PACE",
    "POSS",
    "PIE",
]

TRUNC = ["FTA", "FTM"]
NET_TRUNC = [item for item in NET_FULL if item not in TRUNC]

NET_FULL_FEATURES = ["A_" + item for item in NET_FULL] + [
    "H_" + item for item in NET_FULL
]
NET_TRUNC_FEATURES = ["A_" + item for item in NET_TRUNC] + [
    "H_" + item for item in NET_TRUNC
]

MASSEY_FULL_FEATURES = ["A_" + item for item in NET_FULL + ["Massey"]] + [
    "H_" + item for item in NET_FULL + ["Massey"]
]
MASSEY_TRUNC_FEATURES = ["A_" + item for item in NET_TRUNC + ["Massey"]] + [
    "H_" + item for item in NET_TRUNC + ["Massey"]
]

SIM_PRED_FEATURES = NET_FULL_FEATURES + ["O/U", "H_ML", "A_ML", "Spread"]
SIM_PRED_DISPLAY_FEATURES = (
    ["Date", "Away", "Home", "SeasonID", "MOV", "Outcome"]
    + NET_FULL_FEATURES
    + ["Pred", "O/U", "H_ML", "A_ML", "Spread"]
)

"""
add notes
"""

TABLE_STATS_FULL = [
    "Team",
    "Record",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "EFG_PCT",
    "TS_PCT",
    "PIE",
]

"""
Features/Columns for Scraping/Transforming Boxscore Data
"""

BASIC = [
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "FT",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]

ADVANCED = [
    "TS%",
    "eFG%",
    "3PAr",
    "FTr",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "ORtg",
    "DRtg",
]

GAME = [
    "Date",
    "Time",
    "Away",
    "A-Pts",
    "Home",
    "H-Pts",
    "OT",
    "SeasonID",
    "MOV",
    "Outcome",
]

BOX_FEATURES = (
    GAME
    + ["A_" + item for item in BASIC]
    + ["A_" + item for item in ADVANCED]
    + ["H_" + item for item in BASIC]
    + ["H_" + item for item in ADVANCED]
)

MATCHUP_FEATURES = [
    "PTS",
    "AST",
    "STL",
    "BLK",
    "ORB",
    "DRB",
    "TOV",
    "PF",
    "TS%",
    "eFG%",
    "3PAr",
    "FTr",
    "ORtg",
    "DRtg",
]

AWAY_FEATURES = ["A_" + item for item in MATCHUP_FEATURES]
HOME_FEATURES = ["H_" + item for item in MATCHUP_FEATURES]

COMPARE_COLS = [
    "Team",
    "Record",
    "Massey",
    "PTS",
    "AST",
    "ORB",
    "DEF",
    "OFF",
    "NET",
    "PIE",
    "TS%",
]

ODDS_COLS = ["Team", "Fav_W%", "UD_W%", "Cover%", "Under%", "Over%", "Def_W%", "Off_W%"]

""" Constants used in ELO Calculations """

MEAN_ELO = 1500
ELO_WIDTH = 400
K_FACTOR = 64

""" Constant feature sets used by Simulator  """

SIM_FEATURES = ["Date", "Away", "Home", "SeasonID", "MOV", "Outcome", "Pred."]
