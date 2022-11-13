"""
Docstring
"""
import time
import functools
from sqlalchemy import create_engine

ENGINE = create_engine("sqlite:///pred.db")

SCH_HEADER = {
    "user-agent": "Mozilla/5.0 (Windows NT 6.2; WOW64)"
    "Chrome/57.0.2987.133 Safari/537.36",
    "Accept-Language": "en",
    "origin": "http://stats.nba.com",
    "Referer": "https://google.com",
}


"""
Seasons vary in months played throughout the years
The below section contains various month lists and a year:months dictionary for mapping
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

#  Anything further in years than 1976 will need tweaking.
months_map = {
    1976: MONTHS_REG,
    1977: MONTHS_REG,
    1978: MONTHS_REG,
    1979: MONTHS_REG,
    1980: MONTHS_REG,
    1981: MONTHS_REG,
    1982: MONTHS_REG,
    1983: MONTHS_REG,
    1984: MONTHS_REG,
    1985: MONTHS_REG,
    1986: MONTHS_REG,
    1987: MONTHS_REG,
    1988: MONTHS_NO_OCT,
    1989: MONTHS_NO_OCT,
    1990: MONTHS_NO_OCT,
    1991: MONTHS_NO_OCT,
    1992: MONTHS_NO_OCT,
    1993: MONTHS_NO_OCT,
    1994: MONTHS_NO_OCT,
    1995: MONTHS_NO_OCT,
    1996: MONTHS_NO_OCT,
    1997: MONTHS_NO_OCT,
    1998: MONTHS_REG,
    1999: MONTHS_1999,
    2000: MONTHS_NO_OCT,
    2001: MONTHS_REG,
    2002: MONTHS_REG,
    2003: MONTHS_REG,
    2004: MONTHS_REG,
    2005: MONTHS_NO_OCT,
    2006: MONTHS_NO_OCT,
    2007: MONTHS_REG,
    2008: MONTHS_REG,
    2009: MONTHS_REG,
    2010: MONTHS_REG,
    2011: MONTHS_REG,
    2012: MONTHS_2012,
    2013: MONTHS_REG,
    2014: MONTHS_REG,
    2015: MONTHS_REG,
    2016: MONTHS_REG,
    2017: MONTHS_REG,
    2018: MONTHS_REG,
    2019: MONTHS_REG,
    2020: MONTHS_2020,
    2021: MONTHS_2021,
    2022: MONTHS_REG,
    2023: MONTHS_REG,
}


"""
The below section contains dicts for:
    month   :   month code
    team    :   team abbreviation
    team    :   conference
"""
month_dict = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}

team_dict = {
    "Boston Celtics": "BOS",
    "Phoenix Suns": "PHO",
    "Utah Jazz": "UTA",
    "Memphis Grizzlies": "MEM",
    "Golden State Warriors": "GSW",
    "Miami Heat": "MIA",
    "Dallas Mavericks": "DAL",
    "Milwaukee Bucks": "MIL",
    "Philadelphia 76ers": "PHI",
    "Minnesota Timberwolves": "MIN",
    "Denver Nuggets": "DEN",
    "Toronto Raptors": "TOR",
    "Cleveland Cavaliers": "CLE",
    "Atlanta Hawks": "ATL",
    "Brooklyn Nets": "BRK",
    "New Jersey Nets": "NJN",
    "Charlotte Hornets": "CHH",
    "Charlotte Bobcats": "CHA",
    "San Antonio Spurs": "SAS",
    "Los Angeles Clippers": "LAC",
    "New York Knicks": "NYK",
    "Chicago Bulls": "CHI",
    "New Orleans Pelicans": "NOP",
    "New Orleans Hornets": "NOH",
    "Los Angeles Lakers": "LAL",
    "Washington Wizards": "WAS",
    "Indiana Pacers": "IND",
    "Sacramento Kings": "SAC",
    "Detroit Pistons": "DET",
    "Orlando Magic": "ORL",
    "Oklahoma City Thunder": "OKC",
    "Houston Rockets": "HOU",
    "Portland Trail Blazers": "POR",
    "Vancouver Grizzlies": "VAN",
    "Kansas City Kings": "KCK",
    "Washington Bullets": "WSB",
    "San Diego Clippers": "SDC",
    "Seattle SuperSonics": "SEA",
    "New Orleans/Oklahoma City Hornets": "NOK",
}

conf_dict = {
    "Boston Celtics": "East",
    "Phoenix Suns": "West",
    "Utah Jazz": "West",
    "Memphis Grizzlies": "West",
    "Golden State Warriors": "West",
    "Miami Heat": "East",
    "Dallas Mavericks": "West",
    "Milwaukee Bucks": "East",
    "Philadelphia 76ers": "East",
    "Minnesota Timberwolves": "West",
    "Denver Nuggets": "West",
    "Toronto Raptors": "East",
    "Cleveland Cavaliers": "East",
    "Atlanta Hawks": "East",
    "Brooklyn Nets": "East",
    "Charlotte Hornets": "East",
    "Charlotte Bobcats": "East",
    "San Antonio Spurs": "West",
    "LA Clippers": "West",
    "New York Knicks": "East",
    "Chicago Bulls": "East",
    "New Orleans Pelicans": "West",
    "Los Angeles Lakers": "West",
    "Washington Wizards": "East",
    "Indiana Pacers": "East",
    "Sacramento Kings": "West",
    "Detroit Pistons": "East",
    "Orlando Magic": "East",
    "Oklahoma City Thunder": "West",
    "Houston Rockets": "West",
    "Portland Trail Blazers": "West",
}

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

"""
The below section contains parameter grids for testing hyperparameters
"""
xgb_param_grid = {
    "eta": [0.3, 0.4, 0.5, 0.6],
    "max_depth": [1, 2, 3, 5, 7, 9, 13],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15],
    "gamma": [0.0],
    "colsample_bytree": [1.0],
    "n_estimators": [580],
    "subsample": [0.6, 0.8, 1.0],
    "objective": ["multi:softmax"],
}

xgb_narrow_grid = {
    "eta": [0.01],
    "max_depth": [3],
    "min_child_weight": [5, 90, 150],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [1.0],
    "n_estimators": [580],
    "subsample": [0.8],
    "objective": ["multi:softmax"],
}


def timerun(function):
    """
    Simple Function Timing Wrapper
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        time_1 = time.perf_counter()
        result = function(*args, **kwargs)
        time_2 = time.perf_counter()
        total = time_2 - time_1
        print(
            f"Function {function.__name__} Took: {total:.4f} seconds ({total/60:.4f} minutes)"
        )
        return result

    return wrapper

    # data3["Record"] = data3[["W", "L"]].apply(
    #   lambda row: "-".join(row.values.astype(str)), axis=1
    # )


if __name__ == "__main__":
    pass
