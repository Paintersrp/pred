import time
import functools
from sqlalchemy import create_engine

engine = create_engine("sqlite:///pred.db")

months_reg = [
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

months_no_oct = [
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]

months_1999 = ["february", "march", "april", "may", "june"]
months_2012 = ["december", "january", "february", "march", "april", "may", "june"]

months_2020 = [
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

months_2021 = [
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
]

xgb_param_grid = {
    "eta": [0.3, 0.4, 0.5, 0.6],
    "max_depth": [1, 2, 3, 5, 7, 9, 13],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15],
    "gamma": [0.0],
    "colsample_bytree": [1.0],
    "n_estimators": [50, 300, 500, 1000],
    "subsample": [0.6, 0.8, 1.0],
    "objective": ["multi:softmax"],
}

xgb_narrow_grid = {
    "eta": [0.1],
    "max_depth": [4, 5, 6],
    "min_child_weight": [5, 15, 60],
    "gamma": [0.0, 0.5],
    "colsample_bytree": [0.8, 1.0],
    "n_estimators": [385],
    "subsample": [0.8, 1.0],
    "objective": ["multi:softmax"],
}

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

def timerun(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = function(*args, **kwargs)
        t2 = time.perf_counter()
        total = t2 - t1
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
