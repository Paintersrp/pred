from scripts import const

"""
Current model hyperparameters
"""


PARAMS = {
    "max_depth": 3,
    "min_child_weight": 5,
    "eta": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "num_class": 2,
}

TREE_TESTING = {
    "objective": "multi:softmax",
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "nthread": 4,
    "num_class": 2,
    "seed": 27,
}


""" Maps given year to SeasonID code """


season_map = {
    1976: "1975-76",
    1977: "1976-77",
    1978: "1977-78",
    1979: "1978-79",
    1980: "1979-80",
    1981: "1980-81",
    1982: "1981-82",
    1983: "1982-83",
    1984: "1983-84",
    1985: "1984-85",
    1986: "1985-86",
    1987: "1986-87",
    1988: "1987-88",
    1989: "1988-89",
    1990: "1989-90",
    1991: "1990-91",
    1992: "1991-92",
    1993: "1992-93",
    1994: "1993-94",
    1995: "1994-95",
    1996: "1995-96",
    1997: "1996-97",
    1998: "1997-98",
    1999: "1998-99",
    2000: "1999-00",
    2001: "2000-01",
    2002: "2001-02",
    2003: "2002-03",
    2004: "2003-04",
    2005: "2004-05",
    2006: "2005-06",
    2007: "2006-07",
    2008: "2007-08",
    2009: "2008-09",
    2010: "2009-10",
    2011: "2010-11",
    2012: "2011-12",
    2013: "2012-13",
    2014: "2013-14",
    2015: "2014-15",
    2016: "2015-16",
    2017: "2016-17",
    2018: "2017-18",
    2019: "2018-19",
    2020: "2019-20",
    2021: "2020-21",
    2022: "2021-22",
    2023: "2022-23",
}


""" Maps year to given played months in a season """


months_map = {
    1976: const.MONTHS_REG,
    1977: const.MONTHS_REG,
    1978: const.MONTHS_REG,
    1979: const.MONTHS_REG,
    1980: const.MONTHS_REG,
    1981: const.MONTHS_REG,
    1982: const.MONTHS_REG,
    1983: const.MONTHS_REG,
    1984: const.MONTHS_REG,
    1985: const.MONTHS_REG,
    1986: const.MONTHS_REG,
    1987: const.MONTHS_REG,
    1988: const.MONTHS_NO_OCT,
    1989: const.MONTHS_NO_OCT,
    1990: const.MONTHS_NO_OCT,
    1991: const.MONTHS_NO_OCT,
    1992: const.MONTHS_NO_OCT,
    1993: const.MONTHS_NO_OCT,
    1994: const.MONTHS_NO_OCT,
    1995: const.MONTHS_NO_OCT,
    1996: const.MONTHS_NO_OCT,
    1997: const.MONTHS_NO_OCT,
    1998: const.MONTHS_REG,
    1999: const.MONTHS_1999,
    2000: const.MONTHS_NO_OCT,
    2001: const.MONTHS_REG,
    2002: const.MONTHS_REG,
    2003: const.MONTHS_REG,
    2004: const.MONTHS_REG,
    2005: const.MONTHS_NO_OCT,
    2006: const.MONTHS_NO_OCT,
    2007: const.MONTHS_REG,
    2008: const.MONTHS_REG,
    2009: const.MONTHS_REG,
    2010: const.MONTHS_REG,
    2011: const.MONTHS_REG,
    2012: const.MONTHS_2012,
    2013: const.MONTHS_REG,
    2014: const.MONTHS_REG,
    2015: const.MONTHS_REG,
    2016: const.MONTHS_REG,
    2017: const.MONTHS_REG,
    2018: const.MONTHS_REG,
    2019: const.MONTHS_REG,
    2020: const.MONTHS_2020,
    2021: const.MONTHS_2021,
    2022: const.MONTHS_REG,
    2023: const.MONTHS_REG,
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
    "Charlotte Hornets": "CHO",
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

odds_team_dict = {
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "GoldenState": "Golden State Warriors",
    "Golden State": "Golden State Warriors",  # intentional duplicate
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "LAClippers": "Los Angeles Clippers",
    "LA Clippers": "Los Angeles Clippers",  # intentional duplicate
    "LALakers": "Los Angeles Lakers",
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "NewJersey": "Brooklyn Nets",
    "NewOrleans": "New Orleans Pelicans",
    "NewYork": "New York Knicks",
    "OklahomaCity": "Oklahoma City Thunder",
    "Oklahoma City": "Oklahoma City Thunder",  # intentional duplicate
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "SanAntonio": "San Antonio Spurs",
    "Seattle": "Oklahoma City Thunder",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
}


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
