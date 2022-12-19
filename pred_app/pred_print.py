from scripts.handler import MetricsHandler, GeneralHandler
from scripts.updater import Updater
from scripts.predictor import DailyPredictor
from scripts.ratings import current_elos
import scripts.const as const
import pickle
import dill
from dateutil import parser
from datetime import date, datetime, timedelta
from collections import defaultdict
from statistics import mode, mean
from scripts.utils import timerun
import requests as req
import numpy as np
import pandas as pd
import json
import arrow


lines_team_dict = {
    "ATL Hawks": "Atlanta Hawks",
    "BOS Celtics": "Boston Celtics",
    "BKN Nets": "Brooklyn Nets",
    "CHA Hornets": "Charlotte Hornets",
    "CHI Bulls": "Chicago Bulls",
    "CLE Cavaliers": "Cleveland Cavaliers",
    "DAL Mavericks": "Dallas Mavericks",
    "DEN Nuggets": "Denver Nuggets",
    "DET Pistons": "Detroit Pistons",
    "GS Warriors": "Golden State Warriors",
    "HOU Rockets": "Houston Rockets",
    "IND Pacers": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "MEM Grizzlies": "Memphis Grizzlies",
    "MIA Heat": "Miami Heat",
    "MIL Bucks": "Milwaukee Bucks",
    "MIN Timberwolves": "Minnesota Timberwolves",
    "NO Pelicans": "New Orleans Pelicans",
    "NY Knicks": "New York Knicks",
    "OKC Thunder": "Oklahoma City Thunder",
    "ORL Magic": "Orlando Magic",
    "PHI 76ers": "Philadelphia 76ers",
    "PHX Suns": "Phoenix Suns",
    "POR Trail Blazers": "Portland Trail Blazers",
    "SAC Kings": "Sacramento Kings",
    "SA Spurs": "San Antonio Spurs",
    "TOR Raptors": "Toronto Raptors",
    "UTA Jazz": "Utah Jazz",
    "WAS Wizards": "Washington Wizards",
}

team_dict = {
    "Boston Celtics": "BOS",
    "Phoenix Suns": "PHX",
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
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "San Antonio Spurs": "SAS",
    "LA Clippers": "LAC",
    "New York Knicks": "NYK",
    "Chicago Bulls": "CHI",
    "New Orleans Pelicans": "NOP",
    "Los Angeles Lakers": "LAL",
    "Washington Wizards": "WAS",
    "Indiana Pacers": "IND",
    "Sacramento Kings": "SAC",
    "Detroit Pistons": "DET",
    "Orlando Magic": "ORL",
    "Oklahoma City Thunder": "OKC",
    "Houston Rockets": "HOU",
    "Portland Trail Blazers": "POR",
}

DAILY_COLS = [
    "Location",
    "Time",
    "A_Team",
    "H_Team",
    "A_Record",
    "H_Record",
    "A_ELO",
    "H_ELO",
    "A_NET",
    "H_NET",
    "A_Pred",
    "H_Pred",
    "A_Pred_ML",
    "H_Pred_ML",
    "A_Spread",
    "H_Spread",
    "A_S_Odds",
    "H_S_Odds",
    "A_ML_Odds",
    "H_ML_Odds",
    "A_Implied",
    "H_Implied",
    "Hold",
    "A_PG",
    "A_SG",
    "A_SF",
    "A_PF",
    "A_C",
    "H_PG",
    "H_SG",
    "H_SF",
    "H_PF",
    "H_C",
]

TOMORROW_COLS = [
    "Location",
    "Time",
    "A_Team",
    "H_Team",
    "A_Record",
    "H_Record",
    "A_ELO",
    "H_ELO",
    "A_NET",
    "H_NET",
    "A_Pred",
    "H_Pred",
    "A_Pred_ML",
    "H_Pred_ML",
]


def get_key(val):
    for key, value in lines_team_dict.items():
        if str(val) == value:
            return key

    return "key doesn't exist"


def update_today_card_data(
    preds: pd.DataFrame,
    team_stats: pd.DataFrame,
    lineups: dict,
    elos: dict,
) -> None:
    """Func"""

    parse_json_to_dict()
    today, _tmw = load_dicts()

    preds["A_Odds"] = preds["A_Odds"].astype(float)
    preds["H_Odds"] = preds["H_Odds"].astype(float)

    daily_lines = pd.read_csv("today_update.csv")
    daily_lines["commence_time"] = pd.to_datetime(daily_lines["commence_time"])

    daily_lines["away_team"] = np.where(
        daily_lines["away_team"] == "Los Angeles Clippers",
        "LA Clippers",
        daily_lines["away_team"],
    )

    daily_lines["home_team"] = np.where(
        daily_lines["home_team"] == "Los Angeles Clippers",
        "LA Clippers",
        daily_lines["home_team"],
    )

    full_arr = []

    print(preds)

    for row in daily_lines.itertuples():
        arr = []
        a_lineup = lineups[team_dict.get(row.away_team)]
        h_lineup = lineups[team_dict.get(row.home_team)]
        odds_row = preds.loc[preds["A_Team"] == row.away_team].reset_index(drop=True)
        start_time = row.commence_time.strftime("%I:%M %p").lstrip("0")

        if len(row.home_team.split(" ")) == 3:
            location = row.home_team.split(" ")[0] + " " + row.home_team.split(" ")[1]
        else:
            location = row.home_team.split(" ")[0]

        a_stats = team_stats.loc[team_stats["Team"] == row.away_team].reset_index(
            drop=True
        )
        h_stats = team_stats.loc[team_stats["Team"] == row.home_team].reset_index(
            drop=True
        )

        if odds_row["A_Odds"].values < 0.5:
            a_american = ((1 / odds_row["A_Odds"].values) - 1) * 100
        else:
            a_american = -100 / ((1 / odds_row["A_Odds"].values) - 1)

        if mean(today[row.away_team]["ml"]) > 0:
            away_implied = 100 / (mean(today[row.away_team]["ml"]) + 100) * 100
        else:
            away_implied = (
                abs(mean(today[row.away_team]["ml"]))
                / (abs(mean(today[row.away_team]["ml"])) + 100)
                * 100
            )

        if mean(today[row.home_team]["ml"]) > 0:
            home_implied = 100 / (mean(today[row.home_team]["ml"]) + 100) * 100
        else:
            home_implied = (
                abs(mean(today[row.home_team]["ml"]))
                / (abs(mean(today[row.home_team]["ml"])) + 100)
                * 100
            )

        arr.extend(
            [
                location,
                start_time,
                row.away_team,
                row.home_team,
                str(a_stats.at[0, "W"]) + "-" + str(a_stats.at[0, "L"]),
                str(h_stats.at[0, "W"]) + "-" + str(h_stats.at[0, "L"]),
                elos[row.away_team],
                elos[row.home_team],
                a_stats.at[0, "NET_RATING"],
                h_stats.at[0, "NET_RATING"],
                float(odds_row["A_Odds"]) * 100,
                float(odds_row["H_Odds"]) * 100,
                round(float(a_american), 0),
                -round(float(a_american), 0),
                round(mode(today[row.away_team]["spread"]), 0),
                round(mode(today[row.home_team]["spread"]), 0),
                round(mean(today[row.away_team]["spread_odds"]), 0),
                round(mean(today[row.home_team]["spread_odds"]), 0),
                round(mean(today[row.away_team]["ml"]), 0),
                round(mean(today[row.home_team]["ml"]), 0),
                away_implied,
                home_implied,
                (away_implied + home_implied) - 100,
            ]
        )
        arr = arr + a_lineup + h_lineup
        # print(len(arr))
        full_arr.append(arr)

    final_data = pd.DataFrame(full_arr, columns=DAILY_COLS)
    final_data = round(final_data, 2)
    final_data.reset_index(inplace=True)

    for i in final_data.index:
        final_data.at[i, "H_Team"] = get_key(final_data.at[i, "H_Team"])
        final_data.at[i, "A_Team"] = get_key(final_data.at[i, "A_Team"])

    print(final_data)

    final_data.to_json(
        "C:/Python/pred_app/frontend/public/data/cards.json", orient="records"
    )
    final_data.to_sql("daily_card_data", const.ENGINE, if_exists="replace", index=False)


def update_tomorrow_card_data(
    preds: pd.DataFrame,
    team_stats: pd.DataFrame,
    elos: dict,
) -> None:
    """Func"""

    data = pd.read_sql_table(f"2023_upcoming_games", const.ENGINE)
    data["Date"] = pd.to_datetime(data["Date"]).dt.date

    data = data.loc[(data["Date"] == date.today() + timedelta(days=1))].reset_index(
        drop=True
    )

    data["Home"] = np.where(
        data["Home"] == "Los Angeles Clippers",
        "LA Clippers",
        data["Home"],
    )
    data["Away"] = np.where(
        data["Away"] == "Los Angeles Clippers",
        "LA Clippers",
        data["Away"],
    )

    preds["A_Odds"] = preds["A_Odds"].astype(float)
    preds["H_Odds"] = preds["H_Odds"].astype(float)
    full_arr = []

    for row in data.itertuples():
        arr = []
        odds_row = preds.loc[preds["A_Team"] == row.Away].reset_index(drop=True)
        print(odds_row)
        start_time = row.Time

        if len(row.Home.split(" ")) == 3:
            location = row.Home.split(" ")[0] + " " + row.Home.split(" ")[1]
        else:
            location = row.Home.split(" ")[0]

        a_stats = team_stats.loc[team_stats["Team"] == row.Away].reset_index(drop=True)
        h_stats = team_stats.loc[team_stats["Team"] == row.Home].reset_index(drop=True)

        if odds_row["A_Odds"].values < 0.5:
            a_american = ((1 / odds_row["A_Odds"].values) - 1) * 100
        else:
            a_american = -100 / ((1 / odds_row["A_Odds"].values) - 1)

        arr.extend(
            [
                location,
                start_time,
                row.Away,
                row.Home,
                str(a_stats.at[0, "W"]) + "-" + str(a_stats.at[0, "L"]),
                str(h_stats.at[0, "W"]) + "-" + str(h_stats.at[0, "L"]),
                elos[row.Away],
                elos[row.Home],
                a_stats.at[0, "NET_RATING"],
                h_stats.at[0, "NET_RATING"],
                float(odds_row["A_Odds"]) * 100,
                float(odds_row["H_Odds"]) * 100,
                round(float(a_american), 0),
                -round(float(a_american), 0),
            ]
        )

        full_arr.append(arr)

    final_data = pd.DataFrame(full_arr, columns=TOMORROW_COLS)
    final_data = round(final_data, 2)
    final_data.reset_index(inplace=True)

    for i in final_data.index:
        final_data.at[i, "H_Team"] = get_key(final_data.at[i, "H_Team"])
        final_data.at[i, "A_Team"] = get_key(final_data.at[i, "A_Team"])

    print(final_data)

    final_data.to_json(
        "C:/Python/pred_app/frontend/public/data/tmw_cards.json", orient="records"
    )
    final_data.to_sql(
        "tomorrow_card_data", const.ENGINE, if_exists="replace", index=False
    )


def update_odds() -> None:
    API_KEY = "37c1e069a8acd8be4286c84ef93fea3d"

    odds_response = req.get(
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={
            "api_key": API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )

    if odds_response.status_code != 200:
        print(
            f"Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}"
        )

    else:
        odds_json = odds_response.json()

        with open("sample.json", "w") as outfile:
            json.dump(odds_json, outfile)

        print("Remaining requests", odds_response.headers["x-requests-remaining"])
        print("Used requests", odds_response.headers["x-requests-used"])


@timerun
def parse_json_to_dict():
    with open("sample.json", "r") as openfile:
        json_object = json.load(openfile)

    data = pd.DataFrame(json_object)
    data.drop(["id", "sport_key", "sport_title"], axis=1, inplace=True)

    for i in data.index:
        data.at[i, "commence_time"] = parser.parse(
            data.at[i, "commence_time"]
        ) - timedelta(hours=5)

    data["date_time"] = pd.to_datetime(data["commence_time"]).dt.date

    today_update = data.loc[data["date_time"] == date.today()]
    tomorrow_games = data.loc[data["date_time"] == (date.today() + timedelta(days=1))]

    if not today_update.empty:
        today_team_dict = defaultdict(lambda: defaultdict(list))

        for row in today_update.itertuples():
            # for item in row["bookmakers"]:
            for item in row.bookmakers:
                for bookie_lines in item["markets"]:
                    for line in bookie_lines["outcomes"]:
                        if line["name"] == "Los Angeles Clippers":
                            line["name"] = "LA Clippers"

                        if not line["name"] == "Over" and not line["name"] == "Under":
                            if bookie_lines["key"] == "spreads":
                                today_team_dict[line["name"]]["spread"].append(
                                    line["point"]
                                )
                                today_team_dict[line["name"]]["spread_odds"].append(
                                    line["price"]
                                )
                            else:
                                today_team_dict[line["name"]]["ml"].append(
                                    line["price"]
                                )

        with open("today_team_dict.pkl", "wb") as f:
            dill.dump(today_team_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        today_update.drop("bookmakers", axis=1, inplace=True)
        today_update.to_csv("today_update.csv", index=None)

    else:
        print("\n", "No games for today parsed.")

    if not tomorrow_games.empty:
        tomorrow_team_dict = defaultdict(lambda: defaultdict(list))

        for row in tomorrow_games.itertuples():
            for item in row.bookmakers:
                for bookie_lines in item["markets"]:
                    for line in bookie_lines["outcomes"]:
                        if line["name"] == "Los Angeles Clippers":
                            line["name"] = "LA Clippers"

                        if not line["name"] == "Over" and not line["name"] == "Under":
                            if bookie_lines["key"] == "spreads":
                                tomorrow_team_dict[line["name"]]["spread"].append(
                                    line["point"]
                                )
                                tomorrow_team_dict[line["name"]]["spread_odds"].append(
                                    line["price"]
                                )
                            else:
                                tomorrow_team_dict[line["name"]]["ml"].append(
                                    line["price"]
                                )

        with open("tomorrow_team_dict.pkl", "wb") as f:
            dill.dump(tomorrow_team_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        tomorrow_games.drop("bookmakers", axis=1, inplace=True)
        tomorrow_games.to_csv("tomorrow_games.csv", index=None)

    else:
        print("\n", "No games for tomorrow parsed.")


def load_dicts() -> tuple[defaultdict, defaultdict]:
    try:
        today_team_dict = dill.load(open("today_team_dict.pkl", "rb"))

    except FileNotFoundError:
        today_team_dict = None
        print("\n", "No games for today parsed.")

    try:
        tomorrow_team_dict = dill.load(open("tomorrow_team_dict.pkl", "rb"))
    except FileNotFoundError:
        tomorrow_team_dict = None
        print("\n", "No games for tomorrow parsed.")

    return today_team_dict, tomorrow_team_dict


if __name__ == "__main__":
    # daily_lines = pd.read_csv("today_update.csv")
    # print(daily_lines)

    # daily_lines["commence_time"] = pd.to_datetime(daily_lines["commence_time"])
    # print(daily_lines)

    # for row in daily_lines.itertuples():
    #     yarbeans = row.commence_time.strftime("%I:%M %p")
    #     print(yarbeans.lstrip("0"))

    # d = arrow.get(da)
    # print(d.humanize())

    # for row in daily_lines.itertuples():
    #     print(len(row.home_team.split(" ")))
    #     if len(row.home_team.split(" ")) == 3:
    #         test = row.home_team.split(" ")[0] + " " + row.home_team.split(" ")[1]

    #     else:
    #         test = row.home_team.split(" ")[0]

    #     print(test)

    # test = pd.read_sql_table("prediction_scoring", const.ENGINE)
    # print(test)
    # test.to_json(
    #     "C:/Python/pred_app/frontend/public/data/pred_history.json",
    #     orient="records",
    # )

    # update_card_data()
    # parse_json_to_dict()
    # test1, test2 = load_dicts()
    # print(test2)

    # lines = pd.read_csv("today_update.csv")
    # print(lines)

    # test = pd.read_sql_table("prediction_scoring_v2", const.ENGINE)
    # print(test)
    # test["Date"] = pd.to_datetime(test["Date"]).dt.date
    # today = date.today()
    # data = test.loc[test["Date"] < today].reset_index(drop=True)
    # data.to_sql("prediction_history_v2", const.ENGINE, if_exists="replace", index=False)

    # update_odds()
    Updater = Updater()
    # DailyPredictor = DailyPredictor()

    # massey = Updater.update_massey()
    # elos = current_elos()
    # daily_team_stats, grouped_team_stats = Updater.update_team_stats(per_100=True)
    # DailyPredictor.build_test_data(grouped_team_stats, massey, elos)

    # DailyPredictor.prepare_test_data()
    # today_preds, tmw_preds = DailyPredictor.predict()
    # update_tomorrow_card_data(tmw_preds, daily_team_stats, elos)

    # daily_lineups = Updater.update_injuries()
    # update_today_card_data(today_preds, daily_team_stats, daily_lineups, elos)

    Updater.update_history_outcomes()
