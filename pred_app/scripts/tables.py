"""
Script contains table building functions for website
I CAN'T KNOW HOW TO HEAR ANYMORE ABOUT TABLES - Tim Robinson
"""
import pandas as pd
import numpy as np
import requests
import typing as t
from collections import defaultdict
from bs4 import BeautifulSoup
from datetime import date
from scripts import utils
from scripts import scrape
from scripts import transform


def build_upcoming_schedule() -> pd.DataFrame:
    """
    Purpose
    ----------
    Updates daily with a new upcoming schedule of unplayed games

    maybe just branch the main schedule scrapper, sending unplayed here and played to daily
    """
    upcoming_games = pd.read_sql_table("2023_upcoming_games", utils.ENGINE)
    today_mask = upcoming_games["Date"] != str(date.today())
    upcoming_games = upcoming_games.loc[today_mask].reset_index(drop=True)
    upcoming_games.to_sql(
        f"upcoming_schedule_table", utils.ENGINE, if_exists="replace", index=False
    )

    return upcoming_games

def build_prediction_scoring() -> pd.DataFrame:
    """
    Loads most recent prediction history and played game scores
    Adds actual scores and outcome to prediction history
    Commits prediction scoring table to database
    """
    arr = []
    predicted = pd.read_sql_table("prediction_history_net", utils.ENGINE)
    predicted["Actual"] = "TBD"

    played = pd.read_sql_table("2023_played_games", utils.ENGINE)

    pred_mask = predicted["Date"] != str(date.today())
    predicted = predicted.loc[pred_mask].reset_index(drop=True)

    for d in predicted["Date"].unique():
        mov_dict = {}
        played_mask = played["Date"] == d
        filtered_played = played.loc[played_mask].reset_index(drop=True)

        history_mask = predicted["Date"] == d
        filtered_predicted = predicted.loc[history_mask].reset_index(drop=True)

        for i in filtered_played.index:
            mov_dict[filtered_played.at[i, "Away"]] = filtered_played.at[i, "MOV"]

        for i in filtered_predicted.index:
            filtered_predicted.at[i, "Actual"] = mov_dict[
                filtered_predicted.at[i, "A_Team"]
            ]

            if float(filtered_predicted.at[i, "A_Odds"]) > 0.5:
                if filtered_predicted.at[i, "Actual"] > 0:
                    filtered_predicted.at[i, "Outcome"] = 0
                else:
                    filtered_predicted.at[i, "Outcome"] = 1
            else:
                if filtered_predicted.at[i, "Actual"] > 0:
                    filtered_predicted.at[i, "Outcome"] = 1
                else:
                    filtered_predicted.at[i, "Outcome"] = 0

        arr.append(filtered_predicted)

    arr = pd.concat(arr, axis=0, join="outer")
    # arr.to_sql(f"prediction_scoring", utils.ENGINE, if_exists="replace", index=False)
    print(arr)

    return arr

def build_compare_trad_stats(
    home_team: str = "Atlanta Hawks", away_team: str = "Boston Celtics"
) -> pd.DataFrame:
    """
    a
    """
    team_stats = pd.read_sql_table("team_stats", utils.ENGINE)
    team_stats = team_stats.groupby("Team")

    h_stats = team_stats.get_group(home_team)
    a_stats = team_stats.get_group(away_team)

    data = pd.concat([h_stats, a_stats], axis=0, join="outer")
    data = data[utils.TABLE_STATS_FULL]

    data = data.rename(
        columns={
            "TS_PCT": "TS%",
            "OFF_RATING": "OFF",
            "DEF_RATING": "DEF",
            "NET_RATING": "NET",
            "OREB": "ORB",
            "DREB": "DRB",
            "FG_PCT": "FG%",
            "FG3_PCT": "FG3%",
            "FT_PCT": "FT%",
            "EFG_PCT": "EFG%",
            "AST_PCT": "AST%",
        }
    )

    return data

def build_compare_games(
    team_one: str = "Boston Celtics", team_two: str = "Atlanta Hawks"
) -> pd.DataFrame:
    """
    Funcstring
    """

    box = pd.read_sql_table("boxscore_data", utils.ENGINE)

    mask = ((box["Away"] == team_one) & (box["Home"] == team_two)) | (
        (box["Away"] == team_two) & (box["Home"] == team_one)
    )

    filtered_box = box.loc[mask].reset_index(drop=True).tail(5)

    t1_final = return_averages(filtered_box, team_one)
    t2_final = return_averages(filtered_box, team_two)
    data_final = pd.concat([t1_final, t2_final]).reset_index(drop=True)

    return round(data_final, 3)


def build_compare_main(
    team_one: str = "Boston Celtics", team_two: str = "Atlanta Hawks"
) -> None:
    """
    Funcstring
    """

    all_stats = pd.read_sql_table("all_stats", utils.ENGINE)[
        utils.COMPARE_COLS
    ].groupby("Team")

    odds_stats = pd.read_sql_table("odds_stats", utils.ENGINE)[utils.ODDS_COLS].groupby(
        "Team"
    )

    t1_full = np.concatenate(
        [all_stats.get_group(team_one), odds_stats.get_group(team_one)], axis=1
    )

    t1_full = pd.DataFrame(t1_full, columns=utils.COMPARE_COLS + utils.ODDS_COLS)
    t1_full = t1_full.loc[:, ~t1_full.columns.duplicated()]

    t2_full = np.concatenate(
        [all_stats.get_group(team_two), odds_stats.get_group(team_two)], axis=1
    )

    t2_full = pd.DataFrame(t2_full, columns=utils.COMPARE_COLS + utils.ODDS_COLS)
    t2_full = t2_full.loc[:, ~t2_full.columns.duplicated()]
    data = pd.concat([t1_full, t2_full], axis=0, join="outer").reset_index(drop=True)

    extra_stats = data[["Team", "Record"]]
    data.drop(["Team", "Record"], axis=1, inplace=True)
    diff_data = data.diff().dropna()

    temp = pd.concat([data, diff_data], axis=0, join="outer").reset_index(drop=True)
    final_data = pd.concat([extra_stats, temp], axis=1, join="outer").reset_index(
        drop=True
    )
    final_data.fillna("-", inplace=True)


def return_averages(data: pd.DataFrame, team_name: str) -> t.Any:
    """
    Funcstring data: pd.DataFrame, team_name: str
    """
    a_games = data.groupby("Away")
    h_games = data.groupby("Home")

    a_avgs = (
        a_games.get_group(team_name)[utils.AWAY_FEATURES]
        .reset_index(drop=True)
        .astype(float)
    )

    h_avgs = (
        h_games.get_group(team_name)[utils.HOME_FEATURES]
        .reset_index(drop=True)
        .astype(float)
    )

    final_avgs = (
        (a_avgs.agg(np.mean).values + h_avgs.agg(np.mean).values) / 2
    ).reshape(1, 14)

    final_avgs = pd.DataFrame(
        list(map(np.ravel, final_avgs)), columns=utils.MATCHUP_FEATURES
    )

    final_avgs.insert(loc=0, column="Team", value=team_name)

    return round(final_avgs, 3)


def collect_injuries() -> defaultdict(list):
    """
    Funcstring
    """
    url = "https://www.rotowire.com/basketball/nba-lineups.php"

    page = requests.get(url, timeout=60)
    soup = BeautifulSoup(page.text, "html.parser")

    team_names = soup.find_all(name="div", attrs={"class": "lineup__abbr"})

    unordered_lists = soup.find_all(name="ul", attrs={"class": "lineup__list"})

    injury_dict = defaultdict(list)

    for i in range(len(unordered_lists)):
        rows = unordered_lists[i].find_all("li")

        for table_row in rows:
            player_name = table_row.find("a")
            status = table_row.find("span")

            if player_name != None and status != None:
                injury_dict[team_names[i].text].append(
                    (player_name.text + "-" + status.text.replace("GTD", "TBD"))
                )

    for item in injury_dict.items():
        print(item)

    print(injury_dict.keys())

    return injury_dict


def filter_stats_year(year: str) -> pd.DataFrame:
    """
    Funcstring
    """

    final_data = []

    data = pd.read_sql_table("boxscore_data", utils.ENGINE)
    mask = data["SeasonID"] == year

    filtered = data.loc[mask].reset_index(drop=True)

    for team in filtered["Away"].unique():
        temp = return_averages(filtered, team)
        final_data.append(temp)

    final_data = pd.DataFrame(list(map(np.ravel, final_data)), columns=temp.columns)

    return final_data

def historical_team_avgs() -> pd.DataFrame:
    """
    Funcstring
    """

    final_data = []

    data = pd.read_sql_table("boxscore_data", utils.ENGINE)

    for team in data["Away"].unique():
        temp = return_averages(data, team)
        final_data.append(temp)

    final_data = pd.DataFrame(list(map(np.ravel, final_data)), columns=temp.columns)

    return final_data

if __name__ == "__main__":
    pass
