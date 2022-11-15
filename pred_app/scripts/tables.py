"""
Script contains table building functions for website
I CAN'T KNOW HOW TO HEAR ANYMORE ABOUT TABLES - Tim Robinson
"""
import pandas as pd
import numpy as np
import typing as t
from datetime import date
from scripts import const
from scripts import scrape


def build_upcoming_schedule() -> pd.DataFrame:
    """
    Updates daily with new upcoming schedule of unplayed games
    """
    upcoming_games = pd.read_sql_table("2023_upcoming_games", const.ENGINE)
    today_mask = upcoming_games["Date"] != str(date.today())
    upcoming_games = upcoming_games.loc[today_mask].reset_index(drop=True)
    upcoming_games.to_sql(
        f"upcoming_schedule_table", const.ENGINE, if_exists="replace", index=False
    )

    return upcoming_games


def build_compare_trad_stats(home_team: str, away_team: str) -> pd.DataFrame:
    """
    Builds table of two given team's current averages
    """
    team_stats = pd.read_sql_table("team_stats", const.ENGINE)
    team_stats = team_stats.groupby("Team")

    h_stats = team_stats.get_group(home_team)
    a_stats = team_stats.get_group(away_team)

    data = pd.concat([h_stats, a_stats], axis=0, join="outer")
    data = data[const.TABLE_STATS_FULL]

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


def build_compare_matchup(
    team_one: str, team_two: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds table of 5 most recent head-to-head games
    Averages each team's stats in those 5 games
    Return games table and averages table
    """

    box = pd.read_sql_table("boxscore_data", const.ENGINE)

    mask = ((box["Away"] == team_one) & (box["Home"] == team_two)) | (
        (box["Away"] == team_two) & (box["Home"] == team_one)
    )

    filtered_box = box.loc[mask].reset_index(drop=True).tail(5)

    t1_final = return_averages(filtered_box, team_one)
    t2_final = return_averages(filtered_box, team_two)
    data_final = pd.concat([t1_final, t2_final]).reset_index(drop=True)

    return filtered_box, round(data_final, 3)


def build_compare_main(team_one: str, team_two: str) -> None:
    """
    Loads odds stats and team stats from database
    Builds table of each teams combined team/odds stats
    Adds difference column highlighting different in stats
    """

    all_stats = pd.read_sql_table("all_stats", const.ENGINE)[
        const.COMPARE_COLS
    ].groupby("Team")

    odds_stats = pd.read_sql_table("odds_stats", const.ENGINE)[const.ODDS_COLS].groupby(
        "Team"
    )

    t1_full = np.concatenate(
        [all_stats.get_group(team_one), odds_stats.get_group(team_one)], axis=1
    )

    t1_full = pd.DataFrame(t1_full, columns=const.COMPARE_COLS + const.ODDS_COLS)
    t1_full = t1_full.loc[:, ~t1_full.columns.duplicated()]

    t2_full = np.concatenate(
        [all_stats.get_group(team_two), odds_stats.get_group(team_two)], axis=1
    )

    t2_full = pd.DataFrame(t2_full, columns=const.COMPARE_COLS + const.ODDS_COLS)
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
    Returns a team's average stats in a given dataset (boxscore)
    """
    a_games = data.groupby("Away")
    h_games = data.groupby("Home")

    a_avgs = (
        a_games.get_group(team_name)[const.AWAY_FEATURES]
        .reset_index(drop=True)
        .astype(float)
    )

    h_avgs = (
        h_games.get_group(team_name)[const.HOME_FEATURES]
        .reset_index(drop=True)
        .astype(float)
    )

    final_avgs = (
        (a_avgs.agg(np.mean).values + h_avgs.agg(np.mean).values) / 2
    ).reshape(1, 14)

    final_avgs = pd.DataFrame(
        list(map(np.ravel, final_avgs)), columns=const.MATCHUP_FEATURES
    )

    final_avgs.insert(loc=0, column="Team", value=team_name)

    return round(final_avgs, 3)


def filter_stats_year(year: str) -> pd.DataFrame:
    """
    Builds and returns table of team averages for a given year
    """

    final_data = []
    season = scrape.map_season(year)
    data = pd.read_sql_table("boxscore_data", const.ENGINE)
    mask = data["SeasonID"] == season
    filtered = data.loc[mask].reset_index(drop=True)

    for team in filtered["Away"].unique():
        temp = return_averages(filtered, team)
        final_data.append(temp)

    final_data = pd.DataFrame(list(map(np.ravel, final_data)), columns=temp.columns)

    return final_data


def historical_team_avgs() -> pd.DataFrame:
    """
    Builds and returns table of team averages over all games in the database
    """

    final_data = []
    data = pd.read_sql_table("boxscore_data", const.ENGINE)

    for team in data["Away"].unique():
        temp = return_averages(data, team)
        final_data.append(temp)

    final_data = pd.DataFrame(list(map(np.ravel, final_data)), columns=temp.columns)

    return final_data

def filter_stats_range(start_year: int = 2021, end_year: int = 2022) -> pd.DataFrame:
    """
    Builds and returns table of team averages for a given year
    """

    raw_data = []
    full_data = []
    data = pd.read_sql_table("boxscore_data", const.ENGINE)

    for year in range(start_year, end_year+1):
        season = scrape.map_season(year)
        mask = (data["SeasonID"] == season)
        temp = data.loc[mask].reset_index(drop=True)
        
        raw_data.append(temp)

    raw_data = pd.concat(raw_data).reset_index(drop=True)

    for team in data["Away"].unique():
        temp = return_averages(data, team)
        full_data.append(temp)

    full_data = pd.DataFrame(list(map(np.ravel, full_data)), columns=temp.columns)

    return full_data


if __name__ == "__main__":
    pass
