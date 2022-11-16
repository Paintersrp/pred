"""
Outside of the collect specific function nothing in this script needs to be reused
All data has already been collected, copies can be found in the /backups folder
"""
from datetime import date
import time
import requests
import sys
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts.transform import set_extras, clean_box_data
from scripts import utils, const, dicts


def scrape_sch(year: str, months: list) -> pd.DataFrame:
    arrays = []

    for month in tqdm(months):
        time.sleep(3)
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

        page = requests.get(url, timeout=60)
        soup = BeautifulSoup(page.text, "html.parser")

        table = soup.find(
            name="table", attrs={"class": "suppress_glossary sortable stats_table"}
        )

        if not table is None:
            body = table.find("tbody")
        else:
            break

        #  Removes some headers that occassionally appear mid-table
        for row in body.find_all("tr", class_="thead"):
            row.decompose()

        rows = body.find_all("tr")

        for table_row in rows:
            row_date = table_row.find("th")
            cells = table_row.find_all("td")
            arr = [row_date.text] + [row.text for row in cells]

            arrays.append(arr)

    return arrays


def map_months(year: int) -> list:
    """
    Returns season's months list for given year
    """
    return dicts.months_map.get(year, const.MONTHS_REG)


def map_season(year: int) -> list:
    """
    Returns season's months list for given year
    """
    return dicts.season_map[year]


@utils.timerun
def collect_sch_by_year(year: int) -> pd.DataFrame:
    """
    Collects schedule data for specific year
    """
    months = map_months(year)
    games = scrape_sch(year, months)

    data = pd.DataFrame(list(map(np.ravel, games)))

    #  Game start times were not included on the schedule table until 2001
    if len(data.columns) == 10:
        data.drop(data.columns[[5, 7, 8, 9]], axis=1, inplace=True)
    else:
        data.drop(data.columns[[6, 8, 9, 10]], axis=1, inplace=True)

    data.columns = ["Date", "Time", "Away", "A-Pts", "Home", "H-Pts", "OT"]
    data["OT"] = data["OT"].str.replace("Unnamed: 7", "")
    data["Date"] = pd.to_datetime(data["Date"])

    played = data[data["H-Pts"] != ""]
    played = set_extras(played)
    played.to_sql(
        f"{year}_played_games", const.ENGINE, if_exists="replace", index=False
    )

    upcoming = data[data["H-Pts"] == ""]
    upcoming.drop(upcoming.columns[[3, 5, 6]], axis=1, inplace=True)
    upcoming.to_sql(
        f"{year}_upcoming_games", const.ENGINE, if_exists="replace", index=False
    )

    return played


def collect_sch_by_range(start_year: int, end_year: int) -> None:
    """
    Collects schedule data for specific range
    """
    first = True

    for year in range(start_year, end_year + 1):
        months = map_months(year)
        games = scrape_sch(year, months)

        if first:
            data = pd.DataFrame(list(map(np.ravel, games)))
            first = False
        else:
            temp = pd.DataFrame(list(map(np.ravel, games)))
            data = pd.concat([data, temp], axis=0, join="outer")

    if len(data.columns) == 10:
        data.drop(data.columns[[5, 7, 8, 9]], axis=1, inplace=True)
    else:
        data.drop(data.columns[[6, 8, 9, 10]], axis=1, inplace=True)

    data.columns = ["Date", "Time", "Away", "A-Pts", "Home", "H-Pts", "OT"]
    data["OT"] = data["OT"].str.replace("Unnamed: 7", "")


@utils.timerun
def get_boxscore_data(
    data: pd.DataFrame,
    from_csv: bool = False,
) -> pd.DataFrame:
    """
    Retrieves boxscore data for every game in a dataset

    otcheck - Sets which tables to index in the HTML based on OT status of the game
    Games with more OTs add more tables, so adjustments are made to retrieve the ones desired
    Table 1:  Away Basic
    Table 2:  Away Advanced
    Table 3:  Home Basic
    Table 4:  Home Advanced
    [Away Basic, Away Advanced, Home Basic, Home Advanced]
    """

    final = []

    for i in tqdm(data.index):
        arrays = []

        if data.at[i, "H-Pts"] > data.at[i, "A-Pts"]:
            data.at[i, "Outcome"] = 1
        else:
            data.at[i, "Outcome"] = 0

        if not from_csv:
            date_split = str(data.at[i, "Date"]).split(" ")
            date_split = date_split[0].split("-")
            month_num = date_split[1]
            year_num = date_split[0]
            day_num = date_split[2].replace(",", "")

        else:
            date_split = data.at[i, "Date"].split(" ")
            month_num = dicts.month_dict[date_split[1]]
            year_num = date_split[3]
            day_num = date_split[2].replace(",", "")

            if 1 <= int(day_num) <= 9:
                day_num = (
                    f"0{day_num}"  #  Adds leading zero to digits under 10. (9 -> 09)
                )

        team_abr = dicts.team_dict[data.at[i, "Home"]]
        ot_check = data.at[i, "OT"]

        url = f"https://www.basketball-reference.com/boxscores/{year_num}{month_num}{day_num}0{team_abr}.html"  #  pylint: disable=line-too-long

        time.sleep(3)
        page = requests.get(url, timeout=60)
        soup = BeautifulSoup(page.text, "html.parser")

        arrays = []

        if ot_check == "OT":
            table_indexes = [0, 8, 9, 17]
        elif ot_check == "2OT":
            table_indexes = [0, 9, 10, 19]
        elif ot_check == "3OT":
            table_indexes = [0, 10, 11, 21]
        elif ot_check == "4OT":
            table_indexes = [0, 11, 12, 23]
        else:
            table_indexes = [0, 7, 8, 15]

        for i in table_indexes:
            table = soup.find_all(
                name="table", attrs={"class": "sortable stats_table"}
            )[i]

            last_row = table("tr")[-1]
            cells = last_row.find_all("td")
            arr = [tr.text for tr in cells]

            arrays.append(arr)

        temp = np.concatenate(arrays, axis=0)
        final.append(temp)

    averages = pd.DataFrame(list(map(np.ravel, final)))

    final = pd.concat([data, averages], axis=1, join="outer")
    final.to_csv("BoxscoreData_2013.csv", index=None)

    return final


@utils.timerun
def collect_training_data(
    data: pd.DataFrame = pd.read_sql_table("full_sch", const.ENGINE)
) -> None:
    """
    Retrieves team averages data for every game in a dataset using nba_api
    """

    full_arrays = []
    check_date = date.today()
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    mask = (
        (data["SeasonID"] == "2018-19")
        | (data["SeasonID"] == "2019-20")
        | (data["SeasonID"] == "2020-21")
        | (data["SeasonID"] == "2021-22")
    )
    filtered = data.loc[mask].reset_index(drop=True)

    for i in tqdm(filtered.index):
        game_date = filtered.at[i, "Date"]
        season_id = filtered.at[i, "SeasonID"]

        if check_date != game_date:
            check_date = game_date

            basic_stats = ldts.LeagueDashTeamStats(
                per_mode_detailed="Per100Possessions",
                season=season_id,
                date_to_nullable=game_date,
            ).league_dash_team_stats.get_data_frame()

            time.sleep(1)

            basic_stats.drop(["TEAM_ID", "CFID", "CFPARAMS"], axis=1, inplace=True)

            basic_stats["TEAM_NAME"] = np.where(
                basic_stats["TEAM_NAME"] == "Vancouver Grizzlies",
                "Memphis Grizzlies",
                basic_stats["TEAM_NAME"],
            )

            basic_stats["TEAM_NAME"] = np.where(
                basic_stats["TEAM_NAME"] == "LA Clippers",
                "Los Angeles Clippers",
                basic_stats["TEAM_NAME"],
            )

            advanced_stats = ldts.LeagueDashTeamStats(
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                season=season_id,
                date_to_nullable=game_date,
            ).league_dash_team_stats.get_data_frame()

            time.sleep(1)

            advanced_stats["TEAM_NAME"] = np.where(
                advanced_stats["TEAM_NAME"] == "Vancouver Grizzlies",
                "Memphis Grizzlies",
                advanced_stats["TEAM_NAME"],
            )

            advanced_stats["TEAM_NAME"] = np.where(
                advanced_stats["TEAM_NAME"] == "LA Clippers",
                "Los Angeles Clippers",
                advanced_stats["TEAM_NAME"],
            )

            advanced_stats.drop(
                [
                    "TEAM_NAME",
                    "TEAM_ID",
                    "CFID",
                    "CFPARAMS",
                    "GP",
                    "W",
                    "L",
                    "W_PCT",
                    "GP_RANK",
                    "W_RANK",
                    "L_RANK",
                    "W_PCT_RANK",
                    "MIN",
                    "MIN_RANK",
                ],
                axis=1,
                inplace=True,
            )

            team_stats = pd.concat([basic_stats, advanced_stats], axis=1, join="outer")
            team_stats = team_stats.groupby("TEAM_NAME")
            home_team = team_stats.get_group(filtered.at[i, "Home"])
            away_team = team_stats.get_group(filtered.at[i, "Away"])

            temp = np.concatenate((away_team, home_team), axis=0)
            full_arrays.append(temp)

        else:
            home_team = team_stats.get_group(filtered.at[i, "Home"])
            away_team = team_stats.get_group(filtered.at[i, "Away"])

            temp = np.concatenate((away_team, home_team), axis=0)
            full_arrays.append(temp)

    features = ["A_" + col for col in away_team.columns] + [
        "H_" + col for col in home_team.columns
    ]

    averages = pd.DataFrame(list(map(np.ravel, full_arrays)), columns=features)

    final = pd.concat([filtered, averages], axis=1, join="outer")
    final.to_csv("TrainDataRawAdv.csv", index=None)


if __name__ == "__main__":
    # collect_range(2005, 2022)
    # final_boxscore_data = get_boxscore_data()
    # set_extras()
    # final_team_data()

    # test = collect_specific(2023)
    # print(test)
    pass