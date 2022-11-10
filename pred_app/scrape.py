from nba_api.stats.endpoints import leaguedashteamstats as ldts
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np
from tqdm import tqdm
from transform import set_extras
import requests
import utils
import time


@utils.timerun
def collect_specific(year, months):
    arrays = []

    for month in tqdm(months):
        #  Sleep function keeps the basketball references overlords from hating you (some)
        time.sleep(1)
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")

        #  Finds the schedule table on the page, grabs the body of that table
        table = soup.find(
            name="table", attrs={"class": "suppress_glossary sortable stats_table"}
        )

        if table == None:
            break
        else:
            body = table.find("tbody")

        #  Removes some headers that occassionally appear mid-table
        for row in body.find_all("tr", class_="thead"):
            row.decompose()

        #  Finds all the rows in the table body, then iterates over each cell in the row appending to an array
        rows = body.find_all("tr")

        for tr in rows:
            date = tr.find("th")
            cells = tr.find_all("td")
            arr = [date.text] + [tr.text for tr in cells]

            arrays.append(arr)

    #  Sets final array so that each game is on an individual row in the DataFrame
    df = pd.DataFrame(list(map(np.ravel, arrays)))

    #  Game start times were not included on the schedule table until 2001, so one more column is dropped when needed.
    if len(df.columns) == 10:
        df.drop(df.columns[[5, 7, 8, 9]], axis=1, inplace=True)
    else:
        df.drop(df.columns[[6, 8, 9, 10]], axis=1, inplace=True)

    #  Sets some column names and replaces some out of place text in the OT Column
    df.columns = ["Date", "Time", "Away", "A-Pts", "Home", "H-Pts", "OT"]
    df = df[df["H-Pts"] != ""]
    df["OT"] = df["OT"].str.replace("Unnamed: 7", "")
    df["Date"] = pd.to_datetime(df["Date"])
    df = set_extras(df)
    df.to_sql(f"{year}_Game_Schedule", utils.engine, if_exists="replace", index=False)

    return df


def collect_range(start_year, end_year):
    arrays = []

    #  Anything further in years than 1976 will need tweaking. Some years they end in April, others in May. Some start in December
    #  For each year in the range, grab the schedule table from each month and append/concat into one array then build/save DataFrame
    for year in range(start_year, end_year + 1):
        if 1988 <= year <= 1997 or 2005 <= year <= 2006 or year == 2000:

            months = utils.months_no_oct

        elif year == 1999:
            months = utils.months_1999

        elif year == 2012:
            months = utils.months_2012

        elif year == 2020:
            months = utils.months_2020

        elif year == 2021:
            months = utils.months_2021

        else:
            months = utils.months_reg

        for month in months:

            #  Sleep function keeps the basketball references overlords from hating you (some)
            time.sleep(1)
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")
            table = soup.find(
                name="table", attrs={"class": "suppress_glossary sortable stats_table"}
            )

            body = table.find("tbody")

            #  Removes some headers that occassionally appear mid-table
            for row in body.find_all("tr", class_="thead"):
                row.decompose()

            rows = body.find_all("tr")

            for tr in rows:
                date = tr.find("th")
                cells = tr.find_all("td")
                arr = [date.text] + [tr.text for tr in cells]

                arrays.append(arr)

    df = pd.DataFrame(list(map(np.ravel, arrays)))

    #  Game start times were not included on the schedule table until 2001, so one more column is dropped when needed.
    if len(df.columns) == 10:
        df.drop(df.columns[[5, 7, 8, 9]], axis=1, inplace=True)
    else:
        df.drop(df.columns[[6, 8, 9, 10]], axis=1, inplace=True)

    df.columns = ["Date", "Time", "Away", "A-Pts", "Home", "H-Pts", "OT"]
    df["OT"] = df["OT"].str.replace("Unnamed: 7", "")
    df.to_csv("FullGamesFinal.csv", index=None)


@utils.timerun
def get_boxscore_data(data=pd.read_csv("FullGamesFinal.csv")):
    """
    Retrieves boxscore data for every game in a dataset
    """

    final = []

    for i in tqdm(data.index):
        arrays = []

        if data.at[i, "H-Pts"] > data.at[i, "A-Pts"]:
            data.at[i, "Outcome"] = 1
        else:
            data.at[i, "Outcome"] = 0

        date_split = data.at[i, "Date"].split(" ")
        month_num = utils.month_dict[date_split[1]]
        year_num = date_split[3]
        day_num = date_split[2].replace(",", "")
        team_abr = utils.team_dict[data.at[i, "Home"]]
        ot_check = data.at[i, "OT"]

        if 1 <= int(day_num) <= 9:
            day_num = f"0{day_num}"  #  Add leading zero to digits under 10. (9 -> 09)

        url = f"https://www.basketball-reference.com/boxscores/{year_num}{month_num}{day_num}0{team_abr}.html"

        time.sleep(1)
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")

        arrays = []

        """
        Sets which tables to index in the HTML based on OT status of the game
        Games with more OTs add more tables, so adjustments are made to retrieve the ones desired
        Table 1:  Away Basic
        Table 2:  Away Advanced
        Table 3:  Home Basic
        Table 4:  Home Advanced 
        [Away Basic, Away Advanced, Home Basic, Home Advanced]
        """

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

        """
        Retrieves data from each table in the table_indexes
        Concats them on one line, then appends to a final array
        """

        for i in table_indexes:
            table = soup.find_all(
                name="table", attrs={"class": "sortable stats_table"}
            )[i]

            last_row = table("tr")[-1]
            td = last_row.find_all("td")
            arr = [tr.text for tr in td]

            arrays.append(arr)

        temp = np.concatenate(arrays, axis=0)
        final.append(temp)

    df = pd.DataFrame(list(map(np.ravel, final)))

    final = pd.concat([data, df], axis=1, join="outer")
    final.to_csv("BoxscoreData.csv", index=None)

    return final


@utils.timerun
def final_team_data(data=pd.read_csv("FullGamesFinal.csv")):
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

    df = pd.DataFrame(list(map(np.ravel, full_arrays)), columns=features)

    final = pd.concat([filtered, df], axis=1, join="outer")
    final.to_csv("TrainDataRawAdv.csv", index=None)


if __name__ == "__main__":
    # collectgames(2005, 2022)
    # final_boxscore_data = get_boxscore_data()
    # set_extras()
    # final_team_data()

    collect_specific(2023, utils.months_reg)
