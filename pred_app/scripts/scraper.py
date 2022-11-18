"""
This module contains Data Scraping Classes and Methods
"""
import time
from datetime import date
import requests
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from bs4 import BeautifulSoup
from tqdm import tqdm
from scripts import const, dicts


class Scraper:
    """
    Contains methods for scraping schedule, boxscore, and team stat data
    """

    def __init__(self):
        self.train_data = pd.read_sql_table("training_data", const.ENGINE)

    def get_sch_by_year(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collects schedule data for specific year
        """

        months = self.__map_months(year)
        games = self.__scrape_sch(year, months)

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
        print(played["Date"])
        played = self.__set_extras(played)

        upcoming = data[data["H-Pts"] == ""]
        upcoming.drop(upcoming.columns[[3, 5, 6]], axis=1, inplace=True)

        return played, upcoming

    def get_sch_by_range(
        self, start_year: int, end_year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collects schedule data for specific range
        """

        first = True

        for year in range(start_year, end_year + 1):
            months = self.__map_months(year)
            games = self.__scrape_sch(year, months)

            if first:
                data = pd.DataFrame(list(map(np.ravel, games)))
                first = False

            else:
                temp = pd.DataFrame(list(map(np.ravel, games)))
                data = pd.concat([data, temp], axis=0, join="outer")

        data = pd.DataFrame(data)

        if len(data.columns) == 10:
            data.drop(data.columns[[5, 7, 8, 9]], axis=1, inplace=True)
        else:
            data.drop(data.columns[[6, 8, 9, 10]], axis=1, inplace=True)

        data.columns = ["Date", "Time", "Away", "A-Pts", "Home", "H-Pts", "OT"]
        data["OT"] = data["OT"].str.replace("Unnamed: 7", "")
        data["Date"] = pd.to_datetime(data["Date"])

        played = data[data["H-Pts"] != ""].reset_index(drop=True)
        print(played["Date"])
        played = self.__set_extras(played)

        played.to_sql("simulator_sch", const.ENGINE, if_exists="replace", index=False)

        upcoming = data[data["H-Pts"] == ""]
        upcoming.drop(upcoming.columns[[3, 5, 6]], axis=1, inplace=True)

        return played, upcoming

    def get_boxscore_data_from_sch(
        self,
        data: pd.DataFrame,
        from_csv: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves boxscore data for every game in a given schedule
        """

        final = []

        for i in tqdm(data.index):
            arrays = []

            if float(data.at[i, "H-Pts"]) > float(data.at[i, "A-Pts"]):
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
                    day_num = f"0{day_num}"  #  Adds leading zero to digits under 10. (9 -> 09)

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
        final.to_csv("BoxscoreData_2008.csv", index=None)

        return final

    def get_training_data_from_sch(self, data: pd.DataFrame) -> pd.DataFrame:
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

                team_stats = pd.concat(
                    [basic_stats, advanced_stats], axis=1, join="outer"
                )
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

        return final

    def clean_boxscore_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans up boxscore data for database committing/returning
        """

        data.columns = data.columns
        data = data.drop(data.columns[[10, 29, 30, 42, 45, 46, 65, 66, 78, 81]], axis=1)
        data.columns = const.BOX_FEATURES

        return data

    def __map_months(self, year: int) -> list:
        """
        Returns season's months list for given year
        """

        return dicts.months_map.get(year, const.MONTHS_REG)

    def __scrape_sch(self, year: str, months: list) -> pd.DataFrame:
        """
        Scrapes schedule of given year, including upcoming games
        """

        arrays = []

        for month in tqdm(months):
            time.sleep(3)
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"

            page = requests.get(url, headers=const.SCH_HEADER, timeout=60)
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

    def __set_extras(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Season ID and MOV columns
        """

        for i in data.index:
            if i == 0:
                print(type(data.at[i, "Date"]))

            start_y = str(data.at[i, "Date"]).split("-")[0]
            start_mon = str(data.at[i, "Date"]).split("-")[1]

            if int(start_mon) <= 8:
                start_y = str(int(start_y) - 1)

            end_y = list(str(int(start_y) + 1))
            end_y = end_y[2] + end_y[3]

            data.at[i, "SeasonID"] = start_y + "-" + end_y
            data.at[i, "MOV"] = int(data.at[i, "H-Pts"]) - int(data.at[i, "A-Pts"])

        return data
