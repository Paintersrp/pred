"""
This module contains Data Updater Classes and Methods
"""
import sys
import typing as t
from datetime import date
from collections import defaultdict
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from scripts import const, dicts
from scripts.ratings import current_massey
from scripts.scraper import Scraper
from scripts.handler import GeneralHandler
from scripts.initialize import clean_odds_data

DATAHANDLER = GeneralHandler()
SCRAPER = Scraper()


class Updater:
    """
    Contains methods for building and updating database/tables
    """

    def __init__(self):
        self.metrics_table = None

    def update_games_json(self) -> t.Any:
        """
        Returns JSON data of today's games from NBA.com
        """

        req = requests.get(const.SCH_JSON_URL, headers=const.SCH_HEADER, timeout=60)

        if req.status_code == 200:
            data = req.json().get("gs").get("g")

            if not data:
                print("No games scheduled today.")
                sys.exit()
        else:
            raise Exception(f"JSON Request: Status Code - {req.status_code} ")

        return data

    def update_team_stats(self) -> t.Any:
        """
        Updates and returns current season team stats, grouped by Team Name
        """

        basic_stats = ldts.LeagueDashTeamStats(
            per_mode_detailed="Per100Possessions", season="2022-23"
        ).league_dash_team_stats.get_data_frame()

        basic_stats.drop(["TEAM_ID", "CFID", "CFPARAMS"], axis=1, inplace=True)

        basic_stats = basic_stats.rename(columns={"TEAM_NAME": "Team"})

        advanced_stats = ldts.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season="2022-23",
        ).league_dash_team_stats.get_data_frame()

        advanced_stats.drop(
            [
                "TEAM_NAME",
                "TEAM_ID",
                "CFID",
                "CFPARAMS",
                "W",
                "L",
                "W_PCT",
                "GP",
                "MIN",
            ],
            axis=1,
            inplace=True,
        )

        team_stats = pd.concat([basic_stats, advanced_stats], axis=1, join="outer")
        team_stats["Conf"] = team_stats["Team"].map(dicts.conf_dict)
        team_stats["Record"] = team_stats[["W", "L"]].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )
        team_stats.to_sql("team_stats", const.ENGINE, if_exists="replace", index=False)
        team_stats = team_stats.groupby("Team")

        return team_stats

    def update_schedule(self) -> None:
        """
        Updates schedule of played and upcoming games
        """

        played, upcoming = SCRAPER.get_sch_by_year(2023)

        played.to_sql(
            "2023_played_games", const.ENGINE, if_exists="replace", index=False
        )

        upcoming.to_sql(
            "2023_upcoming_games", const.ENGINE, if_exists="replace", index=False
        )

    def update_massey(self) -> pd.DataFrame:
        """
        Updates and returns current season massey ratings, grouped by Team Name
        """

        played = DATAHANDLER.schedule_by_year(2023)
        raw_massey = current_massey(played, "2022-23")
        cur_massey = raw_massey.sort_index(axis=0, ascending=False)
        cur_massey = cur_massey.groupby("Name").head(1).reset_index(drop=True)
        cur_massey.drop(cur_massey.tail(1).index, inplace=True)
        cur_massey["Conf"] = cur_massey["Name"].map(dicts.conf_dict)
        cur_massey = cur_massey.sort_values("Massey", ascending=False).reset_index(
            drop=True
        )

        cur_massey.to_sql(
            "Current_Massey", const.ENGINE, if_exists="replace", index=False
        )

        massey_ratings = raw_massey.groupby("Name")

        return massey_ratings

    def update_metrics(self, metrics_data: list) -> pd.DataFrame:
        """
        Builds table of scoring metrics and commits to database
        """

        full_data = []

        self.metrics_table = pd.DataFrame(
            metrics_data,
            columns=[
                "Precision",
                "Recall",
                "Accuracy",
                "Logloss",
                "ROC",
                "Correct",
                "Incorrect",
                "Games Tested",
            ],
        )

        for column in self.metrics_table:
            temp = []
            temp.extend(
                [
                    self.metrics_table[column].agg(np.mean),
                    self.metrics_table[column].agg(np.min),
                    self.metrics_table[column].agg(np.max),
                    self.metrics_table[column].agg(np.std),
                ]
            )
            full_data.append(temp)

        self.metrics_table = pd.DataFrame(
            full_data,
            columns=["Mean", "Min", "Max", "Std"],
            index=self.metrics_table.columns,
        )

        self.metrics_table["Metric"] = [
            "Precision",
            "Recall",
            "Accuracy",
            "Logloss",
            "ROC-AUC",
            "Correct",
            "Incorrect",
            "Games Tested",
        ]
        self.metrics_table = self.metrics_table[["Metric", "Mean", "Min", "Max", "Std"]]

        self.metrics_table.to_sql(
            "metric_scores",
            const.ENGINE,
            if_exists="replace",
            index=self.metrics_table.columns,
        )

    def update_history(self, data: pd.DataFrame, features: list) -> None:
        """
        Commits model predictions to database history table

        Uses parameter features to determine which model is being committed
        """

        pred_history_update = data[["Date", "A_Team", "A_Odds", "H_Team", "H_Odds"]]

        if features == const.NET_FULL_FEATURES:
            # old_pred_history = pd.read_sql_table("prediction_history_net", const.ENGINE)
            old_pred_history = DATAHANDLER.prediction_history()

            new_pred_history = pd.concat(
                [pred_history_update, old_pred_history], axis=0, join="outer"
            )

            new_pred_history.drop_duplicates(
                subset=["Date", "A_Team"], keep="first", inplace=True
            )

            new_pred_history.to_sql(
                "prediction_history_net", const.ENGINE, if_exists="replace", index=False
            )

        elif features == const.MASSEY_FULL_FEATURES:

            # old_pred_history = pd.read_sql_table(
            #     "prediction_history_massey", const.ENGINE
            # )
            old_pred_history = DATAHANDLER.prediction_history_massey()

            new_pred_history = pd.concat(
                [pred_history_update, old_pred_history], axis=0, join="outer"
            )

            new_pred_history.drop_duplicates(
                subset=["Date", "A_Team"], keep="first", inplace=True
            )

            new_pred_history.to_sql(
                "prediction_history_massey",
                const.ENGINE,
                if_exists="replace",
                index=False,
            )

    def update_preds(self, data: pd.DataFrame) -> None:
        """
        Commits model predictions display table to database
        """

        data = data[
            [
                "A_W_PCT",
                "A_NET_RATING",
                "A_Massey",
                "A_Odds",
                "A_Team",
                "Game Time",
                "H_Team",
                "H_Odds",
                "H_Massey",
                "H_NET_RATING",
                "H_W_PCT",
            ]
        ]

        data["Game Time"] = data["Game Time"].str.split(" ").str[0]

        # data.columns = [
        #     [
        #         "Win%",
        #         "Net",
        #         "Massey",
        #         "Odds",
        #         "Away Team",
        #         "Time",
        #         "Home Name",
        #         "Odds.1",
        #         "Massey.1",
        #         "Net.1",
        #         "Win%.1",
        #     ]
        # ]

        data.to_sql("today_preds", const.ENGINE, if_exists="replace", index=False)

    def update_boxscore_data(self) -> None:
        """
        Loads previous boxscore data and most recent list of played games

        Compares games in both, filtering to only what's not in the database

        Scrapes missing game data, committing to boxscore data
        """

        # box = pd.read_sql_table("boxscore_data", const.ENGINE)
        # played = pd.read_sql_table("2023_played_games", const.ENGINE)

        box = DATAHANDLER.boxscore_data()
        played = DATAHANDLER.schedule_by_year(2023)

        box["Date"] = pd.to_datetime(box["Date"]).dt.date
        played["Date"] = pd.to_datetime(played["Date"]).dt.date

        box_dates = box["Date"].unique()
        played_dates = played["Date"].unique()

        update_check = list(played_dates)

        for game_date in played_dates:
            if game_date in box_dates:
                update_check.remove(game_date)

        if not update_check:
            print("Boxscore data is up-to-date.")

        else:
            mask = played["Date"] >= update_check[0]
            games_to_update = played.loc[mask].reset_index(drop=True)

            new_data = SCRAPER.get_boxscore_data_from_sch(games_to_update)
            new_data = SCRAPER.clean_boxscore_data(new_data)
            new_box = pd.concat([box, new_data], axis=0, join="outer").reset_index(
                drop=True
            )

            new_box["Outcome"] = np.where(
                new_box["H-Pts"].astype(float) > new_box["A-Pts"].astype(float), 1, 0
            )

            new_box.to_sql(
                "boxscore_data", const.ENGINE, if_exists="replace", index=False
            )

    def update_injuries(self) -> defaultdict(list):
        """
        Collects daily line up reports and enters them into dictionary

        Returns dictionary of team lineup statuses
        """
        url = "https://www.rotowire.com/basketball/nba-lineups.php"

        page = requests.get(url, timeout=60)
        soup = BeautifulSoup(page.text, "html.parser")

        team_names = soup.find_all(name="div", attrs={"class": "lineup__abbr"})
        unordered_lists = soup.find_all(name="ul", attrs={"class": "lineup__list"})

        injury_dict = defaultdict(list)

        for i, lineup_list in enumerate(unordered_lists):
            for detail in lineup_list.find_all("li"):
                player_name = detail.find("a")
                status = detail.find("span")

                if player_name is not None and status is not None:
                    injury_dict[team_names[i].text].append(
                        (player_name.text + "-" + status.text.replace("GTD", "TBD"))
                    )

        print(injury_dict["ORL"])

        return injury_dict

    def update_full_stats(self) -> None:
        """
        Combines daily team stat and massey rating updates for display
        """

        # massey = pd.read_sql_table("current_massey", const.ENGINE)
        # team = pd.read_sql_table("team_stats", const.ENGINE)

        massey = DATAHANDLER.current_massey()
        team = DATAHANDLER.raw_team_stats()

        massey.sort_values("Name", ascending=True, inplace=True)
        massey.reset_index(drop=True, inplace=True)

        combined = pd.concat([team, massey["Massey"]], axis=1, join="outer")
        combined = combined[combined.columns.drop(list(combined.filter(regex="_RANK")))]

        combined = combined[
            [
                "Team",
                "Record",
                "Conf",
                "Massey",
                "PTS",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "OREB",
                "DREB",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
                "PIE",
                "FG_PCT",
                "FG3_PCT",
                "TS_PCT",
            ]
        ]

        combined = combined.rename(
            columns={
                "TS_PCT": "TS%",
                "OFF_RATING": "OFF",
                "DEF_RATING": "DEF",
                "NET_RATING": "NET",
                "OREB": "ORB",
                "DREB": "DRB",
                "FG_PCT": "FG%",
                "FG3_PCT": "FG3%",
            }
        )

        combined["Massey"] = round(combined["Massey"], 2)
        combined.to_sql("all_stats", const.ENGINE, if_exists="replace", index=False)

    def update_history_outcomes(self) -> None:
        """
        Loads most recent prediction history and played game scores

        Adds actual scores and outcome to prediction history

        Commits prediction scoring table to database
        """

        new_history = []

        # predicted = pd.read_sql_table("prediction_history_net", const.ENGINE)
        predicted = DATAHANDLER.prediction_history()
        predicted["Actual"] = "TBD"

        # played = pd.read_sql_table("2023_played_games", const.ENGINE)
        played = DATAHANDLER.schedule_by_year(2023)

        played["Away"] = np.where(
            played["Away"] == "Los Angeles Clippers", "LA Clippers", played["Away"]
        )

        played["Home"] = np.where(
            played["Home"] == "Los Angeles Clippers", "LA Clippers", played["Home"]
        )

        pred_mask = predicted["Date"] != str(date.today())
        predicted = predicted.loc[pred_mask].reset_index(drop=True)

        for game_date in predicted["Date"].unique():
            mov_dict = {}
            played_mask = played["Date"] == game_date
            filtered_played = played.loc[played_mask].reset_index(drop=True)

            history_mask = predicted["Date"] == game_date
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

            new_history.append(filtered_predicted)

        new_history = pd.concat(new_history, axis=0, join="outer")

        # previous_history = pd.read_sql_table("prediction_scoring", const.ENGINE)
        previous_history = DATAHANDLER.pred_scoring()

        combined = pd.concat([previous_history, new_history], axis=0, join="outer")
        combined.drop_duplicates(subset=["Date", "A_Team"], keep="first", inplace=True)
        combined = combined.sort_values("Date", ascending=False).reset_index(drop=True)
        combined.to_sql(
            "prediction_scoring", const.ENGINE, if_exists="replace", index=False
        )

    def update_upcoming(self) -> None:
        """
        Updates upcoming schedule of unplayed games data
        """

        # upcoming_games = pd.read_sql_table("2023_upcoming_games", const.ENGINE)
        upcoming_games = DATAHANDLER.upcoming_schedule_by_year(2023)
        today_mask = upcoming_games["Date"] != str(date.today())
        upcoming_games = upcoming_games.loc[today_mask].reset_index(drop=True)
        upcoming_games.to_sql(
            "upcoming_schedule_table", const.ENGINE, if_exists="replace", index=False
        )

    def update_feature_scores(self, scores) -> None:
        """
        Commits feature scores to database
        """

        scores.to_sql("feature_scores", const.ENGINE, if_exists="replace", index=False)

    def update_hyper_scores(self, hyper_scores) -> None:
        """
        Commits hyperparameter scores to database
        """

        hyper_scores.to_sql(
            "hyper_scores", const.ENGINE, if_exists="replace", index=False
        )

    def update_odds_full(self) -> None:
        """
        Yar
        """

        old_odd_history = pd.read_sql_table("full_odds_history", const.ENGINE)
        new_odd_history = pd.read_sql_table("current_odds_history", const.ENGINE)

        new_odd_history = pd.concat(
            [old_odd_history, new_odd_history], axis=0, join="outer"
        ).reset_index(drop=True)
        new_odd_history.drop_duplicates(
            subset=["Date", "H_Team"], keep="first", inplace=True
        )
        new_odd_history.to_sql(
            "full_odds_history", const.ENGINE, if_exists="replace", index=False
        )

    def update_odds_current(self) -> None:
        """
        Yar
        """

        data = pd.read_excel(const.ODDS_UPDATE_URL)
        data["Open"] = np.where(data["Open"] == "pk", 0, data["Open"])
        data["Open"] = np.where(data["Open"] == "PK", 0, data["Open"])
        data["Close"] = np.where(data["Close"] == "pk", 0, data["Close"])
        data["ML"] = np.where(data["ML"] == "NL", 0, data["ML"])
        data["Open"] = data["Open"].astype(int)
        data["Close"] = data["Open"].astype(int)

        for i in data["Date"].index:
            date_check = list(str(data.at[i, "Date"]))

            if len(date_check) == 4:
                month_check = date_check[0] + date_check[1]
                day_check = date_check[2] + date_check[3]
            else:
                month_check = date_check[0]
                day_check = date_check[1] + date_check[2]

            if int(month_check) > 9:
                data.at[i, "Date"] = f"{2023-1}" + "-" + month_check + "-" + day_check
            else:
                data.at[i, "Date"] = f"{2023}" + "-" + month_check + "-" + day_check

        data = clean_odds_data(data)
        data.to_sql(
            "current_odds_history", const.ENGINE, if_exists="replace", index=False
        )
