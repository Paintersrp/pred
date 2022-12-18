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
from scripts.initialize import clean_odds_data, clean_train

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
            raise Exception(f"JSON Request: Status Code - {req.status_code}")

        return data

    def update_team_stats(self, per_100: bool = True) -> t.Any:
        """
        Updates and returns current season team stats, grouped by Team Name
        """

        if per_100:
            per_mode = "Per100Possessions"
        else:
            per_mode = "PerGame"

        basic_stats = ldts.LeagueDashTeamStats(
            per_mode_detailed=per_mode, season="2022-23"
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
                "W_RANK",
                "L_RANK",
                "W_PCT_RANK",
                "MIN_RANK",
                "GP_RANK",
            ],
            axis=1,
            inplace=True,
        )

        team_stats = pd.concat([basic_stats, advanced_stats], axis=1, join="outer")
        team_stats["Conf"] = team_stats["Team"].map(dicts.conf_dict)
        team_stats["Record"] = team_stats[["W", "L"]].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )

        elos = DATAHANDLER.current_elos()
        final_data = pd.concat([team_stats, elos["ELO"]], axis=1, join="outer")

        final_data = final_data[
            final_data.columns.drop(list(final_data.filter(regex="_RANK")))
        ]

        final_data = final_data[
            final_data.columns.drop(list(final_data.filter(regex="E_")))
        ]

        final_data.columns = final_data.columns.str.replace("_PCT", "%")
        final_data.columns = final_data.columns.str.replace("_RATING", "")

        if not per_100:
            final_data.to_sql(
                "per_game_team_stats", const.ENGINE, if_exists="replace", index=False
            )

            final_data.to_json(
                "C:/Python/pred_app/pred_react_v2/public/data/per_game_team_stats.json",
                orient="records",
            )
        else:
            final_data.to_sql(
                "team_stats", const.ENGINE, if_exists="replace", index=False
            )
            final_data.to_json(
                "C:/Python/pred_app/pred_react_v2/public/data/per_100_team_stats.json",
                orient="records",
            )

        grouped = team_stats.groupby("Team")

        return team_stats, grouped

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

        upcoming.to_json(
            "C:/Python/pred_app/pred_react_v2/public/data/upcoming.json",
            orient="records",
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

    def update_metrics(self, metrics_data: list) -> None:
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
            old_pred_history = DATAHANDLER.prediction_history()
            old_pred_history = old_pred_history.rename(
                columns={"Away": "A_Team", "Home": "H_Team"}
            )
            old_pred_history["Date"] = pd.to_datetime(old_pred_history["Date"]).dt.date

            new_pred_history = (
                pd.concat([pred_history_update, old_pred_history], axis=0, join="outer")
                .sort_values("Date", ascending=False)
                .reset_index(drop=True)
            )

            new_pred_history.drop_duplicates(
                subset=["Date", "A_Team"], keep="first", inplace=True
            )

            new_pred_history["A_Odds"] = round(
                new_pred_history["A_Odds"].astype(float), 3
            )
            new_pred_history["H_Odds"] = round(
                new_pred_history["H_Odds"].astype(float), 3
            )

            new_pred_history.to_sql(
                "prediction_history_v2", const.ENGINE, if_exists="replace", index=False
            )

        elif features == const.MASSEY_FULL_FEATURES:
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

        data.to_sql("today_preds", const.ENGINE, if_exists="replace", index=False)

    def update_boxscore_data(self) -> None:
        """
        Loads previous boxscore data and most recent list of played games

        Compares games in both, filtering to only what's not in the database

        Scrapes missing game data, committing to boxscore data
        """

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
        url = const.INJURY_URL

        page = requests.get(url, timeout=60)
        soup = BeautifulSoup(page.text, "html.parser")

        team_names = soup.find_all(name="div", attrs={"class": "lineup__abbr"})
        unordered_lists = soup.find_all(name="ul", attrs={"class": "lineup__list"})

        injury_dict = defaultdict(list)

        for i, lineup_list in enumerate(unordered_lists):
            for detail in lineup_list.find_all("li"):
                player_name = detail.find("a")
                status_check = detail.find(name="div", attrs={"style": "width:15px;"})

                if player_name is not None:
                    if status_check is None:
                        injury_dict[team_names[i].text].append((player_name.text))

        return injury_dict

    def update_full_stats_per_100(self) -> None:
        """
        Combines daily team stat and massey rating updates for display
        """

        massey = DATAHANDLER.current_massey()
        elos = DATAHANDLER.current_elos()
        team = DATAHANDLER.raw_team_stats()

        massey.sort_values("Name", ascending=True, inplace=True)
        massey.reset_index(drop=True, inplace=True)

        combined = pd.concat(
            [team, massey["Massey"], elos["ELO"]], axis=1, join="outer"
        )
        combined = combined[combined.columns.drop(list(combined.filter(regex="_RANK")))]

        combined = combined[
            [
                "Team",
                "W",
                "L",
                "Conf",
                "Massey",
                "ELO",
                "PTS",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "OREB",
                "DREB",
                "OFF",
                "DEF",
                "NET",
                "PIE",
                "FG%",
                "FG3%",
                "TS%",
            ]
        ]

        combined["Massey"] = round(combined["Massey"], 2)
        combined["ELO"] = round(combined["ELO"], 2)

        combined.to_sql("all_stats", const.ENGINE, if_exists="replace", index=False)

    def update_history_outcomes(self) -> None:
        """
        Loads most recent prediction history and played game scores

        Adds actual scores and outcome to prediction history

        Commits prediction scoring table to database
        """

        new_history = []
        predicted = DATAHANDLER.prediction_history()

        predicted["A_Team"] = np.where(
            predicted["A_Team"] == "Los Angeles Clippers",
            "LA Clippers",
            predicted["A_Team"],
        )

        predicted["H_Team"] = np.where(
            predicted["H_Team"] == "Los Angeles Clippers",
            "LA Clippers",
            predicted["H_Team"],
        )

        played = DATAHANDLER.schedule_by_year(2023)

        played["Away"] = np.where(
            played["Away"] == "Los Angeles Clippers", "LA Clippers", played["Away"]
        )

        played["Home"] = np.where(
            played["Home"] == "Los Angeles Clippers", "LA Clippers", played["Home"]
        )

        predicted["Date"] = pd.to_datetime(predicted["Date"]).dt.date
        played["Date"] = pd.to_datetime(played["Date"]).dt.date

        pred_mask = predicted["Date"] != date.today()
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
                filtered_predicted.at[i, "MOV"] = mov_dict[
                    filtered_predicted.at[i, "A_Team"]
                ]

                if float(filtered_predicted.at[i, "A_Odds"]) > 0.5:
                    filtered_predicted.at[i, "Pred"] = 0
                else:
                    filtered_predicted.at[i, "Pred"] = 1

                if filtered_predicted.at[i, "MOV"] < 0:
                    filtered_predicted.at[i, "Outcome"] = 0
                else:
                    filtered_predicted.at[i, "Outcome"] = 1

            new_history.append(filtered_predicted)

        new_history = pd.concat(new_history, axis=0, join="outer")
        previous_history = DATAHANDLER.pred_scoring()
        previous_history["Date"] = pd.to_datetime(previous_history["Date"]).dt.date

        new_history = pd.concat([previous_history, new_history], axis=0, join="outer")
        new_history.drop_duplicates(
            subset=["Date", "A_Team", "H_Team"], keep="first", inplace=True
        )
        new_history = new_history.sort_values("Date", ascending=False).reset_index(
            drop=True
        )

        new_history.to_sql(
            "prediction_scoring_v2", const.ENGINE, if_exists="replace", index=False
        )

        new_history["Date"] = new_history["Date"].astype(str)

        new_history.to_json(
            "C:/Python/pred_app/pred_react_v2/public/data/pred_history.json",
            orient="records",
        )

        correct = sum(new_history.Outcome == new_history.Pred)
        incorrect = sum(new_history.Outcome != new_history.Pred)
        ratio = round(correct / (correct + incorrect) * 100, 2)

        scoring = pd.DataFrame([0, correct, incorrect, ratio]).T
        scoring.columns = ["index", "correct", "incorrect", "ratio"]

        scoring.to_json(
            "C:/Python/pred_app/pred_react_v2/public/data/scoring.json",
            orient="records",
        )

    def update_upcoming(self) -> None:
        """
        Updates upcoming schedule of unplayed games data
        """

        upcoming_games = DATAHANDLER.upcoming_schedule_by_year(2023)
        today_mask = upcoming_games["Date"] != str(date.today())
        upcoming_games = upcoming_games.loc[today_mask].reset_index(drop=True)
        upcoming_games.to_sql(
            "upcoming_schedule_table", const.ENGINE, if_exists="replace", index=False
        )

    def update_feature_scores(self, scores: pd.DataFrame) -> None:
        """
        Commits feature scores to database
        """

        scores.to_sql("feature_scores", const.ENGINE, if_exists="replace", index=False)

    def update_hyper_scores(self, hyper_scores: pd.DataFrame) -> None:
        """
        Commits hyperparameter scores to database
        """

        hyper_scores.to_sql(
            "hyper_scores", const.ENGINE, if_exists="replace", index=False
        )

    def update_odds_full(self) -> None:
        """
        Commits new full odds history based on daily additions/updates
        """

        old_odd_history = DATAHANDLER.full_odds_history()
        new_odd_history = DATAHANDLER.current_odds_history()

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
        Updates to most current odds history for this season
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

    # def update_todays_lines(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Updates todays lines and commits to database
    #     """

    #     # add todays preds to general handler?
    #     dh = MetricsHandler()
    #     todays_preds = dh.today_preds()
    #     data = SCRAPER.get_todays_lines()
    #     data = SCRAPER.clean_todays_lines(data)

    #     for i in todays_preds.index:
    #         for j in data.index:
    #             if (
    #                 data.at[j, "Home"] == todays_preds.at[i, "H_Team"]
    #                 and data.at[j, "Away"] == todays_preds.at[i, "A_Team"]
    #             ):
    #                 todays_preds.at[i, "O/U"] = data.at[j, "A_OU"]
    #                 todays_preds.at[i, "A_OU_Line"] = data.at[j, "A_OU_Line"]
    #                 todays_preds.at[i, "H_OU_Line"] = data.at[j, "H_OU_Line"]
    #                 todays_preds.at[i, "Spread"] = data.at[j, "A_Spread"]
    #                 todays_preds.at[i, "A_Spr_Line"] = data.at[j, "A_Spread_Line"]
    #                 todays_preds.at[i, "H_Spr_Line"] = data.at[j, "H_Spread_Line"]
    #                 todays_preds.at[i, "A_ML"] = data.at[j, "A_ML"]
    #                 todays_preds.at[i, "H_ML"] = data.at[j, "H_ML"]

    #     data.to_sql("todays_lines", const.ENGINE, if_exists="replace", index=False)

    #     return data

    def update_training_schedule(self):
        """Func"""

        schedule = DATAHANDLER.training_schedule()
        played = DATAHANDLER.schedule_by_year(2023)

        schedule["Date"] = pd.to_datetime(schedule["Date"]).dt.date
        played["Date"] = pd.to_datetime(played["Date"]).dt.date

        training_dates = schedule["Date"].unique()
        new_dates = played["Date"].unique()

        update_check = list(new_dates)

        for game_date in new_dates:
            if game_date in training_dates:
                update_check.remove(game_date)

        if not update_check:
            print("Training schedule data is up-to-date.")
        else:
            mask = played["Date"] >= update_check[0]
            games_to_update = played.loc[mask].reset_index(drop=True)

            new_schedule = pd.concat(
                [schedule, games_to_update], axis=0, join="outer"
            ).reset_index(drop=True)

            new_schedule.to_sql(
                "training_schedule", const.ENGINE, if_exists="replace", index=False
            )

    def update_training_data(self) -> None:
        """
        Loads previous training data and most recent list of played games

        Compares games in both, filtering to only what's not in the database

        Scrapes missing game data, committing to training data
        """

        training = pd.read_sql_table("training_base", const.ENGINE)
        played = DATAHANDLER.schedule_by_year(2023)
        prev_full_sch = pd.read_sql_table("full_sch", const.ENGINE)
        training["Date"] = pd.to_datetime(training["Date"]).dt.date
        played["Date"] = pd.to_datetime(played["Date"]).dt.date

        training_dates = training["Date"].unique()
        new_dates = played["Date"].unique()

        update_check = list(new_dates)

        for game_date in new_dates:
            if game_date in training_dates:
                update_check.remove(game_date)

        if not update_check:
            print("Training data is up-to-date.")
        else:
            mask = played["Date"] >= update_check[0]
            games_to_update = played.loc[mask].reset_index(drop=True)
            new_data = SCRAPER.get_training_data_from_sch(games_to_update)

            new_full_sch = pd.concat(
                [prev_full_sch, games_to_update], axis=0, join="outer"
            ).reset_index(drop=True)

            new_full_data = pd.concat(
                [training, new_data], axis=0, join="outer"
            ).reset_index(drop=True)

            new_full_data.drop(["A_TEAM_NAME", "H_TEAM_NAME"], axis=1, inplace=True)

            new_final = clean_train(new_full_data, new_full_sch)

            new_full_sch.to_sql(
                "full_sch", const.ENGINE, if_exists="replace", index=False
            )

            new_full_data.to_sql(
                "training_base", const.ENGINE, if_exists="replace", index=False
            )

            new_final.to_sql(
                "training_data_v2", const.ENGINE, if_exists="replace", index=False
            )

    def update_elos(self, elos: dict) -> None:
        """Func"""

        elos = (
            pd.DataFrame.from_dict(elos, orient="index")
            .reset_index()
            .sort_values("index")
            .reset_index(drop=True)
        )

        elos.columns = ["Team", "ELO"]

        elos.to_json(
            "C:/Python/pred_app/pred_react_v2/public/data/elos.json", orient="records"
        )

        elos.to_sql("current_elos", const.ENGINE, if_exists="replace", index=False)

    #           NEEDS UPDATE TO ELO ADDITION VERSION
    # def update_sim_pred(self):
    #     DATAHANDLER = GeneralHandler()
    #     SCRAPER = Scraper()

    #     pred_data = DATAHANDLER.pred_sim_data()
    #     played = DATAHANDLER.schedule_by_year(2023)

    #     pred_data["Date"] = pd.to_datetime(pred_data["Date"]).dt.date
    #     played["Date"] = pd.to_datetime(played["Date"]).dt.date

    #     pred_dates = pred_data["Date"].unique()
    #     new_dates = played["Date"].unique()

    #     update_check = list(new_dates)

    #     for game_date in new_dates:
    #         if game_date in pred_dates:
    #             update_check.remove(game_date)

    #     if not update_check:
    #         print("Sim Prediction Data is up-to-date.")

    #     else:
    #         mask = played["Date"] >= update_check[0]
    #         games_to_update = played.loc[mask].reset_index(drop=True)
    #         new_data = SCRAPER.get_training_data_from_sch(games_to_update)
    #         new_data = clean_train(new_data, games_to_update)

    #         new_preds = self.predict_new(pred_data, new_data)
    #         new_data = pd.concat([new_data, new_preds], axis=1, join="outer").reset_index(
    #             drop=True
    #         )

    #         new_data = self.clean_sim_pred(new_data)
    #         new_data = self.add_analysis_columns(new_data)
    #         new_pred_data = pd.concat([pred_data, new_data], axis=0, join="outer").reset_index(
    #             drop=True
    #         )

    #         new_pred_data.to_sql(
    #             "sim_pred_data", const.ENGINE, if_exists="replace", index=False
    #         )

    # def clean_sim_pred(self, data: pd.DataFrame) -> pd.DataFrame:
    #     DATAHANDLER = GeneralHandler()

    #     odds_history = DATAHANDLER.full_odds_history()

    #     data["Date"] = pd.to_datetime(data["Date"]).dt.date
    #     odds_history["Date"] = pd.to_datetime(odds_history["Date"]).dt.date

    #     for i in data.index:
    #         game_date = data.at[i, "Date"]

    #         mask = odds_history["Date"] == game_date
    #         temp = odds_history.loc[mask].reset_index(drop=True)

    #         for j in temp.index:
    #             if temp.at[j, "H_Team"] == data.at[i, "Home"]:
    #                 data.at[i, "O/U"] = temp.at[j, "O/U"]
    #                 data.at[i, "H_ML"] = temp.at[j, "H_ML"]
    #                 data.at[i, "A_ML"] = temp.at[j, "A_ML"]
    #                 data.at[i, "Spread"] = temp.at[j, "Spread"]
    #                 data.at[i, "OU_Outcome"] = temp.at[j, "O/U_Outcome"]

    #     data.sort_values(["Date", "Home"], ascending=True, inplace=True)
    #     data.reset_index(drop=True, inplace=True)
    #     data.dropna(inplace=True)
    #     mask = (data["Spread"].astype(float) != 0) & (
    #         data["O/U"].astype(float) != 0
    #     )

    #     data = data.loc[mask].reset_index(drop=True)
    #     data = data[const.SIM_PRED_DISPLAY_FEATURES]

    #     return data

    # def add_analysis_columns(self, data: pd.DataFrame) -> pd.DataFrame:
    #     rule_col = []
    #     bet_col = []

    #     for i in data.index:
    #         if 200 <= abs(data.at[i, "A_ML"]) <= 10000:
    #             rule_col.append("Lean-Out")
    #             bet_col.append(1.00)

    #         elif 110 <= abs(data.at[i, "A_ML"]) <= 150:
    #             rule_col.append("Lean-In")
    #             bet_col.append(100 * 2)

    #         else:
    #             rule_col.append("None")
    #             bet_col.append(100)

    #     data["Rule"] = rule_col
    #     data["Bet"] = bet_col

    #     data["ML_Payout"] = np.where(
    #         data["MOV"] < 0, data["A_ML"], data["H_ML"]
    #     )

    #     data["Bet_Status"] = np.where(
    #         data["Outcome"] == data["Pred"], 1, 0
    #     )

    #     data["Value"] = np.where(
    #         data["Bet_Status"] == 1,
    #         (100 / abs(data["ML_Payout"])) * data["Bet"],
    #         -abs(100),
    #     )

    #     return data

    # def predict_new(self, training: pd.DataFrame, testing: pd.DataFrame):
    #     outcomes = training["Outcome"]
    #     testing_outcomes = testing["Outcome"]
    #     outcomes_arr = []

    #     x_matrix = xgb.DMatrix(
    #         training[const.NET_FULL_FEATURES], label=outcomes
    #     )
    #     y_matrix = xgb.DMatrix(
    #         testing[const.NET_FULL_FEATURES], label=testing_outcomes
    #     )

    #     xgb_model = xgb.train(dicts.PARAMS, x_matrix, const.NET_EPOCHS)
    #     preds = xgb_model.predict(y_matrix)

    #     for pred in preds:
    #         outcomes_arr.append(np.argmax(pred))

    #     return pd.DataFrame(outcomes_arr, columns=["Pred"])
