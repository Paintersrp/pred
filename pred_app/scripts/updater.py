import sys
import requests
import typing as t
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import date
from collections import defaultdict
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from scripts import const, dicts
from scripts.ratings import current_massey
from scripts.scrape import collect_sch_by_year, get_boxscore_data, clean_box_data

class Updater:
    def __init__(self):
        pass

    def todays_games_json(self) -> t.Any:
        req = requests.get(const.SCH_JSON_URL, headers=const.SCH_HEADER, timeout=60)
        data = req.json().get("gs").get("g")

        if not data:
            print("No games scheduled today.")
            sys.exit()

        return data

    def team_stats(self) -> t.Any:
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
            ["TEAM_NAME", "TEAM_ID", "CFID", "CFPARAMS", "W", "L", "W_PCT", "GP", "MIN"],
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

    def massey(self):
        data = pd.read_sql_table("prediction_history_net", const.ENGINE)

        if data["Date"].unique()[0] != str(date.today()):
            temp = collect_sch_by_year(2023)
        else:
            temp = pd.read_sql_table("2023_played_games", const.ENGINE)

        massey_ratings = current_massey(temp, "2022-23")

        return massey_ratings

    def todays_games(
    self, data: t.Any, team_stats: t.Any, massey_ratings: t.Any
) -> pd.DataFrame:
    #  pylint: disable=too-many-locals
        """
        Parses general game data from schedule json
        Adds team stats and massey ratings to game data
        Returns game data and team stats to be featured
        """

        full_arrays = []
        ratings_array = []

        for matchup in data:
            massey = []
            game_date = matchup.get("gcode").split("/")[0]
            game_date = "-".join([game_date[:4], game_date[4:6], game_date[6:]])
            game_time = matchup.get("stt")

            home_line = matchup.get("h")
            away_line = matchup.get("v")
            home_team = home_line.get("tc") + " " + home_line.get("tn")
            away_team = away_line.get("tc") + " " + away_line.get("tn")
            home_stats = team_stats.get_group(home_team)
            away_stats = team_stats.get_group(away_team)

            home_massey = (
                massey_ratings.get_group(home_team)
                .sort_index(axis=0, ascending=False)
                .head(1)["Massey"]
            )

            away_massey = (
                massey_ratings.get_group(away_team)
                .sort_index(axis=0, ascending=False)
                .head(1)["Massey"]
            )

            game = np.concatenate((away_stats, home_stats), axis=0)
            full_arrays.append(game)

            massey.extend(
                [
                    game_date,
                    round(float(away_massey), 2),
                    round(float(home_massey), 2),
                    game_time,
                ]
            )

            ratings_array.append(massey)

        features = ["A_" + col for col in away_stats.columns] + [
            "H_" + col for col in home_stats.columns
        ]

        game_data = pd.DataFrame(list(map(np.ravel, full_arrays)), columns=features)

        rating_data = pd.DataFrame(
            ratings_array,
            columns=["Date", "A_Massey", "H_Massey", "Game Time"],
        )

        game_data = pd.concat([game_data, rating_data], axis=1, join="outer")

        return game_data, features

    def commit_metrics(self, metrics_data: list) -> pd.DataFrame:
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

        prec_mean = self.metrics_table["Precision"].agg(np.mean)
        acc_mean = self.metrics_table["Accuracy"].agg(np.mean)
        log_mean = self.metrics_table["Logloss"].agg(np.mean)

        print("      Score Averages     ")
        print(f"Precision: {round(prec_mean,2)}%")
        print(f"Accuracy:  {round(acc_mean,2)}%")
        print(f"Logloss:   {round(log_mean,2)}%")
        print("-----------------------------")

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
            full_data, columns=["Mean", "Min", "Max", "Std"], index=self.metrics_table.columns
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
            "metric_scores", const.ENGINE, if_exists="replace", index=self.metrics_table.columns
        )

        print(self.metrics_table)

    def commit_history(self, data: pd.DataFrame, features: list) -> None:
        """
        Commits model predictions to database history tables
        Uses parameter features to determine which model is being committed
        """
        pred_history_update = data[["Date", "A_Team", "A_Odds", "H_Team", "H_Odds"]]

        if features == const.NET_FULL_FEATURES:
            old_pred_history = pd.read_sql_table("prediction_history_net", const.ENGINE)

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
            old_pred_history = pd.read_sql_table("prediction_history_massey", const.ENGINE)

            new_pred_history = pd.concat(
                [pred_history_update, old_pred_history], axis=0, join="outer"
            )

            new_pred_history.drop_duplicates(
                subset=["Date", "A_Team"], keep="first", inplace=True
            )

            new_pred_history.to_sql(
                "prediction_history_massey", const.ENGINE, if_exists="replace", index=False
            )

    def commit_preds(self, data: pd.DataFrame) -> None:
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

        data.columns = [
            [
                "Win%",
                "Net",
                "Massey",
                "Odds",
                "Away Team",
                "Time",
                "Home Name",
                "Odds.1",
                "Massey.1",
                "Net.1",
                "Win%.1",
            ]
        ]

        data.to_sql("today_preds", const.ENGINE, if_exists="replace", index=False)
        print(data)

    def update_boxscore_data(self) -> None:
        """
        Loads previous boxscore data and most recent list of played games
        Compares games in both, filtering to only what's not in the database
        Scrapes missing game data
        """

        box = pd.read_sql_table("boxscore_data", const.ENGINE)
        played = pd.read_sql_table("2023_played_games", const.ENGINE)
        box["Date"] = pd.to_datetime(box["Date"]).dt.date
        played["Date"] = pd.to_datetime(played["Date"]).dt.date

        box_dates = box["Date"].unique()
        played_dates = played["Date"].unique()

        update_check = [ele for ele in played_dates]

        for d in played_dates:
            if d in box_dates:
                update_check.remove(d)

        if update_check == []:
            print("Data is up-to-date.")
            sys.exit()

        else:
            mask = played["Date"] >= update_check[0]
            games_to_update = played.loc[mask].reset_index(drop=True)

        new_data = get_boxscore_data(games_to_update)
        new_data = clean_box_data(new_data)
        new_box = pd.concat([box, new_data], axis=0, join="outer").reset_index(drop=True)

        new_box["Outcome"] = np.where(
            new_box["H-Pts"].astype(float) > new_box["A-Pts"].astype(float), 1, 0
        )

        new_box.to_sql("boxscore_data", const.ENGINE, if_exists="replace", index=False)

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

        for i in range(len(unordered_lists)):
            rows = unordered_lists[i].find_all("li")

            for table_row in rows:
                player_name = table_row.find("a")
                status = table_row.find("span")

                if player_name != None and status != None:
                    injury_dict[team_names[i].text].append(
                        (player_name.text + "-" + status.text.replace("GTD", "TBD"))
                    )

        # for item in injury_dict.items():
        #     print(item)

        # print(injury_dict.keys())

        return injury_dict

    def commit_full_stats(self):
        """
        Combines daily team stat and massey rating updates
        """
        massey = pd.read_sql_table("current_massey", const.ENGINE)
        team = pd.read_sql_table("team_stats", const.ENGINE)
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
        arr = []
        predicted = pd.read_sql_table("prediction_history_net", const.ENGINE)
        predicted["Actual"] = "TBD"
        print(predicted)

        played = pd.read_sql_table("2023_played_games", const.ENGINE)

        played["Away"] = np.where(
            played["Away"] == "Los Angeles Clippers", "LA Clippers", played["Away"]
        )

        played["Home"] = np.where(
            played["Home"] == "Los Angeles Clippers", "LA Clippers", played["Home"]
        )

        pred_mask = predicted["Date"] != str(date.today())
        predicted = predicted.loc[pred_mask].reset_index(drop=True)

        for d in predicted["Date"].unique():
            mov_dict = {}
            played_mask = played["Date"] == d
            filtered_played = played.loc[played_mask].reset_index(drop=True)

            history_mask = predicted["Date"] == d
            filtered_predicted = predicted.loc[history_mask].reset_index(drop=True)
            print(filtered_predicted)

            for i in filtered_played.index:
                mov_dict[filtered_played.at[i, "Away"]] = filtered_played.at[i, "MOV"]

            print(mov_dict)

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
        # arr.to_sql(f"prediction_scoring", const.ENGINE, if_exists="replace", index=False)
        print(arr)

        print(sum(arr.Outcome == 1))
        print(sum(arr.Outcome == 0))


    def commit_feature_scores(self, scores):
        scores.to_sql("feature_scores", const.ENGINE, if_exists="replace", index=False)

    def commit_hyper_scores(self, hyper_scores):
        hyper_scores.to_sql("hyper_scores", const.ENGINE, if_exists="replace", index=False)