"""
This module contains Data Handler Classes and Methods
"""
import typing as t
from PIL import Image
import pandas as pd
import numpy as np
from scripts import const, dicts


class Handler:
    """
    Base Data Handler Class

    Contains base return and reset methods
    """

    def __init__(self):
        self.data = pd.read_sql_table("boxscore_data", const.ENGINE)

    def print_data(self):
        """Prints most currently loaded data"""

        print(self.data)

    def return_data(self) -> pd.DataFrame:
        """Returns most currently loaded data"""

        return self.data

    def reset_filter(self):
        """Resets data to unfiltered state"""

        self.data = pd.read_sql_table("boxscore_data", const.ENGINE)

    def print_schedule(self):
        """Prints most up-to-date upcoming schedule"""

        schedule_data = pd.read_sql_table("upcoming_schedule_table", const.ENGINE)
        print(schedule_data)

    def return_schedule(self) -> pd.DataFrame:
        """Returns most up-to-date upcoming schedule"""

        schedule_data = pd.read_sql_table("upcoming_schedule_table", const.ENGINE)
        return schedule_data


class TeamsHandler(Handler):
    #  pylint: disable=too-many-instance-attributes
    """
    Team Stats Data Handler Class

    Contains methods for returning individual team stats or comparisons
    """

    def __init__(self):
        super().__init__()
        self.data = pd.read_sql_table("boxscore_data", const.ENGINE)

    def __map_season(self, year: int) -> list:
        """Maps given year to SeasonID of given year"""

        return dicts.season_map[year]

    def return_averages(self, team_name: str) -> t.Any:
        """
        Returns team stat averages of given team
        """

        away_games = self.data.groupby("Away")
        home_games = self.data.groupby("Home")

        away_avgs = (
            away_games.get_group(team_name)[const.AWAY_FEATURES]
            .reset_index(drop=True)
            .astype(float)
        )

        home_avgs = (
            home_games.get_group(team_name)[const.HOME_FEATURES]
            .reset_index(drop=True)
            .astype(float)
        )

        final_avgs = (
            (away_avgs.agg(np.mean).values + home_avgs.agg(np.mean).values) / 2
        ).reshape(1, 14)

        final_avgs = pd.DataFrame(
            list(map(np.ravel, final_avgs)), columns=const.MATCHUP_FEATURES
        )

        final_avgs.insert(loc=0, column="Team", value=team_name)

        return round(final_avgs, 3)

    def filter_data_by_year(self, year: int):
        """
        Filters data to given year
        """

        season = self.__map_season(year)
        mask = self.data["SeasonID"] == season
        self.data = self.data.loc[mask].reset_index(drop=True)

    def filter_data_by_range(self, start_year: int = 2021, end_year: int = 2022):
        """
        Filters data to given range of years
        """

        raw_data = []

        for year in range(start_year, end_year + 1):
            season = self.__map_season(year)
            mask = self.data["SeasonID"] == season
            temp = self.data.loc[mask].reset_index(drop=True)

            raw_data.append(temp)

        self.data = pd.concat(raw_data).reset_index(drop=True)

    def compare_teams_head_to_head(self, team_one: str, team_two: str):
        """
        Compares two teams head-to-head match up averages over last 5 meetings
        """

        mask = ((self.data["Away"] == team_one) & (self.data["Home"] == team_two)) | (
            (self.data["Away"] == team_two) & (self.data["Home"] == team_one)
        )

        self.data = self.data.loc[mask].reset_index(drop=True).tail(5)

        t1_final = self.return_averages(team_one)
        t2_final = self.return_averages(team_two)

        self.data = pd.concat([t1_final, t2_final]).reset_index(drop=True)

    def compare_team_avgs(self, home_team: str, away_team: str) -> pd.DataFrame:
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

    def build_compare_main(self, team_one: str, team_two: str) -> pd.DataFrame:
        """
        Loads odds stats and team stats from database
        Builds table of each teams combined team/odds stats
        Adds difference column highlighting different in stats
        """

        all_stats = pd.read_sql_table("all_stats", const.ENGINE)[
            const.COMPARE_COLS
        ].groupby("Team")

        odds_stats = pd.read_sql_table("odds_stats", const.ENGINE)[
            const.ODDS_COLS
        ].groupby("Team")

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
        data = pd.concat([t1_full, t2_full], axis=0, join="outer").reset_index(
            drop=True
        )

        extra_stats = data[["Team", "Record"]]
        data.drop(["Team", "Record"], axis=1, inplace=True)
        diff_data = data.diff().dropna()

        temp = pd.concat([data, diff_data], axis=0, join="outer").reset_index(drop=True)
        final_data = pd.concat([extra_stats, temp], axis=1, join="outer").reset_index(
            drop=True
        )
        final_data.fillna("-", inplace=True)

        return final_data

    def current_averages(self) -> pd.DataFrame:
        """
        Builds table of all team's current averages
        """

        all_stats = pd.read_sql_table("all_stats", const.ENGINE)

        return all_stats


class OddsHandler(Handler):
    """
    Odds Stats Data Handler Class

    Contains methods for returning individual team odds stats or comparisons
    """

    def __init__(self):
        super().__init__()
        self.data = pd.read_sql_table("odds_stats", const.ENGINE)
        self.grouped = self.data.groupby("Team")

    def get_team(self, team_name) -> t.Any:
        """Returns given team's current odds stats"""

        return self.grouped.get_group(team_name)

    def compare_teams(self, team_one: str, team_two: str):
        """Returns two given team's current odds stats for comparison"""

        t1_final = self.grouped.get_group(team_one)
        t2_final = self.grouped.get_group(team_two)

        return pd.concat([t1_final, t2_final]).reset_index(drop=True)


class MetricsHandler(Handler):
    """
    Model Metrics Data Handler Class

    Contains methods for returning metrics of Daily Predictor
    """

    def __init__(self):
        super().__init__()

    def return_todays(self) -> pd.DataFrame:
        """Returns today' predictions table"""

        predictions = pd.read_sql_table("today_preds", const.ENGINE)
        return predictions

    def return_metrics(self) -> pd.DataFrame:
        """Returns today's metrics table"""

        metrics_data = pd.read_sql_table("metric_scores", const.ENGINE)
        return metrics_data

    def return_pred_history(self) -> pd.DataFrame:
        """Returns full prediction historical data"""

        pred_history = pd.read_sql_table("prediction_scoring", const.ENGINE)
        return pred_history

    def return_feature_scores(self) -> pd.DataFrame:
        """Returns today's feature scoring table"""

        feature_scores = pd.read_sql_table("feature_scores", const.ENGINE)
        return feature_scores

    def return_hyper_scores(self) -> pd.DataFrame:
        """Returns most recent hyperparameter scoring table"""

        hyper_scores = pd.read_sql_table("hyper_scores", const.ENGINE)
        return hyper_scores

    def show_recall_plot(self) -> None:
        """Returns today's Precision Recall Curve plot"""

        image = Image.open("Precision_Recall_Curve.png")
        image.show()

    def show_roc_plot(self) -> None:
        """Returns today's ROC AUC Curve plot"""

        image = Image.open("ROC_AUC_Curve.png")
        image.show()
