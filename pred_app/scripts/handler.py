"""
This module contains Data Handler Classes and Methods
"""
import typing as t
from datetime import datetime
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
        self.data = None
        self.loaded = None
        self.slot_one = None
        self.slot_two = None
        self.slot_three = None

    def load_data(self, data: t.Any) -> None:
        """Loads data for extra processing"""

        self.data = data
        self.loaded = data.copy()

    def stow_data(self, slot: int = 1) -> None:
        """
        Saves a copy of the current data for later use

        Slots available: 3
        """

        self.__slot_check(slot)

        if slot == 1:
            self.slot_one = self.data.copy()
        elif slot == 2:
            self.slot_two = self.data.copy()
        else:
            self.slot_three = self.data.copy()

    def unstow_data(self, slot: int = 1) -> None:
        """
        Loads a copy of the stowed data at given slot

        Slots available: 3
        """

        self.__slot_check(slot)

        if slot == 1:
            self.data = self.slot_one.copy()
            self.loaded = self.slot_one.copy()

        elif slot == 2:
            self.data = self.slot_two.copy()
            self.loaded = self.slot_two.copy()

        else:
            self.data = self.slot_three.copy()
            self.loaded = self.slot_three.copy()

    def concat_to_stowed(
        self, slot: int = 1, stowed_first: bool = False, axis: int = 0
    ) -> None:
        """
        Concats current data to a stowed dataset

        Data should match in length or shape, depending on axis

        Slots available: 3
        """

        self.__slot_check(slot)
        self.__concat_check(stowed_first, axis)

        if not stowed_first:
            if slot == 1:
                self.data = pd.concat(
                    [self.data, self.slot_one], axis=axis
                ).reset_index(drop=True)

            elif slot == 2:
                self.data = pd.concat(
                    [self.data, self.slot_two], axis=axis
                ).reset_index(drop=True)

            elif slot == 3:
                self.data = pd.concat(
                    [self.data, self.slot_three], axis=axis
                ).reset_index(drop=True)
        else:
            if slot == 1:
                self.data = pd.concat(
                    [self.slot_one, self.data], axis=axis
                ).reset_index(drop=True)

            elif slot == 2:
                self.data = pd.concat(
                    [self.slot_two, self.data], axis=axis
                ).reset_index(drop=True)

            elif slot == 3:
                self.data = pd.concat(
                    [self.slot_three, self.data], axis=axis
                ).reset_index(drop=True)

    def print_data(self) -> None:
        """Prints currently loaded data"""

        print(self.data)

    def return_data(self) -> pd.DataFrame:
        """Returns currently loaded data"""

        return self.data

    def reset_data(self) -> None:
        """Resets data to unfiltered state"""

        self.data = self.loaded.copy()

    def drop_columns(self, columns_to_drop: t.Any) -> None:
        """Drops a str, list, tuple, or enumerate of columns"""

        self.data.drop(columns_to_drop, axis=1, inplace=True)

    def sort_columns(self, column: t.Any, ascending: bool) -> None:
        """Sorts given column"""

        self.data.sort_values(column, ascending=ascending, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def filter_columns(self, column, value) -> None:
        """Filters a column by a value"""

        mask = self.data[column] == value
        self.data = self.data.loc[mask].reset_index(drop=True)

    def filter_date_range(self, column, start_date, end_date) -> None:
        """Filters date column by start and end date range"""

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.data[column] = pd.to_datetime(self.data[column])

        mask = (self.data[column] >= start_date) & (self.data[column] <= end_date)
        self.data = self.data.loc[mask].reset_index(drop=True)

    def print_columns(self) -> None:
        """Prints a list of columns in current data"""

        print(*self.data.columns)

    def rename_columns(self, new_columns_dict: dict):
        """Renames columns in current data based on given dict"""

        self.data = self.data.rename(columns=new_columns_dict)

    def regex_drop_columns(self, regex_str: str) -> None:
        """Filters columns by regex input"""

        self.data = self.data[
            self.data.columns.drop(list(self.data.filter(regex=f"{regex_str}")))
        ]

    def to_csv(self, file_name) -> None:
        """Exports current data to .csv file"""

        self.data.to_csv(f"{file_name}.csv", index=None)

    def __slot_check(self, slot) -> None:
        """Func"""

        if slot > 3 or slot < 0:
            raise ValueError(f"Choose from slot 1, 2, or 3. Slot received: {slot}")

    def __concat_check(self, stowed_first, axis) -> None:
        """Func"""

        if not isinstance(stowed_first, bool):
            raise ValueError(
                f"Stowed First must be boolean - True/False. Type received: {type(stowed_first)}"
            )
        if not axis == 0 or not axis == 1:
            raise ValueError(f"Axis must be 0 or 1. Axis received: {axis}")


class GeneralHandler(Handler):
    """
    Class
    """

    def __init__(self):
        super().__init__()

    def upcoming_schedule(self) -> pd.DataFrame:
        """Returns most up-to-date upcoming schedule"""

        return pd.read_sql_table("upcoming_schedule_table", const.ENGINE)

    def general_sim_data(self) -> pd.DataFrame:
        """Returns data used for general simulations"""

        return pd.read_sql_table("simulator_data", const.ENGINE)

    def pred_sim_data(self) -> pd.DataFrame:
        """Returns data used for prediction based simulations"""

        return pd.read_sql_table("sim_pred_data", const.ENGINE)

    def current_odds_history(self) -> pd.DataFrame:
        """Returns current season game line stats (2023)"""

        return pd.read_sql_table("current_odds_history", const.ENGINE)

    def full_odds_history(self) -> pd.DataFrame:
        """Returns full game line stats (2008-2023)"""

        return pd.read_sql_table("full_odds_history", const.ENGINE)

    def training_data(self) -> pd.DataFrame:
        """Returns full model training data"""

        return pd.read_sql_table("training_data", const.ENGINE)

    def prediction_history(self) -> pd.DataFrame:
        """Returns prediction history of Net focused model"""

        return pd.read_sql_table("prediction_history_net", const.ENGINE)

    def prediction_history_massey(self) -> pd.DataFrame:
        """Returns prediction history of Massey focused model"""

        return pd.read_sql_table("prediction_history_massey", const.ENGINE)

    def schedule_by_year(self, year) -> pd.DataFrame:
        """Returns schedule of completed games for given year"""

        return pd.read_sql_table(f"{year}_played_games", const.ENGINE)

    def boxscore_data(self) -> pd.DataFrame:
        """Returns full boxscore dataset"""

        return pd.read_sql_table("boxscore_data", const.ENGINE)

    def raw_team_stats(self) -> pd.DataFrame:
        """Returns current team stats without additions"""

        return pd.read_sql_table("team_stats", const.ENGINE)

    def current_massey(self) -> pd.DataFrame:
        """Returns current massey ratings for each team"""

        return pd.read_sql_table("current_massey", const.ENGINE)

    def current_team_stats(self) -> pd.DataFrame:
        """Returns current team stats with additions"""

        return pd.read_sql_table("all_stats", const.ENGINE)

    def pred_scoring(self) -> pd.DataFrame:
        """Returns full prediction historical data"""

        return pd.read_sql_table("prediction_scoring", const.ENGINE)


class TeamsHandler(Handler):
    #  pylint: disable=too-many-instance-attributes
    """
    Team Stats Data Handler Class

    Contains methods for returning individual team stats or comparisons
    """

    def __init__(self):
        super().__init__()

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

    def return_head_to_head(self, team_one: str, team_two: str):
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

    def return_team_compare(self, home_team: str, away_team: str) -> None:
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

    def return_compare_main(self, team_one: str, team_two: str) -> pd.DataFrame:
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


class OddsHandler(Handler):
    """
    Odds Stats Data Handler Class

    Contains methods for returning individual team odds stats or comparisons
    """

    def __init__(self):
        super().__init__()
        self.data = pd.read_sql_table("odds_stats", const.ENGINE)
        self.grouped = self.data.groupby("Team")

    def team_odds_averages(self, team_name) -> t.Any:
        """Returns given team's current odds stats"""

        return self.grouped.get_group(team_name)

    def team_odds_compare(self, team_one: str, team_two: str):
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

    def today_preds(self) -> pd.DataFrame:
        """Returns today' predictions table"""

        return pd.read_sql_table("today_preds", const.ENGINE)

    def metrics(self) -> pd.DataFrame:
        """Returns today's metrics table"""

        return pd.read_sql_table("metric_scores", const.ENGINE)

    def pred_history(self) -> pd.DataFrame:
        """Returns full prediction historical data"""

        return pd.read_sql_table("prediction_scoring", const.ENGINE)

    def current_odds_history(self) -> pd.DataFrame:
        """Returns current season game line stats (2023)"""

        return pd.read_sql_table("current_odds_history", const.ENGINE)

    def feature_scores(self) -> pd.DataFrame:
        """Returns today's feature scoring table"""

        return pd.read_sql_table("feature_scores", const.ENGINE)

    def hyper_scores(self) -> pd.DataFrame:
        """Returns most recent hyperparameter scoring table"""

        return pd.read_sql_table("hyper_scores", const.ENGINE)

    def general_sim_data(self) -> pd.DataFrame:
        """Returns data used for general simulations"""

        return pd.read_sql_table("simulator_data", const.ENGINE)

    def show_recall_plot(self) -> None:
        """Returns today's Precision Recall Curve plot"""

        image = Image.open("Precision_Recall_Curve.png")
        image.show()

    def show_roc_plot(self) -> None:
        """Returns today's ROC AUC Curve plot"""

        image = Image.open("ROC_AUC_Curve.png")
        image.show()
