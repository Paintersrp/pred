"""
This module contains Data/Bet Analyzer Classes and Methods
"""
import pandas as pd
import numpy as np
from scripts import dicts, const
from scripts.handler import Handler, GeneralHandler

DATAHANDLER = GeneralHandler()


class Analyzer(Handler):
    """
    Class
    """

    def __init__(self) -> None:
        super().__init__()


class SimAnalyzer(Analyzer):
    """
    Contains methods for analyzing aspects of Simulation/Prediction Data
    """

    def __init__(self) -> None:
        super().__init__()

        self.data = DATAHANDLER.pred_sim_data()

        self.__spread_status()

    def months_analysis(self) -> pd.DataFrame:
        """
        Instantiates loop using months group filters and settings

        Returns months analysis DataFrame
        """

        team_groups = ["Date", "Date"]
        months_data, _ = self.__analysis_table_loop(
            team_groups, False, False, False, True
        )

        return months_data

    def teams_analysis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Instantiates loop using teams group filters and settings

        Returns home analysis DataFrame, away analysis DataFrame
        """

        team_groups = ["Home", "Away"]
        home, away = self.__analysis_table_loop(team_groups, False, False, True, False)

        float_cols = const.AWAY_ANALYZER_FEATURES.copy()
        float_cols.remove("Group")
        home[float_cols] = home[float_cols].astype(float)
        away[float_cols] = away[float_cols].astype(float)

        sum_features = pd.DataFrame(
            pd.concat([home, away])
            .reset_index(drop=True)
            .groupby("Group")[const.ANALYZER_SUM_FEATURES]
            .agg(np.sum)
        ).copy()
        mean_features = pd.DataFrame(
            pd.concat([home, away])
            .reset_index(drop=True)
            .groupby("Group")[const.ANALYZER_MEAN_FEATURES]
            .agg(np.mean)
        ).copy()
        combined = pd.concat([sum_features, mean_features], axis=1)
        combined.insert(0, "Group", combined.index)
        combined.reset_index(drop=True, inplace=True)

        return home, away, combined

    def rule_analysis(self) -> pd.DataFrame:
        """
        Instantiates loop using rule group filters and settings

        Returns rules analysis DataFrame
        """

        rule_groups = ["Rule", "Rule"]
        rule_data, _ = self.__analysis_table_loop(
            rule_groups, True, False, False, False
        )

        return rule_data

    def moneyline_analysis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Instantiates loop using moneyline group filters and settings

        Returns moneyline analysis DataFrame
        """

        moneyline_groups = ["H_ML", "A_ML"]

        h_ml_data, a_ml_data = self.__analysis_table_loop(
            moneyline_groups, False, True, False, False
        )

        return h_ml_data, a_ml_data

    def __analysis_table_loop(
        self,
        filter_groups: list[str],
        rule_table: bool,
        moneyline_table: bool,
        teams_table: bool,
        months_table: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # pylint: disable=too-many-locals disable=too-many-arguments
        """
        Purpose:
        --------
        Returns analysis of a given set of filters

        Parameters:
        ----------
        filter_groups:
            List[str] of columns, in sequential order, to filter for analysis.

        rule_table:
            Boolean to determine which set of filter values to use [Enables Rules Analysis]

        moneyline_table:
            Boolean to determine which set of filter values to use [Enables Moneyline Analysis]

        teams_table:
            Boolean to determine which set of filter values to use [Enables Teams Analysis]

        months_table:
            Boolean to determine which set of filter values to use [Enables Months Analysis]

        Variables:
        ---------
        zip_set_one:
            Set of values to filter the data by, based on analysis type enabled.

        zip_set_two:
            Secondary set of values to filter the data, if needed, by analysis type enabled.

        Returns:
        ----------
        Tuple of DataFrames [Away Group Stats, Home Group Stats]

        For returns where data isn't filtered by home/away, a dummy duplicate is returned second.
        """

        append_groups = [[], []]

        if moneyline_table:
            zip_set_one = range(-10000, 10001, 100)
            zip_set_two = range(-9950, 10051, 100)
        elif rule_table:
            zip_set_one = ["Lean-In", "Lean-Out", "None"]
            zip_set_two = ["", "", ""]
        elif teams_table:
            zip_set_one = dicts.conf_dict.keys()
            zip_set_two = range(0, len(dicts.conf_dict.keys()))
        elif months_table:
            zip_set_one = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            zip_set_two = range(0, len(zip_set_one))

        for filter_group, append_group in zip(filter_groups, append_groups):
            for zip_one_value, zip_two_value in zip(zip_set_one, zip_set_two):
                temp = []

                range_concat = str(zip_one_value) + " to " + str(zip_two_value)

                if moneyline_table:
                    range_concat = str(zip_one_value) + " to " + str(zip_two_value)
                    mask = (self.data[filter_group] >= zip_one_value) & (
                        self.data[filter_group] <= zip_two_value
                    )
                elif months_table:
                    range_concat = "Month" + " - " + str(zip_one_value)
                    mask = self.data[filter_group].dt.month == zip_one_value
                else:
                    range_concat = str(zip_one_value)
                    mask = self.data[filter_group] == zip_one_value

                data_group = self.data.loc[mask].reset_index(drop=True).copy()

                if not len(data_group) == 0:
                    game_count = len(data_group[filter_group])
                    pred_wins = sum(data_group["Bet_Status"] == 1)
                    pred_losses = sum(data_group["Bet_Status"] == 0)
                    pred_win_pct = round(pred_wins / game_count, 2)
                    bookie_wins = sum(data_group["ML_Payout"] < 0)
                    bookie_losses = sum(data_group["ML_Payout"] > 0)
                    bookie_win_pct = round(bookie_wins / game_count, 2)
                    net = round(sum(data_group["Value"]), 2)

                    over_pct = round(
                        sum(data_group["OU_Outcome"] == "Over") / game_count, 2
                    )

                    under_pct = round(
                        sum(data_group["OU_Outcome"] == "Under") / game_count, 2
                    )

                    spread_w_pct = round(
                        sum(data_group["Spread_Status"] == 1) / game_count, 2
                    )

                    spread_l_pct = round(
                        sum(data_group["Spread_Status"] == 0) / game_count, 2
                    )

                    temp.extend(
                        [
                            range_concat,
                            game_count,
                            pred_wins,
                            pred_losses,
                            pred_win_pct,
                            bookie_wins,
                            bookie_losses,
                            bookie_win_pct,
                            net,
                            over_pct,
                            under_pct,
                            spread_w_pct,
                            spread_l_pct,
                        ]
                    )
                    append_group.append(temp)

        h_data = pd.DataFrame(
            list(map(np.ravel, append_groups[0])),
            columns=const.HOME_ANALYZER_FEATURES,
        )
        a_data = pd.DataFrame(
            list(map(np.ravel, append_groups[1])),
            columns=const.AWAY_ANALYZER_FEATURES,
        )

        if not months_table:
            h_data["Net"] = h_data["Net"].astype(float)
            h_data = h_data.sort_values("Net", ascending=False).reset_index(drop=True)

            a_data["Net"] = a_data["Net"].astype(float)
            a_data = a_data.sort_values("Net", ascending=False).reset_index(drop=True)

        if months_table or rule_table:
            print(h_data)
        else:
            print(h_data)
            print(a_data)

        return h_data, a_data

    def __spread_status(self) -> None:
        """Adds status of Spread Outcome to Dataset"""

        for i in self.data.index:
            if self.data.at[i, "MOV"] > 0:
                if self.data.at[i, "MOV"] > self.data.at[i, "Spread"]:
                    self.data.at[i, "Spread_Status"] = 1
                else:
                    self.data.at[i, "Spread_Status"] = 0

            else:
                if abs(self.data.at[i, "MOV"]) < self.data.at[i, "Spread"]:
                    self.data.at[i, "Spread_Status"] = 1
                else:
                    self.data.at[i, "Spread_Status"] = 0
