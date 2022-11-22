"""
This module contains Betting Simulator Classes and Methods
"""
import pickle
import pandas as pd
import numpy as np
from scripts import const
from scripts import predictor
from scripts import handler


class Simulator:
    """
    Contains methods for simulating betting under various conditions

    Base Model runs a test of betting value outcomes for the current season's predictions
    """

    def __init__(self, bet_value: float) -> None:
        if isinstance(bet_value, str) or bet_value <= 0:
            raise ValueError(
                f"Bet Value must not be negative, 0, or a string. Bet Value Received: {bet_value}"
            )

        self.data_handler = handler.MetricsHandler()
        self.bet_value = bet_value
        self.value_won = None
        self.value_lost = None
        self.net_total = None
        self.ratio = None
        self.__prepare_data()

    def __prepare_data(self) -> pd.DataFrame:
        """
        Prepares dataset for by matching game odds to predicted games

        Adds column for Payout Rate and Payout Value
        """

        pred_history_data = self.data_handler.pred_history()
        current_data = self.data_handler.current_odds_history()
        pred_history_data = pred_history_data.rename(columns={"Outcome": "Bet_Status"})

        filter_check = list(current_data["Date"].unique())

        for game_date in current_data["Date"].unique():
            if game_date in pred_history_data["Date"].unique():
                filter_check.remove(game_date)

        for game_date in filter_check:
            current_data = current_data.loc[current_data["Date"] != game_date]

        filter_check = list(pred_history_data["Date"].unique())

        for game_date in pred_history_data["Date"].unique():
            if game_date in current_data["Date"].unique():
                filter_check.remove(game_date)

        for game_date in filter_check:
            pred_history_data = pred_history_data.loc[
                pred_history_data["Date"] != game_date
            ]

        current_data.sort_values(["Date", "H_Team"], ascending=False, inplace=True)
        current_data.reset_index(drop=True, inplace=True)
        pred_history_data.sort_values(["Date", "H_Team"], ascending=False, inplace=True)
        pred_history_data.reset_index(drop=True, inplace=True)
        current_data = current_data[["O/U", "H_ML", "A_ML"]]

        combined = pd.concat([pred_history_data, current_data], axis=1, join="outer")

        combined["ML_Payout"] = np.where(
            combined["Actual"] < 0, combined["A_ML"], combined["H_ML"]
        )

        combined["Value"] = np.where(
            combined["Bet_Status"] == 1,
            (100 / abs(combined["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        combined["Value"] = combined["Value"].astype(float)

        self.data = combined.copy()

    def return_data(self) -> pd.DataFrame:
        """
        Returns current dataset
        """

        return self.data

    def calculate(self) -> None:
        """
        Calculates totals for current parameters on the dataset
        """

        bets_won = (
            self.data.loc[self.data["Bet_Status"] == 1].reset_index(drop=True).copy()
        )
        bets_lost = (
            self.data.loc[self.data["Bet_Status"] == 0].reset_index(drop=True).copy()
        )

        self.value_won = sum(bets_won["Value"])
        self.value_lost = sum(bets_lost["Value"])
        self.net_total = self.value_won - abs(self.value_lost)
        self.ratio = round(self.value_won / abs(self.value_lost) * 100, 2)

    def return_totals(self) -> pd.DataFrame:
        """
        Returns
        ---------
        Value Won, Value Lost, Net Total, and Earned/Lost Ratio
        """

        totals = pd.DataFrame(
            [
                round(self.value_won, 2),
                round(self.value_lost, 2),
                round(self.net_total, 2),
                str(self.ratio) + "%",
            ]
        ).T

        totals.columns = ["Earned", "Lost", "Net", "Ratio"]

        return totals

    def best_wins(self) -> pd.DataFrame:
        """
        Returns
        --------
        The five most profitable wins in current simulation session data
        """

        return (
            self.data.sort_values("Value", ascending=False)
            .reset_index(drop=True)
            .head(25)
        )

    def worst_losses(self) -> pd.DataFrame:
        """
        The five games where the biggest upset occured in current simulation session data
        """

        loss_data = self.data.loc[self.data["Bet_Status"] == 0].reset_index(drop=True)

        return (
            loss_data.sort_values("ML_Payout", ascending=False)
            .reset_index(drop=True)
            .head(25)
        )

    def update_bet_value(self, bet_value) -> None:
        """
        Updates parameter bet value and prepares new dataset
        """

        if isinstance(bet_value, str) or bet_value <= 0:
            raise ValueError(
                f"Bet Value must not be negative, 0, or a string. Bet Value Received: {bet_value}"
            )

        self.bet_value = bet_value

        if isinstance(self, SimAdvanced):
            self.adjust_data()
        else:
            self.__prepare_data()

    def to_csv(self, file_name) -> None:
        """
        Save current dataset to CSV file
        """

        totals = self.return_totals()

        final_data = pd.concat([totals, self.data], axis=1, join="outer").reset_index(
            drop=True
        )

        final_data.to_csv(f"{file_name}.csv", index=None)

    def save_sim(self, file_name: str) -> None:
        """
        Saves current simulator object to disk
        """

        with open(f"{file_name}.pkl", "wb") as writer:
            pickle.dump(self, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def load_sim(self, file_name: str) -> object:
        """
        Loads simulator object from disk
        """

        with open(f"{file_name}.pkl", "rb") as writer:
            data = pickle.load(writer)

        return data


class SimRandom(Simulator):
    """
    Contains methods for simulating betting under randomized conditions

    Bet wins are assigned at random to a given win rate over a given number of games
    """

    def __init__(self, bet_value: float, win_rate: float, num_games: int) -> None:
        super().__init__(bet_value)
        self.__check_parameters(win_rate, num_games)

        self.win_rate = win_rate
        self.num_games = num_games
        self.sim_data = []
        self.reroll(win_rate, num_games)

    def reroll(self, win_rate: float, num_games: float) -> None:
        """
        Performs another randomized simulator at given parameters
        """

        self.__check_parameters(win_rate, num_games)
        self.win_rate = win_rate
        self.num_games = num_games

        sim_data = self.data_handler.general_sim_data()
        sim_data = sim_data.sample(self.num_games).reset_index(drop=True)
        faux_outcomes = np.random.binomial(1, self.win_rate, sim_data.shape[0])
        sim_data["Bet_Status"] = faux_outcomes
        sim_data["A_ML"] = sim_data["A_ML"].replace("+", "")
        sim_data["H_ML"] = sim_data["H_ML"].replace("+", "")
        sim_data["A_ML"] = np.where(sim_data["A_ML"] == "NL", 0, sim_data["A_ML"])
        sim_data["H_ML"] = np.where(sim_data["H_ML"] == "NL", 0, sim_data["H_ML"])
        sim_data["A_ML"] = sim_data["A_ML"].astype(float)
        sim_data["H_ML"] = sim_data["H_ML"].astype(float)

        sim_data["ML_Payout"] = np.where(
            sim_data["MOV"] < 0, sim_data["A_ML"], sim_data["H_ML"]
        )

        sim_data["Value"] = np.where(
            sim_data["Bet_Status"] == 1,
            (100 / abs(sim_data["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        self.data = sim_data

    def __check_parameters(self, win_rate: float, num_games: int) -> None:
        """
        Parameter error handling
        """

        if isinstance(win_rate, str) or win_rate <= 0:
            raise ValueError(
                f"Win Rate must not be negative, 0, or a string. Win Rate Received: {win_rate}"
            )

        if win_rate >= 1:
            raise ValueError(
                f"Win Rate must be between 0.01 and 0.99. Win Rate Received: {win_rate}"
            )

        if isinstance(num_games, str) or num_games <= 0:
            raise ValueError(
                f"Number of Games must not be negative, 0, or a string. Number of Games Received: {num_games}"  #  pylint: disable=line-too-long
            )

        if isinstance(num_games, float):
            raise ValueError(
                f"Number of Games must be a whole number. Number of Games Received: {num_games}"
            )


class SimPredictor(Simulator):
    """
    Contains methods for simulating betting with model predictions

    Bet wins are assigned by comparing model prediction to actual result
    """

    def __init__(self, bet_value: float, random: bool, year: int = 2022):
        super().__init__(bet_value)
        self.__check_parameters(random, year)

        self.year = year
        self.sim_predictor = predictor.Predictor()
        self.reroll(random, self.year)

    def reroll(self, random: bool, year: int = 2022) -> pd.DataFrame:
        """
        Builds and predicts a new dataset based on given parameters

        Adds Prediction, ML_Payout (Payout Rate), and Value (Bet Return/Loss Value)
        """

        self.__check_parameters(random, year)
        self.sim_predictor.enable_sim_mode()

        if random:
            self.predictions = self.sim_predictor.predict(True, False)
        else:
            self.predictions = self.sim_predictor.predict(False, False, year)

        self.predictions = pd.DataFrame(self.predictions, columns=["Pred"])
        self.test_data = self.sim_predictor.unfiltered_test_data.reset_index(drop=True)
        self.test_data["Pred."] = self.predictions
        self.test_data = pd.concat(
            [self.test_data, self.predictions], axis=1, join="outer"
        )

        self.test_data = (
            self.test_data[const.SIM_PRED_DISPLAY_FEATURES]
            .sort_values("Date", ascending=True)
            .reset_index(drop=True)
        )

        self.test_data["A_ML"] = np.where(
            self.test_data["A_ML"] == "NL", 0, self.test_data["A_ML"]
        )
        self.test_data["H_ML"] = np.where(
            self.test_data["H_ML"] == "NL", 0, self.test_data["H_ML"]
        )
        self.test_data["A_ML"] = self.test_data["A_ML"].astype(float)
        self.test_data["H_ML"] = self.test_data["H_ML"].astype(float)
        mask = (self.test_data["A_ML"] != 0) | (self.test_data["H_ML"] != 0)
        self.test_data = self.test_data.loc[mask].reset_index(drop=True)

        self.test_data["ML_Payout"] = np.where(
            self.test_data["MOV"] < 0, self.test_data["A_ML"], self.test_data["H_ML"]
        )

        self.test_data["Bet_Status"] = np.where(
            self.test_data["Outcome"] == self.test_data["Pred"], 1, 0
        )

        self.test_data["Value"] = np.where(
            self.test_data["Bet_Status"] == 1,
            (100 / abs(self.test_data["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        self.data = self.test_data.copy()

    def __check_parameters(self, random: bool, year: int):
        """
        Parameter error handling
        """

        if isinstance(random, str):
            raise ValueError(
                f"Random state should not be a string. Type Received: {type(random)}"
            )

        if not isinstance(random, bool):
            raise ValueError(
                f"Random state must be True or False. State Received: {random}"
            )

        if isinstance(year, str):
            raise ValueError(
                f"Year should not be a string. Type Received: {type(year)}"
            )

        if isinstance(year, str) or year <= 0:
            raise ValueError(
                f"Year must not be negative, 0, or a string. Year Received: {year}"
            )

        if isinstance(year, float) == float:
            raise ValueError(f"Year must be a whole number. Year Received: {year}")


class SimAdvanced(SimPredictor):
    #  pylint: disable=too-many-arguments
    """
    Contains methods for simulating betting with model predictions

    Bet wins are assigned by comparing model prediction to actual result

    Betting logic is adjusted based on rule enabling
    """

    def __init__(
        self,
        bet_value: float,
        random: bool = True,
        year: int = 2022,
        lean_in: bool = True,
        lean_out: bool = True,
    ) -> None:
        super().__init__(bet_value, random, year)
        self.__check_parameters(lean_in, lean_out)

        self.lean_in = lean_in
        self.lean_out = lean_out

        self.lean_out_min = 200
        self.lean_out_max = 10000
        self.lean_in_min = 110
        self.lean_in_max = 150

        self.__score_unadjusted()
        self.adjust_data()

    def adjust_rule_parameters(
        self,
        lean_in_minimum: int,
        lean_in_maximum: int,
        lean_out_minimum: int,
        lean_out_maximum: int,
    ) -> None:
        """
        Changes rule parameters based on input and adjusts the data to new parameters
        """

        self.__check_rule_parameters(
            lean_in_minimum, lean_in_maximum, lean_out_minimum, lean_out_maximum
        )

        self.lean_in_min = lean_in_minimum
        self.lean_in_max = lean_in_maximum
        self.lean_out_min = lean_out_minimum
        self.lean_out_max = lean_out_maximum
        self.adjust_data()

    def return_unadjusted(self) -> tuple[float, float, float, float]:
        """Returns score of dataset before rule adjustments"""

        return self.unadjusted_score

    def adjust_data(self) -> None:
        """
        Prepares dataset by applying rule adjustments based on Class parameters
        """

        rule_col = []
        bet_col = []

        for i in self.data.index:
            if self.lean_out_min <= abs(self.data.at[i, "A_ML"]) <= self.lean_out_max:
                if self.lean_out:
                    rule_col.append("Lean-Out")
                    bet_col.append(self.bet_value * 0.01)
                else:
                    rule_col.append("Lean-Out")
                    bet_col.append(self.bet_value)

            elif self.lean_in_min <= abs(self.data.at[i, "A_ML"]) <= self.lean_in_max:
                if self.lean_in:
                    rule_col.append("Lean-In")
                    bet_col.append(self.bet_value * 2)
                else:
                    rule_col.append("Lean-In")
                    bet_col.append(self.bet_value)
            else:
                rule_col.append("None")
                bet_col.append(self.bet_value)

        self.data["Rule"] = rule_col
        self.data["Bet"] = bet_col

        self.data["Value"] = np.where(
            self.data["Bet_Status"] == 1,
            (100 / abs(self.data["ML_Payout"])) * self.data["Bet"],
            -abs(self.data["Bet"]),
        )

    def toggle_rules(self, lean_in: bool, lean_out: bool) -> None:
        """Enables/Disables rule logic based on input"""

        self.__check_parameters(lean_in, lean_out)
        self.lean_in = lean_in
        self.lean_out = lean_out
        self.adjust_data()

    def __score_unadjusted(self) -> None:
        """Saves score before adjustments for comparison"""

        self.calculate()
        self.unadjusted_score = self.return_totals()

    def __check_parameters(self, lean_in: bool, lean_out: bool) -> None:
        """
        Parameter error handling
        """

        if isinstance(lean_in, str):
            raise ValueError(
                f"Lean In must be True or False, not a string. Type Received: {type(lean_in)}"
            )

        if not isinstance(lean_in, bool):
            raise ValueError(
                f"Lean In must be True or False. Value Received: {lean_in}"
            )

        if isinstance(lean_out, str):
            raise ValueError(
                f"Lean Out must be True or False, not a string. Type Received: {type(lean_out)}"
            )

        if not isinstance(lean_out, bool):
            raise ValueError(
                f"Lean Out must be True or False. Value Received: {lean_out}"
            )

    def __check_rule_parameters(
        self,
        lean_in_minimum: int,
        lean_in_maximum: int,
        lean_out_minimum: int,
        lean_out_maximum: int,
    ) -> None:
        """
        Rule parameter error handling
        """

        if not isinstance(lean_in_minimum, (int, float)):
            raise ValueError(
                f"Lean In Minimum should be numeric and not a string. Type Received: {type(lean_in_minimum)}"  #  pylint: disable=line-too-long
            )

        if not isinstance(lean_in_maximum, (int, float)):
            raise ValueError(
                f"Lean In Maximum should be numeric and not a string. Type Received: {type(lean_in_maximum)}"  #  pylint: disable=line-too-long
            )

        if not isinstance(lean_out_minimum, (int, float)):
            raise ValueError(
                f"Lean Out Minimum should be numeric and not a string. Type Received: {type(lean_out_minimum)}"  #  pylint: disable=line-too-long
            )

        if not isinstance(lean_out_maximum, (int, float)):
            raise ValueError(
                f"Lean Out Maximum should be numeric and not a string. Type Received: {type(lean_out_maximum)}"  #  pylint: disable=line-too-long
            )
