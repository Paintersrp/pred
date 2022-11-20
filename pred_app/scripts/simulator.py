"""
WIP
"""
import pickle
import pandas as pd
import numpy as np
from scripts import const
from scripts import predictor
from scripts import handler


class Simulator:
    """
    Class
    """

    def __init__(self, bet_value: float) -> None:
        if isinstance(bet_value, str) or bet_value <= 0:
            raise ValueError(
                f"Bet Value must not be negative, 0, or a string. Bet Value Received: {bet_value}"
            )

        self.data_handler = handler.MetricsHandler()
        self.bet_value = bet_value
        self.data = self.__prepare_data()

        self.value_won = None
        self.value_lost = None
        self.net_total = None
        self.ratio = None

    def __prepare_data(self) -> pd.DataFrame:
        """
        Func
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

        return combined

    def return_data(self) -> pd.DataFrame:
        """
        Func
        """

        return self.data

    def calculate(self) -> None:
        """
        Func
        """

        self.data["Value"] = np.where(
            self.data["Bet_Status"] == 1,
            (100 / abs(self.data["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        bets_won = self.data.loc[self.data["Bet_Status"] == 1].reset_index(drop=True)
        bets_lost = self.data.loc[self.data["Bet_Status"] == 0].reset_index(drop=True)

        self.value_won = sum(bets_won["Value"])
        self.value_lost = sum(bets_lost["Value"])
        self.net_total = self.value_won + self.value_lost
        self.ratio = round(self.value_won / abs(self.value_lost) * 100, 2)

    def return_totals(self) -> tuple[float, float, float, float]:
        """
        Returns
        ---------
        Value Won, Value Lost, Net Total, and Earned/Lost Ratio
        """

        return (
            round(self.value_won, 2),
            round(self.value_lost, 2),
            round(self.net_total, 2),
            str(self.ratio) + "%",
        )

    def best_wins(self) -> pd.DataFrame:
        """
        Returns
        --------
        The five most profitable wins in current simulation session data
        """

        return (
            self.data.sort_values("Value", ascending=False)
            .reset_index(drop=True)
            .head(5)
        )

    def worst_losses(self) -> pd.DataFrame:
        """
        The five games where the biggest upset occured in current simulation session data
        """

        loss_data = self.data.loc[self.data["Bet_Status"] == 0].reset_index(drop=True)

        return (
            loss_data.sort_values("ML_Payout", ascending=False)
            .reset_index(drop=True)
            .head(5)
        )

    def update_bet_value(self, bet_value) -> None:
        """
        Func
        """
        if isinstance(bet_value, str) or bet_value <= 0:
            raise ValueError(
                f"Bet Value must not be negative, 0, or a string. Bet Value Received: {bet_value}"
            )

        self.bet_value = bet_value

    def export_to_csv(self) -> None:
        """
        Func
        """

        #  maybe add return totals to csv before

        data = self.data
        data2 = pd.DataFrame(
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
            + list(self.return_totals())
        ).T

        data2.columns = data.columns
        data = pd.concat([data, data2], axis=0, join="outer").reset_index(drop=True)
        data.to_csv("SimResults.csv", index=None)

    def save_sim(self, file_name: str) -> None:
        """
        Func
        """

        with open(f"{file_name}.pkl", "wb") as writer:
            pickle.dump(self, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def load_sim(self, file_name: str) -> object:
        """
        Func
        """

        with open(f"{file_name}.pkl", "rb") as writer:
            data = pickle.load(writer)

        return data


class SimRandom(Simulator):
    """
    Class
    """

    def __init__(self, bet_value: float, win_rate: float, num_games: int) -> None:
        super().__init__(bet_value)
        self.__check_parameters(win_rate, num_games)

        self.win_rate = win_rate
        self.num_games = num_games
        self.sim_data = []
        self.reroll()

    def reroll(self) -> pd.DataFrame:
        """
        Func
        """

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

    def update_win_rate(self, win_rate) -> None:
        """
        Func
        """

        self.__check_parameters(win_rate, self.num_games)
        self.win_rate = win_rate

    def update_num_games(self, num_games) -> None:
        """
        Func
        """

        self.__check_parameters(self.win_rate, num_games)
        self.num_games = num_games

    def __check_parameters(self, win_rate: float, num_games: int):
        """
        Func
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
    Class
    """

    def __init__(self, bet_value: float, random: bool, year: int = 2022):
        super().__init__(bet_value)
        self.__check_parameters(random, year)

        self.year = year
        self.sim_predictor = predictor.Predictor()
        self.reroll(random, self.year)

    def reroll(self, random: bool, year: int = 2022) -> pd.DataFrame:
        """
        Func
        """

        self.__check_parameters(random, year)
        self.sim_predictor.enable_sim_mode()

        if random:
            self.predictions = self.sim_predictor.predict(True, True, None)
        else:
            self.predictions = self.sim_predictor.predict(False, True, year)

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
            self.test_data["Outcome"] == 1,
            (100 / abs(self.test_data["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        self.data = self.test_data

    def __check_parameters(self, random: bool, year: int):
        """
        Func
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


# bet skip = skip anything 45-55%
# lean in = add bet value for 85-100%
# lean out = lower bet value for 55-65%
class SimAdvanced(Simulator):
    #  pylint: disable=too-many-instance-attributes
    """
    Class
    """

    def __init__(
        self,
        bet_value: float,
        bet_skip: bool = True,
        lean_in: bool = True,
        lean_out: bool = True,
    ) -> None:
        super().__init__(bet_value)
        self.bet_skip = bet_skip
        self.lean_in = lean_in
        self.lean_out = lean_out

        self.bet_skip_min = 0.501
        self.bet_skip_max = 0.550
        self.lean_in_min = 0.850
        self.lean_in_max = 0.999
        self.lean_out_min = 0.551
        self.lean_out_max = 0.650

    def __prepare_data(self) -> pd.DataFrame:
        """
        Func
        """

        data_handler = handler.MetricsHandler()
        prep_data = data_handler.pred_history()

        prep_data = prep_data.rename(columns={"Outcome": "Bet_Status"})

        # load odds data, filter to dates
        # either take the bet_status from pred history and add to odds_data
        # or take ml_payout from odds data and add to pred history
        # adjust math on np wheres to correct

        prep_data["Rule"] = None

        prep_data["Rule"] = np.where(
            self.lean_in_min <= prep_data["A_Odds"] <= self.lean_in_min,
            "Lean-In",
            prep_data["Rule"],
        )

        prep_data["Rule"] = np.where(
            self.lean_out_min <= prep_data["A_Odds"] <= self.lean_out_min,
            "Lean-Out",
            prep_data["Rule"],
        )

        prep_data["Rule"] = np.where(
            self.bet_skip_min <= prep_data["A_Odds"] <= self.bet_skip_min,
            "Skip",
            prep_data["Rule"],
        )

        prep_data["Bet"] = self.bet_value

        prep_data["Bet"] = np.where(
            prep_data["Rule"] == "Lean-In", prep_data["Bet"] * 1.25, prep_data["Bet"]
        )

        prep_data["Bet"] = np.where(
            prep_data["Rule"] == "Lean-Out", prep_data["Bet"] * 0.75, prep_data["Bet"]
        )

        prep_data["Bet"] = np.where(prep_data["Rule"] == "Skip", 0.00, prep_data["Bet"])

        prep_data["Value"] = np.where(
            prep_data["Bet_Status"] == 1,
            prep_data["ML_Payout"] * prep_data["Bet"],
            -abs(prep_data["Bet"]),
        )

        return prep_data

    def adjust_bet_skip_range(self, minimum, maximum) -> None:
        """
        Func
        """

        self.bet_skip_min = minimum
        self.bet_skip_max = maximum
        # self.__prepare_data()

    def adjust_lean_in_range(self, minimum, maximum) -> None:
        """
        Func
        """

        self.lean_in_min = minimum
        self.lean_in_max = maximum
        # self.__prepare_data()

    def adjust_lean_out_range(self, minimum, maximum) -> None:
        """
        Func
        """

        self.lean_out_min = minimum
        self.lean_out_max = maximum
        # self.__prepare_data()

    def test_bet_skip(self) -> None:
        """
        Func
        """

        #   bet skip test - same as lean in, same calculations

        pass

    def test_lean_in(self) -> None:
        """
        Func
        """

        #   lean in filtered test - use handler to get all games with 85-100% for one team
        #       calculate win%, # of losses, earned/lost ratio, profit with, profit without, loss with, etc         #  pylint: disable=line-too-long

        pass

    def test_lean_out(self) -> None:
        """
        Func
        """

        #   lean out filtered test -  as lean in, same calculations

        pass

    def calculate(self) -> None:
        """
        Func
        """

        bets_won = self.data.loc[self.data["Bet_Status"] == 1].reset_index(drop=True)
        bets_lost = self.data.loc[self.data["Bet_Status"] == 0].reset_index(drop=True)

        # after move - can I just keep base calculate method?

        self.value_won = sum(bets_won["Value"])
        self.value_lost = sum(bets_lost["Value"])
        self.net_total = self.value_won - self.value_lost
        self.ratio = self.value_won / self.value_lost
