"""
WIP
"""
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
        # bet value should be positive
        # win_rate should be scaled 1-100

        self.Handler = handler.MetricsHandler()
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

        pred_history_data = self.Handler.return_pred_history()
        current_data = self.Handler.return_current_odds_history()

        pred_history_data = pred_history_data.rename(columns={"Outcome": "Bet_Status"})
        # pred_list = pred_history_data["Date"].unique()
        # cur_list = current_data["Date"].unique()

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
        self.ratio = self.value_won / self.value_lost

    def return_totals(self) -> tuple[float, float, float, float]:
        """
        Returns
        ---------
        Value Won, Value Lost, Net Total, and Earned/Lost Ratio
        """

        return self.value_won, self.value_lost, self.net_total, self.ratio

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

        self.bet_value = bet_value

    def export_to_csv(self) -> None:
        """
        Func
        """

        #  maybe add return totals to csv before sending
        data = self.data
        data2 = pd.DataFrame(
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
            + list(self.return_totals())
        ).T
        data2.columns = data.columns
        data = pd.concat([data, data2], axis=0, join="outer").reset_index(drop=True)
        data.to_csv("SimulatorResults.csv", index=None)


class SimRandom(Simulator):
    """
    Class
    """

    def __init__(self, bet_value, win_rate, num_games) -> None:
        super().__init__(bet_value)
        self.win_rate = win_rate
        self.num_games = num_games
        self.data = self.reroll()

    def reroll(self) -> pd.DataFrame:
        """
        Func
        """

        sim_data = self.Handler.return_sim_data()
        print(sim_data)
        sim_data = sim_data.sample(self.num_games).reset_index(drop=True)
        faux_outcomes = np.random.binomial(1, self.win_rate, sim_data.shape[0])
        sim_data["Bet_Status"] = faux_outcomes
        sim_data["A_ML"] = sim_data["A_ML"].replace("+", "")
        sim_data["H_ML"] = sim_data["H_ML"].replace("+", "")
        sim_data["A_ML"] = np.where(sim_data["A_ML"] == "NL", 0, sim_data["A_ML"])
        sim_data["H_ML"] = np.where(sim_data["H_ML"] == "NL", 0, sim_data["H_ML"])
        sim_data["A_ML"] = sim_data["A_ML"].astype(float)
        sim_data["H_ML"] = sim_data["H_ML"].astype(float)

        # determine winner (based on mov)
        # return their ml odds
        # simdata = features -> [Date, Team Names, Pts, MOV, ML Odds]

        sim_data["ML_Payout"] = np.where(
            sim_data["MOV"] < 0, sim_data["A_ML"], sim_data["H_ML"]
        )

        sim_data["Value"] = np.where(
            sim_data["Bet_Status"] == 1,
            (100 / abs(sim_data["ML_Payout"])) * self.bet_value,
            -abs(self.bet_value),
        )

        print(sim_data)

        return sim_data

    def update_win_rate(self, win_rate) -> None:
        """
        Func
        """

        self.win_rate = win_rate

    def update_num_games(self, num_games) -> None:
        """
        Func
        """

        self.num_games = num_games


class SimPredictor(Simulator):
    """
    Class
    """

    def __init__(self, bet_value, random, year=2022):
        super().__init__(bet_value)
        self.year = year

        if random:
            self.data = self.reroll()

        if not random:
            self.data = self.update_season(self.year)

    def reroll(self) -> pd.DataFrame:
        """
        Func
        """

        Predictor = predictor.Predictor()
        self.predictions = Predictor.predict(True, True, None)
        self.test_data = Predictor.test_data
        self.test_data["Pred."] = self.predictions
        self.test_data = (
            self.test_data[const.SIM_FEATURES]
            .sort_values("Date", ascending=True)
            .reset_index(drop=True)
        )
        self.test_data = self.test_data.rename(columns={"Outcome": "Bet_Status"})

        return self.test_data

    def update_season(self, year) -> pd.DataFrame:
        """
        Func
        """

        Predictor = predictor.Predictor()
        self.predictions = Predictor.predict(False, True, year)
        self.test_data = Predictor.test_data
        self.test_data["Pred."] = self.predictions
        self.test_data = (
            self.test_data[const.SIM_FEATURES]
            .sort_values("Date", ascending=True)
            .reset_index(drop=True)
        )
        self.test_data = self.test_data.rename(columns={"Outcome": "Bet_Status"})

        return self.test_data


# bet skip = skip anything 45-55%
# lean in = add bet value for 85-100%
# lean out = lower bet value for 55-65%


class SimAdvanced(Simulator):
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
        self.lean_in_max = 1.000
        self.lean_out_min = 0.551
        self.lean_out_max = 0.650

    def __prepare_data(self) -> pd.DataFrame:
        """
        Func
        """

        Handler = handler.MetricsHandler()
        prep_data = Handler.return_pred_history()

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

    def adjust_bet_skip_range(self, min, max) -> None:
        """
        Func
        """

        self.bet_skip_min = min
        self.bet_skip_max = max
        # self.__prepare_data()

    def adjust_lean_in_range(self, min, max) -> None:
        """
        Func
        """

        self.lean_in_min = min
        self.lean_in_max = max
        # self.__prepare_data()

    def adjust_lean_out_range(self, min, max) -> None:
        """
        Func
        """

        self.lean_out_min = min
        self.lean_out_max = max
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
        #       calculate win%, # of losses, earned/lost ratio, profit with, profit without, loss with, etc

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
