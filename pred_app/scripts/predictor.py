"""
This module contains ML Model Predictor Classes and Methods
"""
import pickle
from datetime import date, timedelta
import typing as t
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_recall_curve,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scripts import const, dicts, utils, handler


"""
mask = train_data["A_Massey"] != 0
train_data = train_data.loc[mask].reset_index(drop=True)
outcomes = train_data["Outcome"]
massey_data = train_data[const.MASSEY_FULL_FEATURES]
"""

DATAHANDLER = handler.GeneralHandler()


class Predictor:
    #  pylint: disable=too-many-instance-attributes
    """
    Contains methods for scoring and manipulating Prediction models
    """

    def __init__(self):
        self.train_data = DATAHANDLER.training_data_v2()
        self.outcomes = self.train_data["Outcome"]
        self.net_data = self.train_data[const.NET_FULL_FEATURES]
        self.model = None
        self.outcomes_arr = None
        self.outcome_placeholder = None
        self.y_test = None
        self.test_data = None
        self.unfiltered_test_data = None
        self.sim_mode = False

    def enable_sim_mode(self) -> None:
        """
        Sets the training data to data filtered to games with odds history attached
        """

        self.train_data = DATAHANDLER.pred_sim_data()
        self.outcomes = self.train_data["Outcome"]
        self.net_data = self.train_data
        self.net_data.drop(["Pred"], axis=1, inplace=True)
        self.sim_mode = True

    def test_model(
        self,
        cv_count: int = 5,
        epochs: int = 918,
        params: dict = dicts.PARAMS,
        loud: bool = False,
    ) -> list:
        # pylint: disable=too-many-locals disable=dangerous-default-value

        """
        Purpose
        ----------
        Builds DMatrixs for model testing

        x_matrix = training data and training targets

        y_matrix = testing data and testing targets

        Trains model on full dataset (x_matrix)

        Calls prediction function on test dataset (y_matrix)

        Scores metrics for model

        Repeats tests and training x num of times (cv_count)

        Parameters
        ----------
        cv_count:
            Integer denoting how many cross validation tests to run

        epochs:
            Number of estimators/trees for the model to use

        params:
            Parameter grid to set the XGB.Booster hyperparameters

        loud:
            Boolean condition - if true each test will print scoring metrics

        Returns
        ----------
        Returns list of model scoring metrics
        """

        metrics_list = []

        for i in range(cv_count):
            self.outcomes_arr = []

            x_train, x_test, y_train, self.y_test = train_test_split(
                self.net_data, self.outcomes, test_size=0.15
            )

            x_matrix = xgb.DMatrix(x_train, label=y_train)
            y_matrix = xgb.DMatrix(x_test, label=self.y_test)

            self.model = xgb.train(params, x_matrix, epochs)
            preds = self.model.predict(y_matrix)

            for pred in preds:
                self.outcomes_arr.append(np.argmax(pred))

            if loud:
                print(f"      {i+1} of {cv_count} Complete     ")

            arr = self.get_metrics(self.y_test, self.outcomes_arr, loud)
            metrics_list.append(arr)

        return metrics_list

    def get_metrics(self, actual: list, prediction: np.ndarray, loud: bool) -> list:
        """
        Uses actual outcomes and predicted outcomes to score test metrics
        """

        arr = []

        combined = pd.DataFrame(dict(actual=actual, prediction=prediction))
        crosstab = pd.crosstab(index=combined["actual"], columns=combined["prediction"])

        precision = round(precision_score(actual, prediction), 4) * 100
        accuracy = round(accuracy_score(actual, prediction), 4) * 100
        logloss = round(log_loss(actual, prediction), 4)
        roc = round(roc_auc_score(actual, prediction), 4) * 100
        recall = round(recall_score(actual, prediction), 4) * 100
        correct = int(crosstab[0][0]) + int(crosstab[1][1])
        incorrect = int(crosstab[0][1]) + int(crosstab[1][0])
        game_count = len(prediction)

        if loud:
            print(f"{correct} Correct - {incorrect} Incorrect")
            print(f"Precision: {round(precision,4)}%")
            print(f"Accuracy:  {round(accuracy,4)}%")
            print(f"Logloss:   {round(logloss,4)}%")
            print("-----------------------------")

        arr.extend(
            [precision, recall, accuracy, logloss, roc, correct, incorrect, game_count]
        )

        return arr

    def predict(
        self,
        random: bool = True,
        loud: bool = False,
        year: int = 2022,
    ) -> pd.DataFrame:
        """
        Purpose
        ----------
        General prediction function, split by year or randomly

        Builds DMatrixs for model testing

        x_matrix = training data and training targets

        y_matrix = testing data and testing targets

        Trains model on filtered dataset (x_matrix)

        Calls prediction function on test dataset (y_matrix)

        Parameters
        ----------
        random:
            Boolean condition - if true data is randomly split and tested

        year:
            If not random, split and test data on a given year/season

        Returns
        ----------
        DataFrame of Predictions
        """

        if random:
            (
                self.net_data,
                self.test_data,
                self.outcomes,
                self.outcome_placeholder,
            ) = train_test_split(self.net_data, self.outcomes, test_size=0.15)
        else:
            season = utils.map_season(year)
            season_mask = self.train_data["SeasonID"] == season
            self.test_data = self.train_data[season_mask]
            self.net_data = self.train_data.drop(
                self.train_data[season_mask].index, axis=0
            )
            self.outcomes = self.net_data["Outcome"]
            self.outcome_placeholder = self.test_data["Outcome"]

        if self.sim_mode:
            self.unfiltered_test_data = self.test_data
            self.test_data = self.test_data[const.SIM_PRED_FEATURES]
            self.net_data = self.net_data[const.SIM_PRED_FEATURES]

        self.outcomes_arr = []

        x_matrix = xgb.DMatrix(
            self.net_data[const.NET_FULL_FEATURES], label=self.outcomes
        )
        y_matrix = xgb.DMatrix(
            self.test_data[const.NET_FULL_FEATURES], label=self.outcome_placeholder
        )

        xgb_model = xgb.train(dicts.PARAMS, x_matrix, const.NET_EPOCHS)
        preds = xgb_model.predict(y_matrix)

        for pred in preds:
            self.outcomes_arr.append(np.argmax(pred))

        self.get_metrics(self.outcome_placeholder, self.outcomes_arr, loud)

        return self.outcomes_arr, preds

    def feature_scoring(self) -> pd.DataFrame:
        """
        Test Feature Importances and returns DataFrame of scores
        """

        best = SelectKBest(score_func=f_classif, k="all")

        # if self.net_data["Date"]:
        #     data = self.net_data.drop(["Date"], axis=1, inplace=True)

        fit = best.fit(self.net_data[const.NET_FULL_FEATURES], self.outcomes)

        temp = pd.DataFrame(fit.scores_)
        columns = pd.DataFrame(self.net_data[const.NET_FULL_FEATURES].columns)

        scores = pd.concat([temp, columns], axis=1)
        scores.columns = ["Specs", "Score"]
        scores = scores.sort_values(["Specs"], ascending=False).reset_index(drop=True)

        return scores

    def hyperparameter_tuning(self, loud: bool = False) -> pd.DataFrame:
        """
        Tests hyperparameters based on a narrow grid of options randomly, scoring each

        Prints breakdown of hyperparameter scoring
        """

        parameters = ["param_" + params for params in dicts.xgb_narrow_grid] + [
            "mean_test_score"
        ]

        rs_model = RandomizedSearchCV(
            estimator=const.DEF_CLASSIFIER,
            param_distributions=dicts.xgb_narrow_grid,
            cv=3,
            verbose=10,
            n_jobs=-1,
            return_train_score=False,
            n_iter=128,
        )

        rs_model.fit(self.net_data, self.outcomes)
        hypers = pd.DataFrame(rs_model.cv_results_)[parameters]
        hypers = hypers.sort_values(by=["mean_test_score"], ascending=False)

        parameters.remove("mean_test_score")

        if loud:
            for parameter in parameters:
                temp1 = (
                    hypers.groupby(parameter)["mean_test_score"]
                    .agg(np.mean)
                    .reset_index()
                )
                temp1 = temp1.sort_values(by=["mean_test_score"], ascending=False)
                print(temp1)

        return hypers

    def find_trees(
        self,
    ) -> int:
        """
        Finds the ideal number of trees/estimators for the model
        """

        x_matrix = xgb.DMatrix(self.net_data, label=self.outcomes)

        cv_result = xgb.cv(
            dicts.TREE_TESTING,
            x_matrix,
            num_boost_round=10000,
            nfold=5,
            metrics="auc",
            early_stopping_rounds=50,
        )

        trees = cv_result.shape[0]
        print(trees)

        return trees

    def plot_roc_curve(self) -> None:
        """
        Plots and saves a ROC-AUC plot image
        """

        fpr, tpr, dummy = roc_curve(self.y_test, self.outcomes_arr)
        plt.plot(fpr, tpr, lw=2, color="royalblue", marker=".", label="PRED")
        plt.plot([0, 1], [0, 1], "--", color="firebrick", label="Baseline")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title("ROC-AUC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.gcf().savefig("ROC_AUC_Curve.png", dpi=1200)
        plt.clf()

    def plot_precision_recall(self) -> None:
        """
        Plots and saves a Precision-Recall plot image
        """

        precision, recall, dummy = precision_recall_curve(
            self.y_test, self.outcomes_arr
        )

        _, axis = plt.subplots()
        axis.plot(recall, precision, color="royalblue", label="PRED")
        axis.plot(
            [0, 1], [0.01, 0.01], color="firebrick", linestyle="--", label="Baseline"
        )
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        axis.set_title("Precision-Recall Curve")
        axis.set_ylabel("Precision")
        axis.set_xlabel("Recall")
        plt.legend(loc="best")
        plt.gcf().savefig("Precision_Recall_Curve.png", dpi=1200)

    def save_model(self, model: xgb.Booster) -> None:
        """
        Saves current model
        """

        with open(f"{const.FILE_NAME}", "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self) -> xgb.Booster:
        """
        Loads saved model
        """

        with open(f"{const.FILE_NAME}", "rb") as handle:
            self.model = pickle.load(handle)


class DailyPredictor(Predictor):
    """
    Daily Predictor Class

    Contains methods for predicting today's games
    """

    def __init__(self):
        super().__init__()
        self.game_data = None
        self.team_data = None
        self.features = None

    def build_test_data(
        self,
        team_stats: t.Any,
        massey_ratings: t.Any,
        elo_ratings: dict,
    ) -> None:
        #  pylint: disable=too-many-locals

        """
        Purpose
        ------------
        Parses general game data from schedule json

        Adds team stats and massey ratings to game data

        Parameters
        -----------
        data:
            Schedule data JSON

        team_stats:
            Up-to-date team stats, grouped by team name

        massey_ratings:
            Up-to-date massey ratings, grouped by team name

        Returns
        ------------
        Game data and team stats to be featured
        """

        full_arrays = []
        ratings_array = []
        data = pd.read_sql_table(f"2023_upcoming_games", const.ENGINE)
        data["Date"] = pd.to_datetime(data["Date"]).dt.date

        data["Away"] = np.where(
            data["Away"] == "Los Angeles Clippers", "LA Clippers", data["Away"]
        )

        data["Home"] = np.where(
            data["Home"] == "Los Angeles Clippers", "LA Clippers", data["Home"]
        )

        for matchup in data.itertuples():
            massey = []
            game_date = matchup.Date
            game_time = matchup.Time

            home_team = matchup.Home
            away_team = matchup.Away
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

            home_elo = elo_ratings[home_team]
            away_elo = elo_ratings[away_team]

            game = np.concatenate((away_stats, home_stats), axis=0)
            full_arrays.append(game)

            massey.extend(
                [
                    game_date,
                    round(float(away_massey), 2),
                    round(float(home_massey), 2),
                    game_time,
                    away_elo,
                    home_elo,
                ]
            )

            ratings_array.append(massey)

        self.features = ["A_" + col for col in away_stats.columns] + [
            "H_" + col for col in home_stats.columns
        ]

        game_data = pd.DataFrame(
            list(map(np.ravel, full_arrays)), columns=self.features
        )

        rating_data = pd.DataFrame(
            ratings_array,
            columns=["Date", "A_Massey", "H_Massey", "Game Time", "A_ELO", "H_ELO"],
        )

        self.game_data = pd.concat([game_data, rating_data], axis=1, join="outer")

    def prepare_test_data(
        self,
    ) -> None:
        """
        Adds placeholder outcomes for today's games

        Builds team data table from game data for condesenced display table

        Returns amended game data, placeholder outcomes, and team data
        """

        self.game_data["Outcome"] = 0
        self.outcome_placeholder = self.game_data["Outcome"]

        self.team_data = self.game_data[
            [
                "Date",
                "A_Team",
                "H_Team",
                "A_Massey",
                "H_Massey",
                "A_W_PCT",
                "H_W_PCT",
                "A_NET_RATING",
                "H_NET_RATING",
                "Game Time",
            ]
        ]

        self.test_data = self.game_data[
            self.features + ["A_Massey", "H_Massey", "H_ELO", "A_ELO"]
        ]

    def predict(
        self,
    ) -> pd.DataFrame:

        """
        Purpose
        ----------
        Builds DMatrixs for model testing

        x_matrix = training data and training targets

        y_matrix = testing data and testing targets

        Trains model on full dataset (x_matrix)

        Calls prediction function on test dataset (y_matrix)

        Combines model predictions with teams display data

        Returns
        ----------
        Returns combined prediction table
        """

        full_preds = []

        x_matrix = xgb.DMatrix(self.net_data, label=self.outcomes)
        y_matrix = xgb.DMatrix(
            self.test_data[self.net_data.columns], label=self.outcome_placeholder
        )

        xgb_model = xgb.train(dicts.PARAMS, x_matrix, const.NET_EPOCHS)
        preds = xgb_model.predict(y_matrix)

        for pred in preds:
            full_preds.append(
                (
                    round(pred[0], 3),
                    round(pred[1], 3),
                    f"{pred[0]:.3f} vs. {pred[1]:.3f}",
                )
            )

        pred_data = pd.DataFrame(
            list(map(np.ravel, full_preds)), columns=["A_Odds", "H_Odds", "G_Odds"]
        )

        final_data = pd.concat([self.team_data, pred_data], axis=1, join="outer")

        final_data.to_sql(
            "upcoming_predictions", const.ENGINE, if_exists="replace", index=False
        )

        today_games = final_data.loc[(final_data["Date"] == date.today())].reset_index(
            drop=True
        )
        tomorrow_games = final_data.loc[
            (final_data["Date"] == date.today() + timedelta(days=1))
        ].reset_index(drop=True)

        final_data["Date"] = final_data["Date"].astype(str)
        final_data.to_json(
            "C:/Python/pred_app/pred_react_v2/public/data/upcoming.json",
            orient="records",
        )

        return today_games, tomorrow_games
