"""
This script contains functions for updating daily tables/running daily predictions
"""
from datetime import date
import typing as t
import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from model import test_model
from ratings import current_massey
from scrape import collect_specific
from transform import combine_dailies
import utils

SCH_HEADER = {
    "user-agent": "Mozilla/5.0 (Windows NT 6.2; WOW64)"
    "Chrome/57.0.2987.133 Safari/537.36",
    "Accept-Language": "en",
    "origin": "http://stats.nba.com",
    "Referer": "https://google.com",
}

YEAR = "20" + date.today().strftime("%y")
URL = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{YEAR}/scores/00_todays_scores.json"  # pylint: disable=line-too-long

PARAMS = {
    "max_depth": 3,
    "min_child_weight": 5,
    "eta": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "num_class": 2,
}

NET_EPOCHS = 918
MASSEY_EPOCHS = 5000


def get_todays_games(url: str) -> t.Any:
    """
    Retrieves schedule json from URL
    Returns relevant game data from retrieved json
    """

    req = requests.get(url, headers=SCH_HEADER, timeout=60)
    data = req.json().get("gs").get("g")

    if not data:
        print("No games scheduled today.")
        sys.exit()

    return data


def update_team_stats() -> t.Any:
    """
    Retrieves the most update-to-date team stats from NBA API
    Combines basic and advanced stats, dropping duplicates.
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
        ["TEAM_NAME", "TEAM_ID", "CFID", "CFPARAMS", "W", "L", "W_PCT", "GP", "MIN"],
        axis=1,
        inplace=True,
    )

    team_stats = pd.concat([basic_stats, advanced_stats], axis=1, join="outer")
    team_stats["Conf"] = team_stats["Team"].map(utils.conf_dict)
    team_stats["Record"] = team_stats[["W", "L"]].apply(
        lambda row: "-".join(row.values.astype(str)), axis=1
    )
    team_stats.to_sql("team_stats", utils.ENGINE, if_exists="replace", index=False)
    team_stats = team_stats.groupby("Team")

    return team_stats


def update_massey():
    """
    Retrieves the most up-to-date game results
    Calculates up-to-date Massey Ratings
    Returns Massey Ratings grouped by Team Name
    """

    temp = collect_specific(2023, utils.MONTHS_REG)
    massey_ratings = current_massey(temp, "2022-23")

    return massey_ratings


def build_pred_data(
    data: t.Any, team_stats: t.Any, massey_ratings: t.Any
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


def split_pred_data(
    data: pd.DataFrame, features: list
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Adds placeholder outcomes for today's games
    Builds team data table from game data for condesenced display table
    Returns amended game data, placeholder outcomes, and team data
    """

    data["Outcome"] = 0
    outcome_placeholder = data["Outcome"]

    team_data = data[
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

    data = data[features + ["A_Massey", "H_Massey"]]

    return data, outcome_placeholder, team_data


def daily_pred(
    test_data: pd.DataFrame, test_outcomes: pd.Series, team_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Purpose
    ----------
    Loads the full training data
    Performs cross-validation (15) testing on training data
    Calls prediction function for Net Rating Focused Model
    Calls prediction function for Massey Rating Focused Model
    Commits predictions to running prediction histories (for each model)
    Commits today's prediction table to database

    Parameters
    ----------
    test_data:
        DataFrame containing today's games along with team stats/ratings

    test_outcomes:
        Placeholder outcomes to be filled with actuals

    team_data:
        Condensed DataFrame of today's games and stats used for display

    Returns
    ----------
    Returns today's prediction table
    """

    train_data = pd.read_sql_table("training_data", utils.ENGINE)
    mask = train_data["A_Massey"] != 0
    train_data = train_data.loc[mask].reset_index(drop=True)
    outcomes = train_data["Outcome"]

    net_data = train_data[utils.NET_FULL_FEATURES]
    massey_data = train_data[utils.MASSEY_FULL_FEATURES]

    test_model(net_data, outcomes, 2, NET_EPOCHS)

    final_net = predict_today(
        net_data, outcomes, test_data, test_outcomes, team_data, NET_EPOCHS
    )
    final_massey = predict_today(
        massey_data, outcomes, test_data, test_outcomes, team_data, MASSEY_EPOCHS
    )

    commit_history(final_net, utils.NET_FULL_FEATURES)
    commit_history(final_massey, utils.MASSEY_FULL_FEATURES)

    commit_preds(final_net)

    return final_net


def predict_today(
    train: pd.DataFrame,
    targets: pd.Series,
    test: pd.DataFrame,
    test_targets: pd.Series,
    teams: pd.DataFrame,
    epochs: int,
) -> pd.DataFrame:
    #  pylint: disable=too-many-arguments
    """
    Purpose
    ----------
    Builds DMatrixs for model testing
    x_matrix = training data and training targets
    y_matrix = testing data and testing targets

    Trains model on full dataset (x_matrix)
    Calls prediction function on test dataset (y_matrix)
    Combines model predictions with teams display data

    Parameters
    ----------
    train:
        DataFrame containing feature filtered training data

    targets:
        DataFrame containing training prediction targets (Outcome/Spread)

    test:
        DataFrame containing feature filtered testing data

    test_targets:
       DataFrame containing testing prediction targets (Outcome/Spread)

    teams:
        Condensed DataFrame of today's games and stats used for display

    epochs:
        Number of estimators/trees for the model to use

    Returns
    ----------
    Returns combined prediction table
    """

    full_preds = []

    x_matrix = xgb.DMatrix(train, label=targets)
    y_matrix = xgb.DMatrix(test[train.columns], label=test_targets)

    xgb_model = xgb.train(PARAMS, x_matrix, epochs)
    preds = xgb_model.predict(y_matrix)

    for pred in preds:
        full_preds.append(
            (round(pred[0], 3), round(pred[1], 3), f"{pred[0]:.3f} vs. {pred[1]:.3f}")
        )

    pred_data = pd.DataFrame(
        list(map(np.ravel, full_preds)), columns=["A_Odds", "H_Odds", "G_Odds"]
    )

    final_data = pd.concat([teams, pred_data], axis=1, join="outer")

    return final_data


def commit_history(data: pd.DataFrame, features: list) -> None:
    """
    Commits model predictions to database history tables
    Uses parameter features to determine which model is being committed
    """
    pred_history_update = data[["Date", "A_Team", "A_Odds", "H_Team", "H_Odds"]]

    if features == utils.NET_FULL_FEATURES:
        old_pred_history = pd.read_sql_table("prediction_history_net", utils.ENGINE)

        new_pred_history = pd.concat(
            [pred_history_update, old_pred_history], axis=0, join="outer"
        )

        new_pred_history.drop_duplicates(
            subset=["Date", "A_Team"], keep="first", inplace=True
        )

        new_pred_history.to_sql(
            "prediction_history_net", utils.ENGINE, if_exists="replace", index=False
        )

    elif features == utils.MASSEY_FULL_FEATURES:
        old_pred_history = pd.read_sql_table("prediction_history_massey", utils.ENGINE)

        new_pred_history = pd.concat(
            [pred_history_update, old_pred_history], axis=0, join="outer"
        )

        new_pred_history.drop_duplicates(
            subset=["Date", "A_Team"], keep="first", inplace=True
        )

        new_pred_history.to_sql(
            "prediction_history_massey", utils.ENGINE, if_exists="replace", index=False
        )


def commit_preds(data: pd.DataFrame) -> None:
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

    data.to_sql("today_preds", utils.ENGINE, if_exists="replace", index=False)
    print(data)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    games = get_todays_games(URL)
    daily_team_stats = update_team_stats()
    ratings = update_massey()

    raw_data, predictors = build_pred_data(games, daily_team_stats, ratings)
    todays_games, ph, team_names = split_pred_data(raw_data, predictors)
    predictions = daily_pred(todays_games, ph, team_names)
    combine_dailies()
