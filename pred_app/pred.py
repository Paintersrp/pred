"""
Docstring
"""
from datetime import date
import typing as t
import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from model import build_model
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
URL = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{YEAR}/scores/00_todays_scores.json"

PARAMS = {
    "max_depth": 4,
    "min_child_weight": 60,
    "eta": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "num_class": 2,
}

EPOCHS = 500


def update_team_stats() -> t.Any:
    """
    Funcstring
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

    advanced_stats.drop(["TEAM_ID", "CFID", "CFPARAMS"], axis=1, inplace=True)

    advanced_stats = advanced_stats.rename(
        columns={
            "TEAM_NAME": "Team Name",
            "OFF_RATING": "Off",
            "DEF_RATING": "Def",
            "NET_RATING": "Net",
        }
    )

    advanced_stats = advanced_stats[
        [
            "Off",
            "Def",
            "Net",
            "PIE",
            "TS_PCT",
            "EFG_PCT",
            "NET_RATING_RANK",
            "TS_PCT_RANK",
            "EFG_PCT_RANK",
            "PIE_RANK",
        ]
    ]

    team_stats = pd.concat([basic_stats, advanced_stats], axis=1, join="outer")
    team_stats["Conf"] = team_stats["Team"].map(utils.conf_dict)
    team_stats["Record"] = team_stats[["W", "L"]].apply(
        lambda row: "-".join(row.values.astype(str)), axis=1
    )
    team_stats.to_sql("team_stats", utils.engine, if_exists="replace", index=False)
    team_stats = team_stats.groupby("Team")

    return team_stats


def games_today(
    url: str, team_stats: t.Any
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Scrapes the NBA's schedule JSON for today's games
    Updates the most recent Team Stats and Massey Ratings
    Returns team names and test_data for DMatrix
    """

    req = requests.get(url, headers=SCH_HEADER, timeout=60)
    data = req.json().get("gs").get("g")

    if not data:
        print("No games scheduled today.")
        sys.exit()

    full_arrays = []
    ratings_array = []

    temp = collect_specific(2023, utils.months_reg)
    massey_ratings = current_massey(temp, "2022-23")

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
    game_data["Outcome"] = 0
    game_placeholder = game_data["Outcome"]

    team_data = game_data[
        [
            "Date",
            "A_Team",
            "H_Team",
            "A_Massey",
            "H_Massey",
            "A_W_PCT",
            "H_W_PCT",
            "A_Net",
            "H_Net",
            "Game Time",
        ]
    ]

    game_data = game_data[features + ["A_Massey", "H_Massey"]]

    return game_data, game_placeholder, team_data


def daily_pred(
    test_data: pd.DataFrame, outcomes: pd.Series, team_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Train/Score Model
    Prints teams and their respective odds for each of today's games
    """

    full_preds = []
    train_data = pd.read_csv("Train_Ready.csv")

    mask = train_data["A_Massey"] != 0
    train_data = train_data.loc[mask].reset_index(drop=True)
    outcome = train_data["Outcome"]

    train_data = train_data[
        [
            "A_NET_RATING",
            "H_NET_RATING",
            "A_Massey",
            "H_Massey",
            "A_PIE",
            "H_PIE",
            "A_TS_PCT",
            "H_TS_PCT",
            "A_DREB",
            "H_DREB",
        ]
    ]

    train_data.columns = [
        "A_Net",
        "H_Net",
        "A_Massey",
        "H_Massey",
        "A_PIE",
        "H_PIE",
        "A_TS_PCT",
        "H_TS_PCT",
        "A_DREB",
        "H_DREB",
    ]

    # train_data.drop(
    #     [
    #         "Outcome",
    #         "Date",
    #         "Home",
    #         "Away",
    #         "MOV",
    #     ],
    #     axis=1,
    #     inplace=True,
    # )

    # train_data = train_data[
    #     train_data.columns.drop(
    #         list(
    #            train_data.filter(
    #             regex=r"(_RANK|_E_|_PACE|_ELO|_DEF|_OFF|_PFD|_PF|_POSS|_MIN|_W|_L|_PLUS|_GP|_REB)"
    #            )
    #         )
    #     )
    # ]

    xgb_model = build_model(train_data, outcome, 10)

    x_matrix = xgb.DMatrix(train_data, label=outcome)
    y_matrix = xgb.DMatrix(test_data[train_data.columns], label=outcomes)

    xgb_model = xgb.train(PARAMS, x_matrix, EPOCHS)
    preds = xgb_model.predict(y_matrix)

    for pred in preds:
        full_preds.append(
            (round(pred[0], 3), round(pred[1], 3), f"{pred[0]:.3f} vs. {pred[1]:.3f}")
        )

    pred_data = pd.DataFrame(
        list(map(np.ravel, full_preds)), columns=["A_Odds", "H_Odds", "G_Odds"]
    )

    final_data = pd.concat([team_data, pred_data], axis=1, join="outer")
    pred_history_update = final_data[["Date", "A_Team", "A_Odds", "H_Team", "H_Odds"]]
    old_pred_history = pd.read_sql_table("prediction_history", utils.engine)

    new_pred_history = pd.concat(
        [pred_history_update, old_pred_history], axis=0, join="outer"
    )

    new_pred_history.drop_duplicates(
        subset=["Date", "A_Team"], keep="first", inplace=True
    )

    new_pred_history.to_sql(
        "prediction_history", utils.engine, if_exists="replace", index=False
    )

    final_data = final_data[
        [
            "A_W_PCT",
            "A_Net",
            "A_Massey",
            "A_Odds",
            "A_Team",
            "Game Time",
            "H_Team",
            "H_Odds",
            "H_Massey",
            "H_Net",
            "H_W_PCT",
        ]
    ]

    final_data["Game Time"] = final_data["Game Time"].str.split(" ").str[0]

    final_data.columns = [
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

    final_data.to_sql("today_preds", utils.engine, if_exists="replace", index=False)
    print(final_data)

    return final_data


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    daily_team_stats = update_team_stats()
    todays_games, ph, team_names = games_today(URL, daily_team_stats)
    combine_dailies()
    final = daily_pred(todays_games, ph, team_names)
