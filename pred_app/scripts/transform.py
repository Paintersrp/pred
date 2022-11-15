"""
This script contains data cleaning/transforming/amending functions
"""
import pandas as pd
import numpy as np
from datetime import date
from scripts import utils, const
from scripts.ratings import add_massey, add_elo


def combine_datasets() -> None:
    """
    Combines hard-copy seasons of training data into one
    ---------
    Included in the backups folder, or can be scraped.
    """
    sets = (
        pd.read_csv("TrainDataRawAdv_2005-09.csv"),
        pd.read_csv("TrainDataRawAdv_2010-14.csv"),
        pd.read_csv("TrainDataRawAdv_2015-18.csv"),
        pd.read_csv("TrainDataRawAdv_2019-22.csv"),
    )

    final = pd.concat(sets, axis=0, join="outer").reset_index(drop=True)
    final.drop(final[["A_TEAM_NAME", "H_TEAM_NAME"]], axis=1, inplace=True)
    final.to_sql("raw_data", const.ENGINE, if_exists="replace", index=False)


def combine_odds_dataset() -> pd.DataFrame:
    """
    Combines hard-copy seasons of odds data into one
    ----------
    Available at:
    https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
    """
    sets = (
        pd.read_excel("nba odds 2007-08.xlsx"),
        pd.read_excel("nba odds 2008-09.xlsx"),
        pd.read_excel("nba odds 2010-11.xlsx"),
        pd.read_excel("nba odds 2009-10.xlsx"),
        pd.read_excel("nba odds 2011-12.xlsx"),
        pd.read_excel("nba odds 2012-13.xlsx"),
        pd.read_excel("nba odds 2013-14.xlsx"),
        pd.read_excel("nba odds 2014-15.xlsx"),
        pd.read_excel("nba odds 2015-16.xlsx"),
        pd.read_excel("nba odds 2016-17.xlsx"),
        pd.read_excel("nba odds 2018-19.xlsx"),
        pd.read_excel("nba odds 2019-20.xlsx"),
        pd.read_excel("nba odds 2020-21.xlsx"),
        pd.read_excel("nba odds 2021-22.xlsx"),
    )

    data = pd.concat(sets, axis=0, join="outer").reset_index(drop=True)
    data.drop(columns=["Unnamed: 0", "Date"], inplace=True)
    data.rename(
        columns={
            "OU": "O/U",
            "ML_Home": "H_ML",
            "ML_Away": "A_ML",
            "Win_Margin": "MOV",
        },
        inplace=True,
    )

    return data


def clean_odds_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds some new columns to the data for analysis
    Makes some naming adjustments to keep things consistent
    Currently only using the last three seasons of odds data
    """

    data = data.drop(data.index[:11500])
    mask = (data["H_ML"] != "NL") & (data["Spread"] != "PK")
    data = data.loc[mask].reset_index(drop=True)

    data["H_ML"] = data["H_ML"].replace(",", "").astype(float)
    data["A_ML"] = data["A_ML"].replace(",", "").astype(float)
    data["Spread"] = data["Spread"].astype(float)
    data["H_Status"] = np.where(data["H_ML"] < 0, "Fav", "UD")
    data["A_Status"] = np.where(data["A_ML"] < 0, "Fav", "UD")
    data["O/U_Outcome"] = np.where(data["Points"] > data["O/U"], "Over", "Under")
    data["Spread_Outcome"] = np.where(data["MOV"] > data["Spread"], 1, 0)

    data["Home"] = np.where(
        data["Home"] == "New Jersey Nets", "Brooklyn Nets", data["Home"]
    )
    data["Away"] = np.where(
        data["Away"] == "New Jersey Nets", "Brooklyn Nets", data["Away"]
    )
    data["Home"] = np.where(
        data["Home"] == "Charlotte Bobcats", "Charlotte Hornets", data["Home"]
    )
    data["Away"] = np.where(
        data["Away"] == "Charlotte Bobcats", "Charlotte Hornets", data["Away"]
    )
    data["Home"] = np.where(
        data["Home"] == "Seattle SuperSonics", "Oklahoma City Thunder", data["Home"]
    )
    data["Away"] = np.where(
        data["Away"] == "Seattle SuperSonics", "Oklahoma City Thunder", data["Away"]
    )

    return data


def initial_odds_stats(data: pd.DataFrame) -> None:
    #  pylint: disable=too-many-locals
    """
    Purpose
    ----------
    Builds odds stats table

    Notable variables:
        fav_count:
            Numbers of games as the favorite

        ud_count:
            Number of games as the underdog

        fav_rate:
            Number of games as favorite (fav_count) / total games (game_count)

        ud_rate:
            Number of games as underdog (ud_count) / total games (game_count)

        fav_win:
            Percentage of games a team won as favorite (fav wins / fav_count)

        ud_win:
            Percentage of games won as underdog (ud wins / ud_count)

        cover_pct:
            Percentage of games where the Vegas spread is covered

        over_pct:
            Percentage of games where the Vegas point estimate is over

        under_pct:
            Percentage of games where the Vegas point estimate is under

        upset_def_opps:
            Number of games where a team is the heavy favorite (>80%)

        upset_def_win_pct:
            Number of heavy favorite games won / number of opportunities (upset_def_opps)

        upset_off_opps:
            Number of games where a team is the heavy underdog (<20%)

        upset_off_win_pct:
            Number of heavy underdog games won / number of opportunities (upset_off_opps)

    Commits initial odds stats table to database
    """
    final_arr = []
    team_list = data["Home"].unique()

    for team in team_list:
        arr = []

        h_games = data.loc[data["Home"] == team].reset_index(drop=True)
        a_games = data.loc[data["Away"] == team].reset_index(drop=True)

        fav_count = sum(h_games["H_Status"] == "Fav") + sum(
            a_games["A_Status"] == "Fav"
        )
        ud_count = sum(h_games["H_Status"] == "UD") + sum(a_games["A_Status"] == "UD")
        game_count = len(h_games.index) + len(a_games.index)

        fav_rate = fav_count / game_count
        ud_rate = ud_count / game_count

        h_fav_mask = (h_games["H_Status"] == "Fav") & (h_games["MOV"] > 0)
        a_fav_mask = (a_games["A_Status"] == "Fav") & (a_games["MOV"] < 0)
        h_fav_win = sum(h_fav_mask)
        a_fav_win = sum(a_fav_mask)
        fav_win = (h_fav_win + a_fav_win) / fav_count

        h_ud_mask = (h_games["H_Status"] == "UD") & (h_games["MOV"] > 0)
        a_ud_mask = (a_games["A_Status"] == "UD") & (a_games["MOV"] < 0)
        h_ud_win = sum(h_ud_mask)
        a_ud_win = sum(a_ud_mask)
        ud_win = (h_ud_win + a_ud_win) / ud_count

        h_spread_mask = h_games["Spread_Outcome"] == 1
        a_spread_mask = a_games["Spread_Outcome"] == 1
        cover_pct = (sum(h_spread_mask) + sum(a_spread_mask)) / game_count

        under_mask = h_games["O/U_Outcome"] == "Under"
        over_mask = a_games["O/U_Outcome"] == "Over"
        under_pct = (sum(under_mask) + (len(over_mask) - sum(over_mask))) / game_count
        over_pct = (sum(over_mask) + (len(under_mask) - sum(under_mask))) / game_count

        upset_h_def_games_mask = h_games["H_ML"] < -400
        upset_a_def_games_mask = a_games["A_ML"] < -400
        upset_def_opps = sum(upset_h_def_games_mask) + sum(upset_a_def_games_mask)

        upset_h_def_wins_mask = (h_games["H_ML"] < -400) & (h_games["MOV"] > 0)
        upset_a_def_wins_mask = (a_games["A_ML"] < -400) & (a_games["MOV"] < 0)
        upset_def_wins = sum(upset_h_def_wins_mask) + sum(upset_a_def_wins_mask)
        upset_def_win_pct = upset_def_wins / upset_def_opps

        upset_h_off_games_mask = h_games["H_ML"] > 400
        upset_a_off_games_mask = a_games["A_ML"] > 400
        upset_off_opps = sum(upset_h_off_games_mask) + sum(upset_a_off_games_mask)

        upset_h_off_wins_mask = (h_games["H_ML"] > 400) & (h_games["MOV"] > 0)
        upset_a_off_wins_mask = (a_games["A_ML"] > 400) & (a_games["MOV"] < 0)
        upset_off_wins = sum(upset_h_off_wins_mask) + sum(upset_a_off_wins_mask)
        upset_off_win_pct = upset_off_wins / upset_off_opps

        arr.extend(
            [
                team,
                fav_count,
                fav_rate,
                fav_win,
                ud_count,
                ud_rate,
                ud_win,
                cover_pct,
                under_pct,
                over_pct,
                upset_def_opps,
                upset_def_wins,
                upset_def_win_pct,
                upset_off_opps,
                upset_off_wins,
                upset_off_win_pct,
            ]
        )

        final_arr.append(arr)

    final_data = pd.DataFrame(
        final_arr,
        columns=[
            "Team",
            "Fav_GP",
            "Fav_R",
            "Fav_W%",
            "UD_GP",
            "UD_R",
            "UD_W%",
            "Cover%",
            "Under%",
            "Over%",
            "Def_GP",
            "Def_Wins",
            "Def_W%",
            "Off_GP",
            "Off_Wins",
            "Off_W%",
        ],
    )

    final_data = round(final_data, 3)
    final_data.to_sql("odds_stats", const.ENGINE, if_exists="replace", index=False)


@utils.timerun
def clean_train() -> None:
    """
    Cleans the raw scraped file for training the model
    Adds Massey/Elo Ratings to Raw Data
    """

    data = pd.read_sql_table("raw_data", const.ENGINE)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Outcome"] = np.where(
        data["H-Pts"].astype(float) > data["A-Pts"].astype(float), 1, 0
    )

    data.drop(["Time", "A-Pts", "H-Pts", "OT"], axis=1, inplace=True)

    data = add_massey(data)
    # data = add_elo(data)
    data.to_sql("training_data", const.ENGINE, if_exists="replace", index=False)


def commit_sch():
    """
    Commits hard copy full schedule file to database
    """
    data = pd.read_csv("FullGamesFinal.csv")
    data = set_extras(data)
    data.to_sql("full_sch", const.ENGINE, if_exists="replace", index=False)


def set_extras(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Season ID and MOV columns
    """
    data["Date"] = pd.to_datetime(data["Date"])

    for i in data.index:
        start_year = str(data.at[i, "Date"]).split("-")[0]
        start_month = str(data.at[i, "Date"]).split("-")[1]

        if int(start_month) <= 8:
            start_year = str(int(start_year) - 1)

        end_year = list(str(int(start_year) + 1))
        end_year = end_year[2] + end_year[3]

        data.at[i, "SeasonID"] = start_year + "-" + end_year
        data.at[i, "MOV"] = int(data.at[i, "H-Pts"]) - int(data.at[i, "A-Pts"])

    return data


def combine_dailies():
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


def update_history_outcomes() -> None:
    """
    Loads most recent prediction history and played game scores
    Adds actual scores and outcome to prediction history
    Commits prediction scoring table to database
    """
    arr = []
    predicted = pd.read_sql_table("prediction_history_net", const.ENGINE)
    predicted["Actual"] = "TBD"

    played = pd.read_sql_table("2023_played_games", const.ENGINE)

    pred_mask = predicted["Date"] != str(date.today())
    predicted = predicted.loc[pred_mask].reset_index(drop=True)

    for d in predicted["Date"].unique():
        mov_dict = {}
        played_mask = played["Date"] == d
        filtered_played = played.loc[played_mask].reset_index(drop=True)

        history_mask = predicted["Date"] == d
        filtered_predicted = predicted.loc[history_mask].reset_index(drop=True)

        for i in filtered_played.index:
            mov_dict[filtered_played.at[i, "Away"]] = filtered_played.at[i, "MOV"]

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


def clean_box_data(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = data.columns
    data = data.drop(data.columns[[10, 29, 30, 42, 45, 46, 65, 66, 78, 81]], axis=1)
    data.columns = const.BOX_FEATURES

    return data


if __name__ == "__main__":
    # combine_datasets()
    # clean_train()

    # raw_data = combine_odds_dataset()
    # full_data = clean_odds_data(raw_data)
    # initial_odds_stats(full_data)

    pass
