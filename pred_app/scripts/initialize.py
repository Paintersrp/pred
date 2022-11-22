"""
This script contains data cleaning/transforming/amending functions
"""
from datetime import date
import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts import utils, const, handler
from scripts.ratings import get_massey, adjust_elo, update_elo_new_season


def initialize_training() -> None:
    """
    Initializes training dataset from backups/hardcopies
    """

    data = combine_datasets()
    commit_sch()
    data = clean_train(data)
    commit_train(data)


def combine_datasets() -> pd.DataFrame:
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

    return final


# update to scrape schedule
def commit_sch() -> None:
    """
    Commits hard copy full schedule file to database
    """

    data = pd.read_csv("FullGamesFinal.csv")
    data = set_extras(data)
    data.to_sql("full_sch", const.ENGINE, if_exists="replace", index=False)


def set_extras(data: pd.DataFrame) -> pd.DataFrame:
    #  pylint: disable=use-maxsplit-arg
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


def clean_train(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw scraped file for training the model
    Adds Massey/Elo Ratings to Raw Data
    """

    data["Date"] = pd.to_datetime(data["Date"])
    data["Outcome"] = np.where(
        data["H-Pts"].astype(float) > data["A-Pts"].astype(float), 1, 0
    )

    data.drop(["Time", "A-Pts", "H-Pts", "OT"], axis=1, inplace=True)

    data = add_massey(data)
    # data = add_elo(data)

    return data


def add_massey(concat_to: pd.DataFrame) -> pd.DataFrame:
    #  pylint: disable=too-many-locals
    """
    Calculates Massey Ratings for all seasons in Training File

    Concats Massey Ratings to input dataset
    """

    data = pd.read_sql_table("full_sch", const.ENGINE)
    data["Date"] = pd.to_datetime(data["Date"])

    check_date = date.today()
    full_arrays = []

    for season in tqdm(data["SeasonID"].unique()):
        j = 0
        filtered_data = data.loc[data["SeasonID"] == season].reset_index(drop=True)

        for i in filtered_data.index:
            arr = []

            if j < 55:
                arr.extend([0, 0])
                full_arrays.append(arr)
                j += 1

            else:
                game_date = filtered_data.at[i, "Date"]

                if check_date != game_date:
                    check_date = game_date
                    date_mask = filtered_data["Date"] < game_date
                    dated_data = filtered_data.loc[date_mask].reset_index(drop=True)
                    massey = get_massey(dated_data, season, game_date)

                    away_rating = massey.get_group(filtered_data.at[i, "Away"])[
                        "Rating"
                    ]
                    home_rating = massey.get_group(filtered_data.at[i, "Home"])[
                        "Rating"
                    ]

                    arr.extend([float(away_rating), float(home_rating)])
                    full_arrays.append(arr)

                else:
                    away_rating = massey.get_group(filtered_data.at[i, "Away"])[
                        "Rating"
                    ]
                    home_rating = massey.get_group(filtered_data.at[i, "Home"])[
                        "Rating"
                    ]

                    arr.extend([float(away_rating), float(home_rating)])
                    full_arrays.append(arr)

    final = pd.DataFrame(full_arrays, columns=["A_Massey", "H_Massey"])
    final = pd.concat([concat_to, final], axis=1, join="outer")

    return final


def add_elo(concat_to: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Elo to the file containing the full schedule (2008-2022)
    """

    data = pd.read_sql_table("full_sch", const.ENGINE)
    data["Date"] = pd.to_datetime(data["Date"])

    season = 0
    full_arrays = []

    for season in tqdm(data["SeasonID"].unique()):
        filtered_data = data.loc[data["SeasonID"] == season].reset_index(drop=True)
        filtered_data.drop(filtered_data.columns[[1, 6, 7, 8]], axis=1, inplace=True)

        if season > 0:
            current_elos = update_elo_new_season(current_elos)

        else:
            current_elos = (
                np.ones(shape=(len(filtered_data["Away"].unique()))) * const.MEAN_ELO
            )

        map_df = filtered_data.drop(filtered_data.columns[[0, 2, 4]], axis=1)
        teams = map_df.stack().unique()
        teams.sort()
        f_codes = pd.factorize(teams)
        f_codes = pd.Series(f_codes[0], f_codes[1])
        teams = map_df.stack().map(f_codes).unstack()

        filtered_data["Outcome"] = np.where(
            filtered_data["H-Pts"] > filtered_data["A-Pts"], 1, 0
        )

        final = pd.concat([teams, filtered_data["Outcome"]], axis=1, join="outer")

        season += 1

        for i in final.index:
            arr = []

            a_end_elo, h_end_elo = adjust_elo(
                current_elos[final.at[i, "Away"]],
                current_elos[final.at[i, "Home"]],
                final.at[i, "Outcome"],
            )

            current_elos[final.at[i, "Away"]] = a_end_elo
            current_elos[final.at[i, "Home"]] = h_end_elo

            arr.extend([round(a_end_elo, 2), round(h_end_elo, 2)])
            full_arrays.append(arr)

    temp = pd.DataFrame(full_arrays, columns=["A_ELO", "H_ELO"])
    final = pd.concat([concat_to, temp], axis=1, join="outer")

    return final


def commit_train(data) -> None:
    """Commits initial training dataset to database"""

    data.to_sql("training_data", const.ENGINE, if_exists="replace", index=False)


def initialize_odds() -> None:
    """
    Initializes odds dataset from backups/hardcopies
    """

    data = combine_odds_dataset()
    data = clean_odds_data(data)
    initial_odds_stats(data)
    commit_full_odds_initial(data)


def combine_odds_dataset() -> pd.DataFrame:
    """
    Combines hard-copy seasons of odds data into one
    ----------
    Available at:
    https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
    """

    datasets = (
        pd.read_excel("nba odds 2007-08.xlsx"),
        pd.read_excel("nba odds 2008-09.xlsx"),
        pd.read_excel("nba odds 2009-10.xlsx"),
        pd.read_excel("nba odds 2010-11.xlsx"),
        pd.read_excel("nba odds 2011-12.xlsx"),
        pd.read_excel("nba odds 2012-13.xlsx"),
        pd.read_excel("nba odds 2013-14.xlsx"),
        pd.read_excel("nba odds 2014-15.xlsx"),
        pd.read_excel("nba odds 2015-16.xlsx"),
        pd.read_excel("nba odds 2016-17.xlsx"),
        pd.read_excel("nba odds 2017-18.xlsx"),
        pd.read_excel("nba odds 2018-19.xlsx"),
        pd.read_excel("nba odds 2019-20.xlsx"),
        pd.read_excel("nba odds 2020-21.xlsx"),
        pd.read_excel("nba odds 2021-22.xlsx"),
    )

    year_range = range(2008, 2023)

    for dataset, year in zip(datasets, year_range):
        dataset["Open"] = np.where(dataset["Open"] == "pk", 0, dataset["Open"])
        dataset["Open"] = np.where(dataset["Open"] == "PK", 0, dataset["Open"])
        dataset["Close"] = np.where(dataset["Close"] == "pk", 0, dataset["Close"])
        dataset["ML"] = np.where(dataset["ML"] == "NL", 0, dataset["ML"])
        dataset["Open"] = dataset["Open"].astype(int)
        dataset["Close"] = dataset["Open"].astype(int)

        for i in dataset["Date"].index:
            test = list(str(dataset.at[i, "Date"]))

            if len(test) == 4:
                month_check = test[0] + test[1]
                day_check = test[2] + test[3]
            else:
                month_check = test[0]
                day_check = test[1] + test[2]

            if int(month_check) > 9:
                dataset.at[i, "Date"] = (
                    f"{year-1}" + "-" + month_check + "-" + day_check
                )
            else:
                dataset.at[i, "Date"] = f"{year}" + "-" + month_check + "-" + day_check

    data = pd.concat(datasets, axis=0, join="outer").reset_index(drop=True)

    return data


def clean_odds_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds some new columns to the data for analysis
    Makes some naming adjustments to keep things consistent
    Currently only using the last three seasons of odds data
    """

    data = data[["Date", "VH", "Team", "Final", "Open", "ML"]]

    for i in data["Team"].index:
        data.at[i, "Team"] = utils.map_odds_team(data.at[i, "Team"])

    data = data[data["VH"].isin(["V", "H"])]

    data["Game No"] = (data["VH"] == "V").cumsum()
    data = data.set_index(["Game No", "VH"]).unstack().reset_index(drop=True)
    data.columns = data.columns.map("_".join)
    data = data.drop(["Date_V"], axis=1)

    data = data.rename(
        columns={
            "Open_H": "Spread",
            "Open_V": "O/U",
            "ML_H": "H_ML",
            "ML_V": "A_ML",
            "Team_H": "H_Team",
            "Team_V": "A_Team",
            "Final_H": "H_Pts",
            "Final_V": "A_Pts",
            "Date_H": "Date",
        }
    )

    data["MOV"] = data["H_Pts"] - data["A_Pts"]

    for i in data.index:
        if data.at[i, "Spread"] > 50:
            temp = data.at[i, "Spread"]
            data.at[i, "Spread"] = data.at[i, "O/U"]
            data.at[i, "O/U"] = temp

    data["H_ML"] = data["H_ML"].astype(float)
    data["A_ML"] = data["A_ML"].astype(float)
    data["Spread"] = data["Spread"].astype(float)
    data["H_Status"] = np.where(data["H_ML"] < 0, "Fav", "UD")
    data["A_Status"] = np.where(data["A_ML"] < 0, "Fav", "UD")
    data["O/U_Outcome"] = np.where(
        (data["H_Pts"] + data["A_Pts"]) > data["O/U"], "Over", "Under"
    )
    data["Spread_Outcome"] = np.where(data["MOV"] > data["Spread"], 1, 0)

    return data


def commit_full_odds_initial(odds_data: pd.DataFrame) -> None:
    """
    Commits initial odds dataset to database
    """

    odds_data.to_sql(
        "full_odds_history_initial", const.ENGINE, if_exists="replace", index=False
    )
    odds_data.to_sql(
        "full_odds_history", const.ENGINE, if_exists="replace", index=False
    )


def initial_odds_stats(data: pd.DataFrame) -> None:
    #  pylint: disable=too-many-locals
    """
    Purpose
    ----------
    Builds odds stats table

    Don't hate, appreciate.

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
    team_list = data["H_Team"].unique()

    for team in team_list:
        arr = []

        h_games = data.loc[data["H_Team"] == team].reset_index(drop=True)
        a_games = data.loc[data["A_Team"] == team].reset_index(drop=True)

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


def initial_sim_pred():
    """
    Initializes simulation dataset
    """

    data_handler = handler.GeneralHandler()
    training_data = data_handler.training_data()
    odds_history = data_handler.full_odds_history()

    training_data["Date"] = pd.to_datetime(training_data["Date"])
    odds_history["Date"] = pd.to_datetime(odds_history["Date"])

    for i in training_data.index:
        game_date = training_data.at[i, "Date"]

        mask = odds_history["Date"] == game_date
        temp = odds_history.loc[mask].reset_index(drop=True)

        for j in temp.index:
            if temp.at[j, "H_Team"] == training_data.at[i, "Home"]:
                training_data.at[i, "O/U"] = temp.at[j, "O/U"]
                training_data.at[i, "H_ML"] = temp.at[j, "H_ML"]
                training_data.at[i, "A_ML"] = temp.at[j, "A_ML"]
                training_data.at[i, "Spread"] = temp.at[j, "Spread"]
                training_data.at[i, "OU_Outcome"] = temp.at[j, "O/U_Outcome"]

    training_data.sort_values(["Date", "Home"], ascending=True, inplace=True)
    training_data.reset_index(drop=True, inplace=True)
    training_data.dropna(inplace=True)
    mask = (training_data["Spread"].astype(float) != 0) & (
        training_data["O/U"].astype(float) != 0
    )
    training_data = training_data.loc[mask].reset_index(drop=True)

    print(training_data)

    training_data.to_sql(
        "sim_pred_data", const.ENGINE, if_exists="replace", index=False
    )


def initialize_sim_random_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Initializes simulation dataset (Random Variant)
    """

    sch_data = pd.read_sql_table("simulator_sch", const.ENGINE)

    num_r = len(sch_data) - len(data)
    np.random.seed(10)
    drop_indices = np.random.choice(sch_data.index, num_r, replace=False)
    sch_data = sch_data.drop(drop_indices).reset_index(drop=True)

    sch_data["H-Pts"] = sch_data["H-Pts"].astype(int)
    final_data = pd.concat([sch_data, data], axis=1, join="outer")
    final_data["ML_Payout"] = np.where(
        final_data["MOV"] > 0, final_data["H_ML"], final_data["A_ML"]
    )

    rearrange = [col for col in final_data.columns if col != "H_ML"] + ["H_ML"]
    final_data = final_data[rearrange]

    return final_data


def initialize_boxscore_data() -> None:
    """
    Initializes boxscore dataset into database
    """

    datasets = []
    year_range = range(2008, 2023)

    for year in year_range:
        datasets.append(pd.read_csv(f"BoxscoreData_{year}.csv"))

    final = pd.concat(datasets, axis=0, join="outer").reset_index(drop=True)
    final.columns = final.columns
    final = final.drop(final.columns[[10, 29, 30, 42, 45, 46, 65, 66, 78, 81]], axis=1)
    final.columns = const.BOX_FEATURES
    final[const.BOX_DISPLAY_FEATURES] = final[const.BOX_DISPLAY_FEATURES].astype(float)

    print(final)
    # final.to_sql("boxscore_data", const.ENGINE, if_exists="replace", index=False)

    return final


if __name__ == "__main__":
    pass
