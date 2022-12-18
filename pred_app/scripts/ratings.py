"""
This scripts contains functions for adding ELO/Massey Ratings to new and historical data
"""
from datetime import date
import typing as t
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts.handler import GeneralHandler
from scripts import const

DATAHANDLER = GeneralHandler()


def add_elo(concat_to: pd.DataFrame, data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Adds Elo to the file containing the full schedule (2008-2022)
    """

    if data is None:
        data = pd.read_sql_table("full_sch", const.ENGINE)
        data["Date"] = pd.to_datetime(data["Date"])

    season = 0.0
    full_arrays = []
    data["H-Pts"] = data["H-Pts"].astype(float)
    data["A-Pts"] = data["A-Pts"].astype(float)

    for season_id in tqdm(data["SeasonID"].unique()):
        filtered_data = data.loc[data["SeasonID"] == season_id].reset_index(drop=True)
        filtered_data.drop(filtered_data.columns[[1, 6, 7, 8]], axis=1, inplace=True)

        if season > 0:
            cur_elos = update_elo_new_season(cur_elos)

        else:
            cur_elos = (
                np.ones(shape=(len(filtered_data["Away"].unique()))) * const.MEAN_ELO
            )

        filtered_data["MOV"] = np.where(
            filtered_data["A-Pts"] > filtered_data["H-Pts"],
            filtered_data["A-Pts"] - filtered_data["H-Pts"],
            filtered_data["H-Pts"] - filtered_data["A-Pts"],
        )
        filtered_data["Outcome"] = np.where(
            filtered_data["A-Pts"] > filtered_data["H-Pts"], 0, 1
        )

        map_df = filtered_data.drop(filtered_data.columns[[0, 2, 4, 5, 6]], axis=1)
        teams = map_df.stack().unique()
        teams.sort()
        f_codes = pd.factorize(teams)
        f_codes = pd.Series(f_codes[0], f_codes[1])
        teams = map_df.stack().map(f_codes).unstack()

        final = pd.concat(
            [teams, filtered_data[["MOV", "Outcome"]]], axis=1, join="outer"
        )

        season += 1

        for idx in final.index:
            arr = []

            if final.at[idx, "Outcome"] == 0:
                a_end_elo, h_end_elo, a_start_elo, h_start_elo = adjust_elo(
                    cur_elos[final.at[idx, "Away"]],
                    cur_elos[final.at[idx, "Home"]],
                    final.at[idx, "MOV"],
                    0,
                )
                cur_elos[final.at[idx, "Away"]] = a_end_elo
                cur_elos[final.at[idx, "Home"]] = h_end_elo
            else:
                h_end_elo, a_end_elo, h_start_elo, a_start_elo = adjust_elo(
                    cur_elos[final.at[idx, "Home"]],
                    cur_elos[final.at[idx, "Away"]],
                    final.at[idx, "MOV"],
                    1,
                )
                cur_elos[final.at[idx, "Home"]] = h_end_elo
                cur_elos[final.at[idx, "Away"]] = a_end_elo

            arr.extend([round(a_start_elo, 2), round(h_start_elo, 2)])
            full_arrays.append(arr)

    temp = pd.DataFrame(full_arrays, columns=["A_ELO", "H_ELO"])
    final = pd.concat([concat_to, temp], axis=1, join="outer")
    print(final)

    return final


def current_elos() -> dict:
    """Func"""

    schedule = DATAHANDLER.training_schedule()

    schedule["H-Pts"] = schedule["H-Pts"].astype(float)
    schedule["A-Pts"] = schedule["A-Pts"].astype(float)
    schedule["Home"] = schedule["Home"].astype(str)
    schedule["Away"] = schedule["Away"].astype(str)

    schedule = schedule.rename(
        columns={
            "H-Pts": "H_Pts",
            "A-Pts": "A_Pts",
        }
    )

    schedule["Home"] = np.where(
        schedule["Home"] == "Seattle SuperSonics",
        "Oklahoma City Thunder",
        schedule["Home"],
    )
    schedule["Home"] = np.where(
        schedule["Home"] == "New Orleans Hornets",
        "New Orleans Pelicans",
        schedule["Home"],
    )
    schedule["Home"] = np.where(
        schedule["Home"] == "New Orleans/Oklahoma City Hornets",
        "New Orleans Pelicans",
        schedule["Home"],
    )
    schedule["Home"] = np.where(
        schedule["Home"] == "New Jersey Nets", "Brooklyn Nets", schedule["Home"]
    )
    schedule["Home"] = np.where(
        schedule["Home"] == "Charlotte Bobcats", "Charlotte Hornets", schedule["Home"]
    )
    schedule["Home"] = np.where(
        schedule["Home"] == "Los Angeles Clippers", "LA Clippers", schedule["Home"]
    )

    schedule["Away"] = np.where(
        schedule["Away"] == "Seattle SuperSonics",
        "Oklahoma City Thunder",
        schedule["Away"],
    )
    schedule["Away"] = np.where(
        schedule["Away"] == "New Orleans Hornets",
        "New Orleans Pelicans",
        schedule["Away"],
    )
    schedule["Away"] = np.where(
        schedule["Away"] == "New Orleans/Oklahoma City Hornets",
        "New Orleans Pelicans",
        schedule["Away"],
    )
    schedule["Away"] = np.where(
        schedule["Away"] == "New Jersey Nets", "Brooklyn Nets", schedule["Away"]
    )
    schedule["Away"] = np.where(
        schedule["Away"] == "Charlotte Bobcats", "Charlotte Hornets", schedule["Away"]
    )
    schedule["Away"] = np.where(
        schedule["Away"] == "Los Angeles Clippers", "LA Clippers", schedule["Away"]
    )

    team_ids = schedule["Home"].unique()
    elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

    for row in schedule.itertuples():
        if row.H_Pts > row.A_Pts:
            winner = row.Home
            loser = row.Away
            home_win = 1
        else:
            winner = row.Away
            loser = row.Home
            home_win = 0

        margin = abs(row.MOV)

        new_w_elo, new_l_elo, _old_w_elo, _old_l_elo = adjust_elo(
            elo_dict[winner], elo_dict[loser], margin, home_win
        )

        elo_dict[winner] = new_w_elo
        elo_dict[loser] = new_l_elo

    return elo_dict


def adjust_elo(
    w_elo: float, l_elo: float, margin: float, home_status: int
) -> tuple[float, float]:
    """
    Changes Elo based on actual and expected outcome
    """

    if home_status == 1:
        elo_diff = (w_elo + const.HOME_ADVANTAGE) - l_elo
    else:
        elo_diff = w_elo - (l_elo + const.HOME_ADVANTAGE)

    ex_outcome = expected_outcome(elo_diff)
    ex_margin = ((margin + 3.0) ** 0.8) / expected_margin(elo_diff)
    elo_change = const.K_FACTOR * ex_margin * (1 - ex_outcome)

    new_w_elo = w_elo + elo_change
    new_l_elo = l_elo - elo_change

    return new_w_elo, new_l_elo, w_elo, l_elo


def expected_outcome(elo_diff: float) -> float:
    """
    Calculates expected outcome
    """
    return 1.0 / (math.pow(10.0, (-elo_diff / 400.0)) + 1.0)


def expected_margin(elo_diff):
    """Func"""

    return 7.5 + 0.006 * elo_diff


def update_elo_new_season(elos: t.Any) -> t.Any:
    """
    Uses mean regression to shift teams towards base ELO
    """

    diff_from_mean = elos - const.MEAN_ELO
    elos -= diff_from_mean / 3

    return elos


def get_massey(data: pd.DataFrame, season_id: str, game_date: date) -> pd.DataFrame:
    """
    Calculates up-to-date Massey Ratings for given season/end date
    """

    #  Filters the data so that it only calculates rating based off season and date provided
    mask = (data["SeasonID"] == season_id) & (data["Date"] < game_date)
    filtered_data = data.loc[mask].reset_index(drop=True)
    filtered_data.drop(filtered_data.columns[[1, 3, 5, 6, 7]], axis=1, inplace=True)

    #  Converts team name to numeric codes for the algorithm
    map_df = filtered_data.drop(filtered_data.columns[[0, 3]], axis=1)
    teams = map_df.stack().unique()
    teams.sort()
    codes = pd.factorize(teams)
    codes = pd.Series(codes[0], codes[1])
    map_teams = map_df.stack().map(codes).unstack()

    #  Creates a new DataFrame with Team Names replaced with Team Codes
    #  Team and Game Counts will shape the equation matrixes
    filtered_data = pd.concat([map_teams, filtered_data["MOV"]], axis=1, join="outer")

    #  Setting up the right side matrix for the equation
    #  The +1 to team count is to account for adding Home Court Advantage
    right = np.zeros([len(filtered_data["Home"]), len(teams) + 1])

    for i, game in filtered_data.iterrows():

        #  Fills Matrix with numeric values representing which teams played in the game
        right[i, int(game[1])] = 1
        right[i, int(game[0])] = -1

        #  Appends another "team" representing Home Court Advantage for calculating adjusted rating
        right[i, len(teams)] = 1

    #  Setting up the left side matrix for the equation
    left = np.zeros([len(filtered_data["Home"])])

    for i, game in filtered_data.iterrows():
        #  Adds score margin for game in matrix
        left[i] = game[2]

    #  In order to avoid an unsolvable equation, connectivity is added
    #  Simply put, adds a fake game into the dataset
    #  The appended 0 is to keep this fake "game" factored out of Home Court Advantage as well
    connectivity = np.ones(len(teams))
    connectivity = np.append(connectivity, 0)
    right = np.vstack((right, connectivity))
    left = np.append(left, 0)

    #  Solves equations using a least squared algorithm
    solutions = np.linalg.lstsq(right, left, rcond=None)[0]
    massey = list(zip(teams, solutions))

    massey = pd.DataFrame(massey, columns=["Name", "Rating"]).groupby("Name")

    return massey


def current_massey(data: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """
    Gets Massey Ratings for the current season up to the most recently played game
    """

    check_date = date.today()
    full_arrays = []
    j = 0

    for i in data.index:
        arr = []

        if j < 20:
            arr.extend([0, 0])
            full_arrays.append(arr)
            j += 1

        else:
            game_date = data.at[i, "Date"]

            if check_date != game_date:
                check_date = game_date
                date_mask = data["Date"] < game_date
                dated_data = data.loc[date_mask].reset_index(drop=True)
                massey = get_massey(dated_data, season_code, game_date)

                away_rating = massey.get_group(data.at[i, "Away"])["Rating"]
                home_rating = massey.get_group(data.at[i, "Home"])["Rating"]

                arr.extend([data.at[i, "Away"], float(away_rating)])
                full_arrays.append(arr)

                arr = []

                arr.extend([data.at[i, "Home"], float(home_rating)])
                full_arrays.append(arr)

            else:
                away_rating = massey.get_group(data.at[i, "Away"])["Rating"]
                home_rating = massey.get_group(data.at[i, "Home"])["Rating"]

                arr.extend([data.at[i, "Away"], float(away_rating)])
                full_arrays.append(arr)

                arr = []

                arr.extend([data.at[i, "Home"], float(home_rating)])
                full_arrays.append(arr)

    final = pd.DataFrame(full_arrays, columns=["Name", "Massey"])
    final["Name"] = np.where(
        final["Name"] == "Los Angeles Clippers", "LA Clippers", final["Name"]
    )

    return final


if __name__ == "__main__":
    pass
