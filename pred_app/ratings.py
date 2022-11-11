"""
Docstring
"""
from datetime import date
import typing as t
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils

MEAN_ELO = 1500
ELO_WIDTH = 400
K_FACTOR = 64


@utils.timerun
def add_elo(concat_to: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Elo to the file containing the full schedule (2008-2022)
    """
    data = pd.read_csv("FullGamesFinal.csv")
    data["Date"] = pd.to_datetime(data["Date"])

    season = 0
    full_arrays = []

    for season in tqdm(data["SeasonID"].unique()):
        filtered_data = data.loc[data["SeasonID"] == season].reset_index(drop=True)
        filtered_data.drop(filtered_data.columns[[1, 6, 7, 8]], axis=1, inplace=True)

        if season > 0:
            current_elos = update_new_season(current_elos)

        else:
            current_elos = (
                np.ones(shape=(len(filtered_data["Away"].unique()))) * MEAN_ELO
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


def adjust_elo(a_elo: float, h_elo: float, outcome: int) -> tuple[float, float]:
    """
    Changes Elo based on actual and expected outcome
    """
    expected_win = expected_outcome(a_elo, h_elo, outcome)
    change_in_elo = K_FACTOR * (1 - expected_win)

    if outcome == 1:
        h_elo += change_in_elo
        a_elo -= change_in_elo
    else:
        h_elo -= change_in_elo
        a_elo += change_in_elo

    return a_elo, h_elo


def expected_outcome(a_elo: float, h_elo: float, outcome: int) -> float:
    """
    Calculates expected outcome
    """
    if outcome == 1:
        expected = 1.0 / (1 + 10 ** ((a_elo - h_elo) / ELO_WIDTH))
    else:
        expected = 1.0 / (1 + 10 ** ((h_elo - a_elo) / ELO_WIDTH))

    return expected


def update_new_season(elos: t.Any) -> t.Any:
    """
    Uses mean regression to shift teams towards base ELO
    """

    diff_from_mean = elos - MEAN_ELO
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


@utils.timerun
def add_massey(concat_to: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Massey Ratings for all seasons in Training File
    Concats Massey Ratings to provided file (usually schedule or raw stats)
    """

    data = pd.read_csv("FullGamesFinal.csv")
    data["Date"] = pd.to_datetime(data["Date"])

    check_date = date.today()
    full_arrays = []

    for season in tqdm(data["SeasonID"].unique()):
        j = 0
        filtered_data = data.loc[data["SeasonID"] == season].reset_index(drop=True)

        for i in filtered_data.index:
            arr: list[float] = []

            if j < 20:
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


def current_massey(data: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """
    Gets Massey Ratings for the current season up to the most recent played game
    """
    check_date = date.today()
    full_arrays = []
    j = 0

    for i in data.index:
        arr: list[float] = []

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

    cur_massey = final.sort_index(axis=0, ascending=False)
    cur_massey = cur_massey.groupby("Name").head(1).reset_index(drop=True)
    cur_massey.drop(cur_massey.tail(1).index, inplace=True)
    cur_massey["Conf"] = cur_massey["Name"].map(utils.conf_dict)
    cur_massey = cur_massey.sort_values("Massey", ascending=False).reset_index(
        drop=True
    )
    cur_massey.to_sql("Current_Massey", utils.engine, if_exists="replace", index=False)

    massey_ratings = final.groupby("Name")

    return massey_ratings


if __name__ == "__main__":
    # data = pd.read_csv('FullGamesFinal.csv')
    # season_id = '2021-22'
    # game_date = '2022-05-01'

    # massey = get_massey(data, season_id, game_date)
    # add_massey()
    # add_elo()
    pass
