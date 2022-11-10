from ratings import add_massey, add_elo
import pandas as pd
import numpy as np
import utils


def combine_datasets():
    sets = []
    sets.extend(
        [
            pd.read_csv("TrainDataRawAdv_2005-09.csv"),
            pd.read_csv("TrainDataRawAdv_2010-14.csv"),
            pd.read_csv("TrainDataRawAdv_2015-18.csv"),
            pd.read_csv("TrainDataRawAdv_2019-22.csv"),
        ]
    )

    final = pd.concat(sets, axis=0, join="outer").reset_index(drop=True)
    final.drop(final[["A_TEAM_NAME", "H_TEAM_NAME"]], axis=1, inplace=True)
    final.to_csv("Train.csv")


def combine_odds_dataset():
    sets = []
    sets.extend(
        [
            pd.read_excel("2007-08.xlsx"),
            pd.read_excel("2008-09.xlsx"),
            pd.read_excel("2009-10.xlsx"),
            pd.read_excel("2010-11.xlsx"),
            pd.read_excel("2011-12.xlsx"),
            pd.read_excel("2012-13.xlsx"),
            pd.read_excel("2013-14.xlsx"),
            pd.read_excel("2014-15.xlsx"),
            pd.read_excel("2015-16.xlsx"),
            pd.read_excel("2016-17.xlsx"),
            pd.read_excel("2018-19.xlsx"),
            pd.read_excel("2019-20.xlsx"),
            pd.read_excel("2020-21.xlsx"),
            pd.read_excel("2021-22.xlsx"),
        ]
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


def initial_odds_stats(df):
    final_arr = []

    #  Loads data, removes some games marked PK/NL (No Favorite)
    df = df.drop(df.index[:11500])
    mask = (df["H_ML"] != "NL") & (df["Spread"] != "PK")
    df = df.loc[mask].reset_index(drop=True)

    #  Sets up dataframe by converting column types
    #  Adds Fav/UD Status Columns
    #  Adds O/U and Spread Outcome Columns
    #  Renames some team names for consistency
    df["H_ML"] = df["H_ML"].replace(",", "").astype(float)
    df["A_ML"] = df["A_ML"].replace(",", "").astype(float)
    df["Spread"] = df["Spread"].astype(float)
    df["H_Status"] = np.where(df["H_ML"] < 0, "Fav", "UD")
    df["A_Status"] = np.where(df["A_ML"] < 0, "Fav", "UD")
    df["O/U_Outcome"] = np.where(df["Points"] > df["O/U"], "Over", "Under")
    df["Spread_Outcome"] = np.where(df["MOV"] > df["Spread"], 1, 0)
    df["Home"] = np.where(df["Home"] == "New Jersey Nets", "Brooklyn Nets", df["Home"])
    df["Away"] = np.where(df["Away"] == "New Jersey Nets", "Brooklyn Nets", df["Away"])
    df["Home"] = np.where(
        df["Home"] == "Charlotte Bobcats", "Charlotte Hornets", df["Home"]
    )
    df["Away"] = np.where(
        df["Away"] == "Charlotte Bobcats", "Charlotte Hornets", df["Away"]
    )
    df["Home"] = np.where(
        df["Home"] == "Seattle SuperSonics", "Oklahoma City Thunder", df["Home"]
    )
    df["Away"] = np.where(
        df["Away"] == "Seattle SuperSonics", "Oklahoma City Thunder", df["Away"]
    )
    team_list = df["Home"].unique()

    for team in team_list:
        arr = []

        #  Team filters and datasets
        h_team_mask = df["Home"] == team
        a_team_mask = df["Away"] == team
        h_games = df.loc[h_team_mask].reset_index(drop=True)
        a_games = df.loc[a_team_mask].reset_index(drop=True)

        #  COUNT OF GAMES PLAYED FOR EACH CATEGORY
        fav_count = sum(h_games["H_Status"] == "Fav") + sum(
            a_games["A_Status"] == "Fav"
        )
        ud_count = sum(h_games["H_Status"] == "UD") + sum(a_games["A_Status"] == "UD")
        game_count = len(h_games.index) + len(a_games.index)

        #  TEAM % OF APPEARANCE AS
        fav_rate = fav_count / game_count
        ud_rate = ud_count / game_count

        #  WIN % AS FAVORITE
        h_fav_mask = (h_games["H_Status"] == "Fav") & (h_games["MOV"] > 0)
        a_fav_mask = (a_games["A_Status"] == "Fav") & (a_games["MOV"] < 0)
        h_fav_win = sum(h_fav_mask)
        a_fav_win = sum(a_fav_mask)
        fav_win = (h_fav_win + a_fav_win) / fav_count

        #  WIN % AS UNDERDOG
        h_ud_mask = (h_games["H_Status"] == "UD") & (h_games["MOV"] > 0)
        a_ud_mask = (a_games["A_Status"] == "UD") & (a_games["MOV"] < 0)
        h_ud_win = sum(h_ud_mask)
        a_ud_win = sum(a_ud_mask)
        ud_win = (h_ud_win + a_ud_win) / ud_count

        #  % of Team's Games where the spread is covered
        h_spread_mask = h_games["Spread_Outcome"] == 1
        a_spread_mask = a_games["Spread_Outcome"] == 1
        cover_pct = (sum(h_spread_mask) + sum(a_spread_mask)) / game_count

        #  % of Team's Games ending in Under/Over respectively
        under_mask = h_games["O/U_Outcome"] == "Under"
        over_mask = a_games["O/U_Outcome"] == "Over"
        under_pct = (sum(under_mask) + (len(over_mask) - sum(over_mask))) / game_count
        over_pct = (sum(over_mask) + (len(under_mask) - sum(under_mask))) / game_count

        #  Team Upset Defense and Upset Offense
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

    final_data = round(final_data,3)
    final_data.to_sql("odds_stats", utils.engine, if_exists="replace", index=False)


@utils.timerun
def clean_train():
    """
    Cleans the raw scraped file for training the model
    Adds Massey/Elo Ratings to Raw Data
    """

    data = pd.read_csv("Train.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data["Outcome"] = np.where(data["H-Pts"] > data["A-Pts"], 1, 0)

    data.drop(
        ["Unnamed: 0", "Time", "A-Pts", "H-Pts", "OT", "SeasonID"], axis=1, inplace=True
    )

    data = add_massey(data)
    data = add_elo(data)
    data.to_csv("Train_Ready.csv", index=None)


def set_extras(data):
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
    massey = pd.read_sql_table("current_massey", utils.engine)
    team = pd.read_sql_table("team_stats", utils.engine)
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
            "Off",
            "Def",
            "Net",
            "PIE",
            "FG_PCT",
            "FG3_PCT",
            "TS_PCT",
        ]
    ]

    combined = combined.rename(
        columns={
            "TS_PCT": "TS%",
            "Off": "OFF",
            "Def": "DEF",
            "Net": "NET",
            "OREB": "ORB",
            "DREB": "DRB",
            "FG_PCT": "FG%",
            "FG3_PCT": "FG3%",
        }
    )

    combined["Massey"] = round(combined["Massey"], 2)
    combined.to_sql("all_stats", utils.engine, if_exists="replace", index=False)


if __name__ == "__main__":
    # combine_datasets()
    # clean_train()
    data = combine_odds_dataset()
    initial_odds_stats(data)
