import pandas as pd
import numpy as np
import typing as t
from datetime import date
from scripts import const, dicts

class Handler:
    def print_data(self):
        print(self.data)

    def return_data(self) -> pd.DataFrame:
        return self.data

    def reset_filter(self):
        self.__init__()

    def upcoming_schedule(self) -> pd.DataFrame:
        """
        Updates daily with new upcoming schedule of unplayed games
        """
        upcoming_games = pd.read_sql_table("2023_upcoming_games", const.ENGINE)
        today_mask = upcoming_games["Date"] != str(date.today())
        upcoming_games = upcoming_games.loc[today_mask].reset_index(drop=True)
        upcoming_games.to_sql(
            f"upcoming_schedule_table", const.ENGINE, if_exists="replace", index=False
        )

        return upcoming_games


class TeamsHandler(Handler):
    def __init__(self):
        self.data = pd.read_sql_table("boxscore_data", const.ENGINE)

    def __map_months(self, year: int) -> list:
        return dicts.months_map.get(year, const.MONTHS_REG)

    def __map_season(self, year: int) -> list:
        return dicts.season_map[year]   

    def return_averages(self, team_name: str) -> t.Any:
        self.away_games = self.data.groupby("Away")
        self.home_games = self.data.groupby("Home")

        self.away_avgs = (
            self.away_games.get_group(team_name)[const.AWAY_FEATURES]
            .reset_index(drop=True)
            .astype(float)
        )

        self.home_avgs = (
            self.home_games.get_group(team_name)[const.HOME_FEATURES]
            .reset_index(drop=True)
            .astype(float)
        )

        final_avgs = (
            (self.away_avgs.agg(np.mean).values + self.home_avgs.agg(np.mean).values) / 2
        ).reshape(1, 14)

        final_avgs = pd.DataFrame(
            list(map(np.ravel, final_avgs)), columns=const.MATCHUP_FEATURES
        )

        final_avgs.insert(loc=0, column="Team", value=team_name)

        return round(final_avgs, 3)

    def filter_data_by_year(self, year: int):
        self.season = self.map_season(year)
        self.mask = self.data["SeasonID"] == self.season
        self.data = self.data.loc[self.mask].reset_index(drop=True)

    def filter_data_by_range(self, start_year: int = 2021, end_year: int = 2022):
        self.raw_data = []

        for year in range(start_year, end_year+1):
            self.season = self.map_season(year)
            self.mask = (self.data["SeasonID"] == self.season)
            self.temp = self.data.loc[self.mask].reset_index(drop=True)
            
            self.raw_data.append(self.temp)

        self.data = pd.concat(self.raw_data).reset_index(drop=True)

    def compare_teams_head_to_head(
        self, team_one: str, team_two: str
    ):
        self.mask = ((self.data["Away"] == team_one) & (self.data["Home"] == team_two)) | (
            (self.data["Away"] == team_two) & (self.data["Home"] == team_one)
        )

        self.data = self.data.loc[self.mask].reset_index(drop=True).tail(5)

        self.t1_final = self.return_averages(team_one)
        self.t2_final = self.return_averages(team_two)

        self.data = pd.concat([self.t1_final, self.t2_final]).reset_index(drop=True)

    def compare_team_avgs(self, home_team: str, away_team: str) -> pd.DataFrame:
        """
        Builds table of two given team's current averages
        """
        team_stats = pd.read_sql_table("team_stats", const.ENGINE)
        team_stats = team_stats.groupby("Team")

        h_stats = team_stats.get_group(home_team)
        a_stats = team_stats.get_group(away_team)

        data = pd.concat([h_stats, a_stats], axis=0, join="outer")
        data = data[const.TABLE_STATS_FULL]

        data = data.rename(
            columns={
                "TS_PCT": "TS%",
                "OFF_RATING": "OFF",
                "DEF_RATING": "DEF",
                "NET_RATING": "NET",
                "OREB": "ORB",
                "DREB": "DRB",
                "FG_PCT": "FG%",
                "FG3_PCT": "FG3%",
                "FT_PCT": "FT%",
                "EFG_PCT": "EFG%",
                "AST_PCT": "AST%",
            }
        )

        return data

    def build_compare_main(self, team_one: str, team_two: str) -> pd.DataFrame:
        """
        Loads odds stats and team stats from database
        Builds table of each teams combined team/odds stats
        Adds difference column highlighting different in stats
        """

        all_stats = pd.read_sql_table("all_stats", const.ENGINE)[
            const.COMPARE_COLS
        ].groupby("Team")

        odds_stats = pd.read_sql_table("odds_stats", const.ENGINE)[const.ODDS_COLS].groupby(
            "Team"
        )

        t1_full = np.concatenate(
            [all_stats.get_group(team_one), odds_stats.get_group(team_one)], axis=1
        )

        t1_full = pd.DataFrame(t1_full, columns=const.COMPARE_COLS + const.ODDS_COLS)
        t1_full = t1_full.loc[:, ~t1_full.columns.duplicated()]

        t2_full = np.concatenate(
            [all_stats.get_group(team_two), odds_stats.get_group(team_two)], axis=1
        )

        t2_full = pd.DataFrame(t2_full, columns=const.COMPARE_COLS + const.ODDS_COLS)
        t2_full = t2_full.loc[:, ~t2_full.columns.duplicated()]
        data = pd.concat([t1_full, t2_full], axis=0, join="outer").reset_index(drop=True)

        extra_stats = data[["Team", "Record"]]
        data.drop(["Team", "Record"], axis=1, inplace=True)
        diff_data = data.diff().dropna()

        temp = pd.concat([data, diff_data], axis=0, join="outer").reset_index(drop=True)
        final_data = pd.concat([extra_stats, temp], axis=1, join="outer").reset_index(
            drop=True
        )
        final_data.fillna("-", inplace=True)

        return final_data

    
class OddsHandler(Handler):
    def __init__(self):
        self.data = pd.read_sql_table("odds_stats", const.ENGINE)
        self.grouped = self.data.groupby("Team")

    def get_team(self, team_name) -> t.Any:
        return self.grouped.get_group(team_name)

    def compare_teams(
        self, team_one: str, team_two: str
    ):
        self.t1_final = self.grouped.get_group(team_one)
        self.t2_final = self.grouped.get_group(team_two)

        return pd.concat([self.t1_final, self.t2_final]).reset_index(drop=True)