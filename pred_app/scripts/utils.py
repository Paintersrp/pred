"""
This script contains utility functions
"""
import time
import functools
from scripts import dicts, const


def timerun(function):
    """
    Simple Function Timing Wrapper
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        time_1 = time.perf_counter()
        result = function(*args, **kwargs)
        time_2 = time.perf_counter()
        total = time_2 - time_1
        print(
            f"Function {function.__name__} Took: {total:.4f} seconds ({total/60:.4f} minutes)"
        )
        return result

    return wrapper


def map_months(year: int) -> list:
    """
    Returns season's months list for given year
    """

    return dicts.months_map.get(year, const.MONTHS_REG)


def map_season(year: int) -> list:
    """
    Returns season's months list for given year
    """

    return dicts.season_map[year]


def map_odds_team(team: str) -> list:
    """
    Returns season's months list for given year
    """

    return dicts.odds_team_dict[team]


def map_lines_team(team: str) -> list:
    """
    Returns season's months list for given year
    """

    return dicts.lines_team_dict[team]


if __name__ == "__main__":
    pass
