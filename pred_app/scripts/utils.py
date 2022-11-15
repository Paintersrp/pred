"""
This script contains utility functions
"""
import time
import functools


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


if __name__ == "__main__":
    pass
