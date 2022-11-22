"""
This module contains Betting Odds Calculator Classes and Methods
"""
import typing as t
import re
import pandas as pd
import numpy as np


class Calculator:
    """
    Moneyline to Decimal Formula:
    ---------
    If Negative Moneyline:
        (100 / Negative Moneyline) +  1

    If Positive Moneyline:
        (Positive Moneyline / 100) +  1

    Fractional to Decimal Formula:
        (numerator / denominator) + 1
        (3/1) + 1
        = 4

    Implied Probability (Moneyline Formula):
    ---------
    If Negative Moneyline:
        Negative Moneyline / (Negative Moneyline + 100) * 100 = Implied Probability

    If Positive Moneyline:
        100 / (Positive Moneyline + 100) * 100 = Implied Probability


    Implied Probability (Decimal Formula):
    ---------
    ( 1 / Decimal Odds ) * 100 = Implied Probability


    Implied Probability (Fractional Formula)
    ---------
    denominator / (denominator + numerator) * 100 = Implied Probability

    Parlay Formula:
    ---------
    wager * (Decimal Odds(Game 1) * Decimal Odds(Game 3) * Decimal Odds(Game 3) * etc)

    Example:
        $50 * ( (Moneyline -120)    * (Moneyline +200)  * (Moneyline +150))
        $50 * ( ((100 / 120) + 1)   * ((200 / 100) + 1) * ((150 / 100) + 1) )
        $50 * (       1.83          *       3.0         *       2.5 )
        $50 * 13.725
        = $686.25 - $50 (Wager)
        = $636.25 (Profit)

    Hedge Formulas:
    ---------
    Minimum Bet Formula to Prevent Loss:
        prevent_loss = original_wager / hedge_odds
        x            = 100 / 1.5
        x = 66.667

    Hedge Wager Formula to Maximize Guaranteed Return:
        maximize_hedge = (original_profit + original_wager) / hedge_odds
        x              = (900 + 100) / 1.5
        x = 666.667

    Guaranteed Return Formula (with Maximized Hedge Wager):
        guaranteed_return = original_profit - maximize_hedge
        x                  = 900 - 666.667
        x = 233.33
    """

    def __init__(
        self,
        decimal: bool = False,
        fractional: bool = False,
        moneyline: bool = False,
    ) -> None:
        if decimal + fractional + moneyline > 1:
            raise ValueError(
                "Too many options enabled. Choose either decimal, fractional, or moneyline inputs"
            )

        if decimal + fractional + moneyline == 0:
            raise ValueError(
                "No options enabled. Choose either decimal, fractional, or moneyline inputs"
            )

        self.__check_parameters(decimal, fractional, moneyline)
        self.decimal = decimal
        self.fractional = fractional
        self.moneyline = moneyline

    def calc_probability(self, line_value: t.Any) -> float:
        """Calculates Implied Probability of given line"""

        self.__check_line_value(line_value)

        if self.decimal:
            return (1 / line_value) * 100

        if self.moneyline:
            if line_value > 0:
                return 100 / (line_value + 100) * 100

            if line_value < 0:
                return abs(line_value) / (abs(line_value) + 100) * 100

        if self.fractional:
            split_value = str(line_value).split("/")
            return (
                float(split_value[1])
                / (float(split_value[1]) + float(split_value[0]))
                * 100
            )

        return None

    def calc_parlay(self, wager: float, betting_lines: list) -> float:
        """
        Calculates return of given Parlay lines
        """

        self.__check_wager(wager)

        for line in betting_lines:
            self.__check_line_value(line)

        if self.moneyline:
            parlay_odds_value = self.__convert_moneyline(betting_lines)

        elif self.fractional:
            parlay_odds_value = self.__convert_fractional(betting_lines)

        else:
            parlay_odds_value = np.prod(betting_lines)

        return pd.DataFrame(
            [
                (wager * parlay_odds_value),
                (wager * parlay_odds_value) - wager,
                wager,
            ],
            index=["Total", "Profit", "Wager"],
        ).T

    def calc_hedge(
        self, original_wager: float, original_odds: t.Any, hedge_odds: t.Any
    ) -> tuple[float, float]:
        """
        Calculates prevent loss wager, maximize hedge wager, and guaranteed return of hedge
        """

        self.__check_wager(original_wager)
        self.__check_line_value(original_odds)
        self.__check_line_value(hedge_odds)

        if self.moneyline:
            if hedge_odds > 0:
                hedge_odds = (hedge_odds / 100) + 1
                prevent_loss = original_wager / hedge_odds

            else:
                hedge_odds = (100 / abs(hedge_odds)) + 1
                prevent_loss = original_wager / hedge_odds

        elif self.fractional:
            split_value = str(hedge_odds).split("/")
            hedge_odds = (float(split_value[0]) / float(split_value[1])) + 1
            prevent_loss = original_wager / hedge_odds

        else:
            prevent_loss = original_wager / hedge_odds

        if self.moneyline:
            if original_odds > 0:
                original_odds = (original_odds / 100) + 1
                original_profit = (original_wager * original_odds) - original_wager
                maximize_hedge = (original_profit + original_wager) / hedge_odds

            else:
                original_odds = (100 / abs(original_odds)) + 1
                original_profit = (original_wager * original_odds) - original_wager
                maximize_hedge = (original_profit + original_wager) / hedge_odds

        elif self.fractional:
            split_value = str(original_odds).split("/")
            original_odds = (float(split_value[0]) / float(split_value[1])) + 1
            original_profit = (original_wager * original_odds) - original_wager
            maximize_hedge = (original_profit + original_wager) / hedge_odds

        else:
            original_profit = (original_wager * original_odds) - original_wager
            maximize_hedge = (original_profit + original_wager) / hedge_odds

        guaranteed_return = original_profit - maximize_hedge

        return prevent_loss, maximize_hedge, guaranteed_return

    def change_mode(
        self, decimal: bool = False, fractional: bool = False, moneyline: bool = False
    ) -> None:
        """Changes metric system used for betting odds"""

        if decimal + fractional + moneyline > 1:
            raise ValueError(
                "Too many options enabled. Choose either decimal, fractional, or moneyline inputs"
            )

        if decimal + fractional + moneyline == 0:
            raise ValueError(
                "No options enabled. Choose either decimal, fractional, or moneyline inputs"
            )

        self.__check_parameters(decimal, fractional, moneyline)

        self.decimal = decimal
        self.fractional = fractional
        self.moneyline = moneyline

    def __convert_moneyline(self, line_values: list) -> float:
        """Converts moneyline values to decimal values"""

        converted_lines = []

        for line in line_values:
            if line > 0:
                converted_lines.append((line / 100) + 1)
            else:
                converted_lines.append((100 / abs(line)) + 1)

        converted_line_value = np.prod(converted_lines)

        return converted_line_value

    def __convert_fractional(self, line_values: list) -> float:
        """Converts fractional values to decimal values"""

        converted_lines = []

        for line in line_values:
            split_value = str(line).split("/")
            converted_lines.append((float(split_value[0]) / float(split_value[1])) + 1)

        converted_line_value = np.prod(converted_lines)

        return converted_line_value

    def __check_parameters(
        self, decimal: bool, fractional: bool, moneyline: bool
    ) -> None:
        """Mode parameter error checking and handling"""

        if isinstance(decimal, str):
            raise ValueError(
                f"Decimal parameter must be True or False, not a string. Type Received: {type(decimal)}"  # pylint: disable=line-too-long
            )

        if not isinstance(decimal, bool):
            raise ValueError(
                f"Decimal must be True or False. Value Received: {decimal}"
            )

        if isinstance(fractional, str):
            raise ValueError(
                f"Fractional parameter must be True or False, not a string. Type Received: {type(fractional)}"  # pylint: disable=line-too-long
            )

        if not isinstance(fractional, bool):
            raise ValueError(
                f"Fractional must be True or False. Value Received: {fractional}"
            )

        if isinstance(moneyline, str):
            raise ValueError(
                f"Moneyline parameter must be True or False, not a string. Type Received: {type(moneyline)}"  # pylint: disable=line-too-long
            )

        if not isinstance(moneyline, bool):
            raise ValueError(
                f"Moneyline must be True or False. Value Received: {moneyline}"
            )

    def __check_line_value(self, line_value: t.Any) -> None:
        """Bet line value(s) error checking and handling"""

        if self.fractional:
            if isinstance(line_value, str):
                if not bool(re.search(r"[.0-9]*/[.0-9]*", line_value)):
                    raise ValueError(
                        f"Fractional does not match required input. Example Input: '9/2'. Value Received: {line_value}"  #  pylint: disable=line-too-long
                    )
            else:
                raise ValueError(
                    f"Fractional Line Values should be a string - such as '9/2'. Type Received: {type(line_value)}"  #  pylint: disable=line-too-long
                )

        elif not isinstance(line_value, (int, float)):
            raise ValueError(
                f"Decimal and Moneyline Line Values should be numeric and not a string. Type Received: {type(line_value)}"  #  pylint: disable=line-too-long
            )

    def __check_wager(self, wager: float) -> None:
        """Wager value error checking and handling"""

        if not isinstance(wager, (int, float)):
            raise ValueError(
                f"Wager Values should be numeric and not a string. Type Received: {type(wager)}"  #  pylint: disable=line-too-long
            )

        if wager < 0:
            raise ValueError(
                f"Wager Values cannot be a negative value. Value Received: {wager}"  #  pylint: disable=line-too-long
            )
