"""
This module contains Betting Odds Calculator Classes and Methods
"""
import typing as t
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
        typeCheck: int = 1,
    ) -> None:
        self.decimal = False
        self.moneyline = False

        if typeCheck == 1:
            self.moneyline = True
        elif typeCheck == 2:
            self.decimal = True

    def payout(self, wager: float, odds: int) -> float:
        """Func"""

        if self.moneyline:
            if odds > 0:
                multiplier = (odds / 100) + 1
            else:
                multiplier = (100 / abs(odds)) + 1

            final = wager * multiplier

        elif self.decimal:
            final = wager * odds

        return final

    def convert_odds(self, odds: int) -> tuple[float, int, float]:
        """Func"""

        if self.moneyline:
            if odds > 0:
                print(odds)
                american_final = odds
                decimal_final = (odds / 100) + 1
                implied_final = 100 / (odds + 100) * 100
            else:
                american_final = odds
                decimal_final = (100 / abs(odds)) + 1
                implied_final = abs(odds) / (abs(odds) + 100) * 100

            return (
                round(decimal_final, 2),
                american_final,
                round(implied_final, 2),
            )

        elif self.decimal:
            american_final = (odds - 1) * 100
            decimal_final = odds
            implied_final = (1 / odds) * 100

            return (
                round(decimal_final, 2),
                american_final,
                round(implied_final, 2),
            )

    def calc_hedge(
        self, original_wager: float, original_odds: t.Any, hedge_odds: t.Any
    ) -> tuple[float, float, float, float, float, float]:
        """
        Calculates prevent loss wager, maximize hedge wager, and guaranteed return of hedge
        """

        if self.moneyline:
            if hedge_odds > 0:
                hedge_odds = (hedge_odds / 100) + 1
                break_even = original_wager / (hedge_odds - 1)
                break_even_payout = break_even * hedge_odds

            else:
                hedge_odds = (100 / abs(hedge_odds)) + 1
                break_even = original_wager / (hedge_odds - 1)
                break_even_payout = break_even * hedge_odds

            if original_odds > 0:
                original_odds = (original_odds / 100) + 1
            else:
                original_odds = (100 / abs(original_odds)) + 1

            original_payout = original_wager * original_odds
            equal_return = (original_wager * original_odds) / hedge_odds
            equal_return_payout = equal_return * hedge_odds
            equal_return_profit = equal_return_payout - (original_wager + equal_return)

        else:
            break_even = original_wager / (hedge_odds - 1)
            break_even_payout = break_even * hedge_odds
            original_payout = original_wager * original_odds
            equal_return = (original_wager * original_odds) / hedge_odds
            equal_return_payout = equal_return * hedge_odds
            equal_return_profit = equal_return_payout - (original_wager + equal_return)

        return (
            float(original_payout),
            float(break_even),
            float(break_even_payout),
            float(equal_return),
            float(equal_return_payout),
            float(equal_return_profit),
        )

    def calc_parlay(self, wager: float, betting_lines: list) -> float:
        """
        Calculates return of given Parlay lines
        """

        if self.moneyline:
            parlay_odds_value = self.convert_odds(betting_lines)

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

    def calc_probability(self, line_value: t.Any) -> None:
        """Calculates Implied Probability of given line"""

        if self.decimal:
            return (1 / line_value) * 100

        if self.moneyline:
            if line_value > 0:
                return 100 / (line_value + 100) * 100

            if line_value < 0:
                return abs(line_value) / (abs(line_value) + 100) * 100
