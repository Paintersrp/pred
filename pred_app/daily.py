"""
This script contains functions for updating daily tables/running daily predictions
"""
import warnings
import scripts.const as const
from scripts.updater import Updater
from scripts.predictor import DailyPredictor
from scripts.handler import MetricsHandler
from scripts.ratings import current_elos
from pred_print import update_today_card_data, update_tomorrow_card_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    Updater = Updater()
    DailyPredictor = DailyPredictor()

    #  Updating data needed for building daily test dataset

    daily_team_stats, grouped_team_stats = Updater.update_team_stats(per_100=True)
    Updater.update_schedule()
    Updater.update_training_schedule()
    massey = Updater.update_massey()
    elos = current_elos()

    #  Daily Predictor handles building and predicting daily games

    DailyPredictor.build_test_data(grouped_team_stats, massey, elos)
    DailyPredictor.prepare_test_data()

    # metrics_list = DailyPredictor.test_model(cv_count=1, loud=True)
    # scores = DailyPredictor.feature_scoring()

    daily_preds, tomorrow_preds = DailyPredictor.predict()
    print(daily_preds)

    # DailyPredictor.plot_roc_curve()
    # DailyPredictor.plot_precision_recall()

    # Updater.update_feature_scores(scores)
    # Updater.update_metrics(metrics_list)

    Updater.update_history(daily_preds, const.NET_FULL_FEATURES)
    Updater.update_preds(daily_preds)
    Updater.update_elos(elos)

    # daily_lines = Updater.update_todays_lines()

    Updater.update_full_stats_per_100()
    Updater.update_history_outcomes()
    Updater.update_upcoming()
    daily_lineups = Updater.update_injuries()

    # update_today_card_data(daily_preds, daily_team_stats, daily_lineups, elos)
    # update_tomorrow_card_data(tomorrow_preds, daily_team_stats, elos)

    # Updater.update_boxscore_data()

    Updater.update_odds_current()
    Updater.update_odds_full()
    Updater.update_team_stats(per_100=False)

    # Updater.update_training_data()

    print(daily_preds)

    #  MetricsHandler handles returns/prints of Modelcd  Metric Data

    history_data = MetricsHandler().pred_history()
    print(sum(history_data.Outcome == 1))
    print(sum(history_data.Outcome == 0))
