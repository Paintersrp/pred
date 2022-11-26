"""
This script contains functions for updating daily tables/running daily predictions
"""
import warnings
from scripts import const
from scripts.updater import Updater
from scripts.predictor import DailyPredictor
from scripts.handler import MetricsHandler

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    Updater = Updater()
    DailyPredictor = DailyPredictor()

    #  Updating data needed for building daily test dataset

    games = Updater.update_games_json()
    daily_team_stats = Updater.update_team_stats()
    Updater.update_schedule()
    ratings = Updater.update_massey()

    #  Daily Predictor handles building and predicting daily games

    DailyPredictor.build_test_data(games, daily_team_stats, ratings)
    DailyPredictor.prepare_test_data()
    metrics_list = DailyPredictor.test_model(cv_count=1, loud=True)
    scores = DailyPredictor.feature_scoring()
    net_final = DailyPredictor.predict_today()
    DailyPredictor.plot_roc_curve()
    DailyPredictor.plot_precision_recall()

    #  Updating Database

    Updater.update_feature_scores(scores)
    Updater.update_metrics(metrics_list)
    Updater.update_history(net_final, const.NET_FULL_FEATURES)
    Updater.update_preds(net_final)
    Updater.update_todays_lines()
    Updater.update_full_stats()
    Updater.update_history_outcomes()
    Updater.update_upcoming()
    Updater.update_injuries()
    Updater.update_boxscore_data()
    Updater.update_odds_current()
    Updater.update_odds_full()

    #  MetricsHandler handles returns/prints of Model Metric Data

    MetricsHandler = MetricsHandler()
    print(MetricsHandler.feature_scores())
    print(MetricsHandler.hyper_scores())
    print(MetricsHandler.metrics())
    print(MetricsHandler.today_preds())

    history_data = MetricsHandler.pred_history()
    print("\n")
    print(sum(history_data.Outcome == 1))
    print(sum(history_data.Outcome == 0))
