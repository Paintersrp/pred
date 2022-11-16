"""
This script contains functions for updating daily tables/running daily predictions
"""
import warnings
from scripts.transform import combine_dailies
from scripts.updater import Updater
from scripts.predictor import DailyPredictor


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    Updater = Updater()
    DailyPredictor = DailyPredictor()

    games = Updater.todays_games_json()
    daily_team_stats = Updater.team_stats()
    ratings = Updater.massey()    
    
    DailyPredictor.build_test_data(games, daily_team_stats, ratings)
    DailyPredictor.prepare_test_data()
    metrics_list = DailyPredictor.test_model(2)
    scores = DailyPredictor.feature_scoring()
    net_final = DailyPredictor.predict_today()
    DailyPredictor.plot_roc_curve()
    DailyPredictor.plot_precision_recall()

    Updater.commit_feature_scores(scores)
    Updater.commit_metrics(metrics_list)
    Updater.commit_history(net_final, DailyPredictor.features)
    Updater.commit_preds(net_final)
    Updater.commit_full_stats()

    # to be deleted/moved
    # games = get_todays_games(const.SCH_JSON_URL)
    # daily_team_stats = update_team_stats()
    # ratings = update_massey()

    
    # raw_data, predictors = build_pred_data(games, daily_team_stats, ratings)
    # todays_games, ph, team_names = split_pred_data(raw_data, predictors)
    # predictions = daily_pred(todays_games, ph, team_names)
    # Updater.build_metric_table(metrics_list, True)
