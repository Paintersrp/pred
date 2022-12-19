"""
This script contains functions for updating daily tables/running daily predictions
"""
import warnings
from scripts.const import NET_FULL_FEATURES
from scripts.updater import Updater
from scripts.predictor import DailyPredictor
from scripts.ratings import current_elos
from pred_print import update_today_card_data, update_tomorrow_card_data, update_odds

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    Updater = Updater()
    DailyPredictor = DailyPredictor()

    daily_team_stats, grouped_team_stats = Updater.update_team_stats(per_100=True)
    Updater.update_schedule()
    Updater.update_training_schedule()
    massey = Updater.update_massey()
    elos = current_elos()

    DailyPredictor.build_test_data(grouped_team_stats, massey, elos)
    DailyPredictor.prepare_test_data()
    daily_preds, tomorrow_preds = DailyPredictor.predict()

    Updater.update_history(daily_preds, NET_FULL_FEATURES)
    Updater.update_preds(daily_preds)
    Updater.update_elos(elos)
    Updater.update_full_stats_per_100()
    Updater.update_history_outcomes()
    Updater.update_upcoming()
    daily_lineups = Updater.update_injuries()
    update_today_card_data(daily_preds, daily_team_stats, daily_lineups, elos)
    update_tomorrow_card_data(tomorrow_preds, daily_team_stats, elos)
    Updater.update_odds_current()
    Updater.update_odds_full()
    Updater.update_team_stats(per_100=False)

    print(daily_preds)

    # Model Metric Updates - Unused for Now until Component/Page Implemented

    # metrics_list = DailyPredictor.test_model(cv_count=1, loud=True)
    # scores = DailyPredictor.feature_scoring()

    # DailyPredictor.plot_roc_curve()
    # DailyPredictor.plot_precision_recall()
    # Updater.update_feature_scores(scores)
    # Updater.update_metrics(metrics_list)
