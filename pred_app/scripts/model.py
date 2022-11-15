"""
This script contains functions for building and testing models
"""
import pickle
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_recall_curve,
    mean_squared_error,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
from scripts import utils, const, dicts


FILE_NAME = "xgb_model.sav"
DEF_CLASSIFIER = XGBClassifier(num_class=2)

PARAMS = {
    "max_depth": 3,
    "min_child_weight": 5,
    "eta": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "num_class": 2,
}

EPOCHS = 918


@utils.timerun
def test_model(
    training_data: pd.DataFrame,
    target: pd.Series,
    cv_count: int = 5,
    epochs: int = 918,
    params: dict = PARAMS,
) -> tuple[list, list, list, list]:
    #pylint: disable=too-many-locals
    """
    Purpose
    ----------
    Builds DMatrixs for model testing
    x_matrix = training data and training targets
    y_matrix = testing data and testing targets

    Trains model on full dataset (x_matrix)
    Calls prediction function on test dataset (y_matrix)
    Scores metrics for model
    Repeats tests and training x num of times (cv_count)
    Builds metrics table based on cross validation metric scores

    Parameters
    ----------
    training_data:
        DataFrame containing feature filtered training data

    target:
        DataFrame containing training prediction targets (Outcome/Spread)

    cv_count:
        Integer denoting how many cross validation tests to run

    test_targets:
       DataFrame containing testing prediction targets (Outcome/Spread)

    epochs:
        Number of estimators/trees for the model to use

    params:
        Parameter grid to set the XGB.Booster hyperparameters

    Returns
    ----------
    Returns tuple:
        x_train:
            Training dataset

        y_train:
            Training dataset target

        y_test:
            Testing dataset target

        outcomes:
            Predicted testing dataset targets
    """
    metrics_list = []

    for i in range(cv_count):
        outcomes = []

        x_train, x_test, y_train, y_test = train_test_split(
            training_data, target, test_size=0.15
        )

        x_matrix = xgb.DMatrix(x_train, label=y_train)
        y_matrix = xgb.DMatrix(x_test, label=y_test)

        xgb_model = xgb.train(params, x_matrix, epochs)
        preds = xgb_model.predict(y_matrix)

        for pred in preds:
            outcomes.append(np.argmax(pred))

        print(f"      {i+1} of {cv_count} Complete     ")

        arr = get_metrics(y_test, outcomes)
        metrics_list.append(arr)

    build_metric_table(metrics_list, True)

    return x_train, y_train, y_test, outcomes


def get_metrics(actual: list, prediction: np.ndarray) -> list:
    """
    Uses actual outcomes and predicted outcomes to score test metrics
    """
    arr = []

    combined = pd.DataFrame(dict(actual=actual, prediction=prediction))
    crosstab = pd.crosstab(index=combined["actual"], columns=combined["prediction"])

    precision = round(precision_score(actual, prediction), 4) * 100
    accuracy = round(accuracy_score(actual, prediction), 4) * 100
    logloss = round(log_loss(actual, prediction), 4)
    roc = round(roc_auc_score(actual, prediction), 4) * 100
    recall = round(recall_score(actual, prediction), 4) * 100
    correct = int(crosstab[0][0]) + int(crosstab[1][1])
    incorrect = int(crosstab[0][1]) + int(crosstab[1][0])
    game_count = len(prediction)

    print(f"{correct} Correct - {incorrect} Incorrect")
    print(f"Precision: {round(precision,4)}%")
    print(f"Accuracy:  {round(accuracy,4)}%")
    print(f"Logloss:   {round(logloss,4)}%")
    print("-----------------------------")

    arr.extend(
        [precision, recall, accuracy, logloss, roc, correct, incorrect, game_count]
    )

    return arr


@utils.timerun
def build_metric_table(metrics_data: list, testing: bool = True) -> pd.DataFrame:
    """
    Builds table of scoring metrics and commits to database
    """
    full_data = []

    table = pd.DataFrame(
        metrics_data,
        columns=[
            "Precision",
            "Recall",
            "Accuracy",
            "Logloss",
            "ROC",
            "Correct",
            "Incorrect",
            "Games Tested",
        ],
    )
    prec_mean = table["Precision"].agg(np.mean)
    acc_mean = table["Accuracy"].agg(np.mean)
    log_mean = table["Logloss"].agg(np.mean)

    print("      Score Averages     ")
    print(f"Precision: {round(prec_mean,2)}%")
    print(f"Accuracy:  {round(acc_mean,2)}%")
    print(f"Logloss:   {round(log_mean,2)}%")
    print("-----------------------------")

    for column in table:
        temp = []
        temp.extend(
            [
                table[column].agg(np.mean),
                table[column].agg(np.min),
                table[column].agg(np.max),
                table[column].agg(np.std),
            ]
        )
        full_data.append(temp)

    table = pd.DataFrame(
        full_data, columns=["Mean", "Min", "Max", "Std"], index=table.columns
    )

    table["Metric"] = [
        "Precision",
        "Recall",
        "Accuracy",
        "Logloss",
        "ROC-AUC",
        "Correct",
        "Incorrect",
        "Games Tested",
    ]
    table = table[["Metric", "Mean", "Min", "Max", "Std"]]

    if testing is False:
        table.to_sql(
            "metric_scores", const.ENGINE, if_exists="replace", index=table.columns
        )

    print(table)

    return table


def save_model(model: xgb.Booster) -> None:
    """
    Saves trained model
    """
    pickle.dump(model, open("model.pkl", "wb"))


def load_model() -> xgb.Booster:
    """
    Loads trained model
    """
    model = pickle.load(open("model.pkl", "rb"))

    return model


@utils.timerun
def feature_scoring(x_train: list, y_train: list, testing: bool = True) -> pd.DataFrame:
    """
    Test Feature Importances and returns DataFrame of scores
    """
    best = SelectKBest(score_func=f_classif, k="all")
    fit = best.fit(x_train, y_train)

    temp = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(x_train.columns)

    scores = pd.concat([temp, columns], axis=1)
    scores.columns = ["Specs", "Score"]
    scores = scores.sort_values(["Specs"], ascending=False).reset_index(drop=True)

    if testing is False:
        scores.to_sql("feature_scores", const.ENGINE, if_exists="replace", index=False)

    print(scores)

    return scores


@utils.timerun
def hyperparameter_tuning(x_train: list, y_train: list, testing: bool = True) -> None:
    """
    Tests hyperparameters based on a narrow grid of options randomly to find the best combinations
    Prints breakdown of hyperparameter scoring
    """
    parameters = ["param_" + params for params in dicts.xgb_narrow_grid] + [
        "mean_test_score"
    ]

    rs_model = RandomizedSearchCV(
        estimator=DEF_CLASSIFIER,
        param_distributions=dicts.xgb_narrow_grid,
        cv=3,
        verbose=10,
        n_jobs=-1,
        return_train_score=False,
        n_iter=128,
    )

    rs_model.fit(x_train, y_train)
    result = pd.DataFrame(rs_model.cv_results_)[parameters]
    result = result.sort_values(by=["mean_test_score"], ascending=False)

    parameters.remove("mean_test_score")

    for parameter in parameters:
        temp1 = result.groupby(parameter)["mean_test_score"].agg(np.mean).reset_index()
        temp1 = temp1.sort_values(by=["mean_test_score"], ascending=False)
        print("\n", temp1)

    print(f"Best score: {rs_model.best_score_}")
    print(f"Best params: {rs_model.best_params_}")
    print(f"Best estimator: {rs_model.best_estimator_}")

    if testing is False:
        result.to_sql("hyper_scores", const.ENGINE, if_exists="replace", index=False)


@utils.timerun
def plot_roc_curve(y_test: list, preds: list) -> None:
    """
    Plots and saves a ROC-AUC plot image
    """
    fpr, tpr, dummy = roc_curve(y_test, preds)
    plt.plot(fpr, tpr, lw=2, color="royalblue", marker=".", label="PRED")
    plt.plot([0, 1], [0, 1], "--", color="firebrick", label="Baseline")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("ROC-AUC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.gcf().savefig("ROC_AUC_Curve.png", dpi=1200)
    plt.clf()


@utils.timerun
def plot_precision_recall(y_test: list, preds: list) -> None:
    """
    Plots and saves a Precision-Recall plot image
    """
    precision, recall, dummy = precision_recall_curve(y_test, preds)

    _, axis = plt.subplots()
    axis.plot(recall, precision, color="royalblue", label="PRED")
    axis.plot([0, 1], [0.01, 0.01], color="firebrick", linestyle="--", label="Baseline")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    axis.set_title("Precision-Recall Curve")
    axis.set_ylabel("Precision")
    axis.set_xlabel("Recall")
    plt.legend(loc="best")
    plt.gcf().savefig("Precision_Recall_Curve.png", dpi=1200)


@utils.timerun
def find_trees(
    train: pd.DataFrame,
    target: pd.Series,
    cv_folds: int = 5,
    early_stopping_rounds: int = 50,
    learning_rate: float = 0.01,
) -> int:
    """
    Finds the ideal number of trees for the model
    """

    xgb_model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=10000,
        max_depth=3,
        min_child_weight=5,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        nthread=4,
        num_class=2,
        seed=27,
    )

    xgb_param = xgb_model.get_xgb_params()
    x_matrix = xgb.DMatrix(train, label=target)

    cv_result = xgb.cv(
        xgb_param,
        x_matrix,
        num_boost_round=xgb_model.get_params()["n_estimators"],
        nfold=cv_folds,
        metrics="auc",
        early_stopping_rounds=early_stopping_rounds,
    )

    trees = cv_result.shape[0]

    return trees


@utils.timerun
def predict_season(season: str) -> list:
    """
    Performs a prediction of a given season and returns metrics
    """
    outcomes = []

    train_data = pd.read_sql_table("training_data", const.ENGINE)
    mask = train_data["A_Massey"] != 0
    train_data = train_data.loc[mask].reset_index(drop=True)

    season_mask = train_data["SeasonID"] == season
    x_train = train_data.drop(train_data[season_mask].index, axis=0)
    x_test = train_data[season_mask]
    y_train = x_train["Outcome"]
    y_test = x_test["Outcome"]

    y_train_mov = x_train['MOV']
    y_test_mov = x_test['MOV']

    x_train = x_train[const.NET_FULL_FEATURES]
    x_test = x_test[const.NET_FULL_FEATURES]

    x_matrix = xgb.DMatrix(x_train, label=y_train_mov)
    y_matrix = xgb.DMatrix(x_test, label=y_test_mov)

    xgb_model = xgb.train(PARAMS, x_matrix, EPOCHS)
    preds = xgb_model.predict(y_matrix)

    # for pred in preds:
    #     outcomes.append(np.argmax(pred))

    #     print(pred)

    mse = mean_squared_error(y_test_mov, preds)
    print(mse)

    #arr = get_metrics(y_test_mov, preds)
    #print(arr)

    return mse


if __name__ == "__main__":
    # data = pd.read_sql_table("training_data", const.ENGINE)
    # mask = data["A_Massey"] != 0
    # data = data.loc[mask].reset_index(drop=True)
    # outcome = data["Outcome"]
    # data = data[const.NET_FULL_FEATURES]

    # training, testing, actuals, predictions = test_model(data, outcome)
    # scores_table = feature_scoring(training, testing, False)
    # print(scores_table)
    # plot_roc_curve(actuals, predictions)
    # plot_precision_recall(actuals, predictions    

    # hyperparameter_tuning(training, testing, False)
    # trees = find_trees(data, outcome)
    # print(trees)

    predict_season("2019-20")