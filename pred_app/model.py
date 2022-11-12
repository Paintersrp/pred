"""
Docstring
"""
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_recall_curve,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import utils


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
def predict_season(season: str):
    """
    Funcstring
    """
    outcomes = []

    train_data = pd.read_sql_table("training_data", utils.ENGINE)
    mask = train_data["A_Massey"] != 0
    train_data = train_data.loc[mask].reset_index(drop=True)

    season_mask = train_data["SeasonID"] == season
    x_train = train_data.drop(train_data[season_mask].index, axis=0)
    x_test = train_data[season_mask]
    y_train = x_train["Outcome"]
    y_test = x_test["Outcome"]

    x_train = x_train[utils.NET_FULL_FEATURES]
    x_test = x_test[utils.NET_FULL_FEATURES]

    x_matrix = xgb.DMatrix(x_train, label=y_train)
    y_matrix = xgb.DMatrix(x_test, label=y_test)   

    xgb_model = xgb.train(PARAMS, x_matrix, EPOCHS)
    preds = xgb_model.predict(y_matrix)

    for pred in preds:
        outcomes.append(np.argmax(pred))

    combined = pd.DataFrame(dict(actual=y_test, prediction=outcomes))
    crosstab = pd.crosstab(index=combined["actual"], columns=combined["prediction"])

    precision = round(precision_score(y_test, outcomes), 4) * 100
    accuracy = round(accuracy_score(y_test, outcomes), 4) * 100
    logloss = round(log_loss(y_test, outcomes), 4)
    correct = int(crosstab[0][0]) + int(crosstab[1][1])
    incorrect = int(crosstab[0][1]) + int(crosstab[1][0])

    print(f"{correct} Correct - {incorrect} Incorrect")
    print(f"Precision: {round(precision,4)}%")
    print(f"Accuracy:  {round(accuracy,4)}%")
    print(f"Logloss:   {round(logloss,4)}%")
    print("-----------------------------")

@utils.timerun
def test_model(
    training_data: pd.DataFrame,
    target: pd.Series,
    cv_count: int = 5,
    epochs: int = 918,
    params: dict = PARAMS,
) -> tuple[list, list, list, list, list]:
    """
    Builds/trains a model with crossfold validation
    Saves newly trained model and returns scoring metrics
    """
    metrics_list = []

    for i in range(cv_count):
        arr = []
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


        # break this into a scoring function/metric function####################################################################
        combined = pd.DataFrame(dict(actual=y_test, prediction=outcomes))
        crosstab = pd.crosstab(index=combined["actual"], columns=combined["prediction"])

        precision = round(precision_score(y_test, outcomes), 4) * 100
        accuracy = round(accuracy_score(y_test, outcomes), 4) * 100
        logloss = round(log_loss(y_test, outcomes), 4)
        roc = round(roc_auc_score(y_test, outcomes), 4) * 100
        recall = round(recall_score(y_test, outcomes), 4) * 100
        correct = int(crosstab[0][0]) + int(crosstab[1][1])
        incorrect = int(crosstab[0][1]) + int(crosstab[1][0])
        game_count = len(outcomes)
        arr.extend(
            [precision, recall, accuracy, logloss, roc, correct, incorrect, game_count]
        )
        metrics_list.append(arr)

        print(f"      {i+1} of {cv_count} Complete     ")
        print(f"{correct} Correct - {incorrect} Incorrect")
        print(f"Precision: {round(precision,4)}%")
        print(f"Accuracy:  {round(accuracy,4)}%")
        print(f"Logloss:   {round(logloss,4)}%")
        print("-----------------------------")

    build_metric_table(metrics_list, True)

    return x_train, y_train, y_test, outcomes


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

    if testing == False:
        table.to_sql(
            "metric_scores", utils.ENGINE, if_exists="replace", index=table.columns
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
    Test Feature Importances and returns DataFrame of Scores
    """
    best = SelectKBest(score_func=f_classif, k="all")
    fit = best.fit(x_train, y_train)

    temp = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(x_train.columns)

    scores = pd.concat([temp, columns], axis=1)
    scores.columns = ["Specs", "Score"]
    scores = scores.sort_values(["Specs"], ascending=False).reset_index(drop=True)

    if testing == False:
        scores.to_sql("feature_scores", utils.ENGINE, if_exists="replace", index=False)

    return scores


@utils.timerun
def hyperparameter_tuning(x_train: list, y_train: list, testing: bool = True) -> None:
    """
    Tests hyperparameters based on a narrow grid of options randomly to find the best combinations
    Prints breakdown of hyperparameter scoring
    """
    parameters = ["param_" + params for params in utils.xgb_narrow_grid] + [
        "mean_test_score"
    ]

    rs_model = RandomizedSearchCV(
        estimator=DEF_CLASSIFIER,
        param_distributions=utils.xgb_narrow_grid,
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

    if testing == False:
        result.to_sql("hyper_scores", utils.ENGINE, if_exists="replace", index=False)


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
    learning_rate: float = 0.01
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


if __name__ == "__main__":
    # data = pd.read_sql_table("training_data", utils.ENGINE)
    # mask = data["A_Massey"] != 0
    # data = data.loc[mask].reset_index(drop=True)
    # outcome = data["Outcome"]

    # data = data[utils.NET_FULL_FEATURES]

    # training, testing, actuals, predictions = test_model(data, outcome)
    # scores_table = feature_scoring(training, testing, False)
    # print(scores_table)
    # plot_roc_curve(actuals, predictions)
    # plot_precision_recall(actuals, predictions)

    predict_season("2019-20")

    # hyperparameter_tuning(training, testing, False)
    # trees = find_trees(data, outcome)
    # print(trees)
