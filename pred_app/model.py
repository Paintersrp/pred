from sklearn.metrics import precision_score, accuracy_score, log_loss, roc_auc_score, roc_curve, recall_score, precision_recall_curve
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

PARAMS = {
    "max_depth": 4,
    "min_child_weight": 60,
    "eta": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "objective": "multi:softprob",
    "num_class": 2,
}

EPOCHS = 5000


@utils.timerun
def build_model(data, outcome, feat_scoring=False, cv_count = 5):
    """
    Builds/trains a model with crossfold validation
    Saves newly trained model and returns scoring metrics
    """
    metric_arr = []
    full_array = []

    for i in range(cv_count):
        arr = []
        outcomes = []

        X_train, X_test, y_train, y_test = train_test_split(
            data, outcome, test_size=0.15
        )

        X = xgb.DMatrix(X_train, label=y_train)
        y = xgb.DMatrix(X_test, label=y_test)

        xgb_model = xgb.train(PARAMS, X, EPOCHS)
        preds = xgb_model.predict(y)

        for pred in preds:
            outcomes.append(np.argmax(pred))

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
        metric_arr.append(arr)

        print(f"      {i+1} of {cv_count} Complete     ")
        print(f"{int(crosstab[0][0]) + int(crosstab[1][1])} Correct - {int(crosstab[0][1]) + int(crosstab[1][0])} Incorrect")
        print(f"Precision: {round(precision,4)}%")
        print(f"Accuracy:  {round(accuracy,4)}%")
        print(f"Logloss:   {round(logloss,4)}%")
        print("-----------------------------")

    metric_table = pd.DataFrame(
        metric_arr,
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
    prec_mean = metric_table["Precision"].agg(np.mean)
    acc_mean = metric_table["Accuracy"].agg(np.mean)
    log_mean = metric_table["Logloss"].agg(np.mean)

    print(f"      Score Averages     ")
    print(f"Precision: {round(prec_mean,2)}%")
    print(f"Accuracy:  {round(acc_mean,2)}%")
    print(f"Logloss:   {round(log_mean,2)}%")
    print("-----------------------------")

    for column in metric_table:
        temp = []
        temp.extend(
            [
                metric_table[column].agg(np.mean),
                metric_table[column].agg(np.min),
                metric_table[column].agg(np.max),
                metric_table[column].agg(np.std),
            ]
        )
        full_array.append(temp)

    metric_table = pd.DataFrame(
        full_array, columns=["Mean", "Min", "Max", "Std"], index=metric_table.columns
    )
    print(metric_table)

    X = xgb.DMatrix(data, label=outcome)
    xgb_model = xgb.train(PARAMS, X, EPOCHS)

    pickle.dump(xgb_model, open(FILE_NAME, "wb"))

    if feat_scoring == True:
        scores = feature_scoring(X_train, y_train)
        scores.to_sql("feature_scores", utils.engine, if_exists="replace", index=False)

    metric_table["Metric"] = [
        "Precision",
        "Recall",
        "Accuracy",
        "Logloss",
        "ROC-AUC",
        "Correct",
        "Incorrect",
        "Games Tested",
    ]
    metric_table = metric_table[["Metric", "Mean", "Min", "Max", "Std"]]
    metric_table.to_sql(
        "metric_scores", utils.engine, if_exists="replace", index=metric_table.columns
    )
    plot_roc_curve(y_test, outcomes)
    plot_precision_recall(y_test, outcomes)

    return xgb_model, X_train, y_train


def feature_scoring(X, y):
    """
    Test Feature Importances and returns DataFrame of Scores
    """
    best = SelectKBest(score_func=f_classif, k="all")
    fit = best.fit(X, y)

    temp = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)

    scores = pd.concat([temp, columns], axis=1)
    scores.columns = ["Specs", "Score"]
    scores = scores.sort_values(["Specs"], ascending=False).reset_index(drop=True)
    print(scores)

    return scores


@utils.timerun
def hyperparameter_tuning(X, y, prints=False):
    """
    Tests hyperparameters based on a narrow grid of options randomly to find the best combinations
    Prints breakdown of hyperparameter scoring
    """
    parameters = ["param_" + params for params in utils.xgb_narrow_grid] + [
        "mean_test_score"
    ]
    xgb_model = xgb.XGBClassifier(num_class=2)

    rs_model = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=utils.xgb_narrow_grid,
        cv=3,
        verbose=10,
        n_jobs=-1,
        return_train_score=False,
        n_iter=48,
    )

    rs_model.fit(X, y)
    result = pd.DataFrame(rs_model.cv_results_)[parameters]
    result = result.sort_values(by=["mean_test_score"], ascending=False)

    for key, items in utils.xgb_narrow_grid.items():
        i = 0

        for item in items:
            count = result.loc[result[f"param_{key}"] == item].count()[i]
            mean = result.loc[result[f"param_{key}"] == item]
            mean = mean["mean_test_score"].mean()

            if prints == True:
                print(f"{key}: {item}")
                print(f"Count: {count}")
                print(f"Mean: {round(mean,6)} \n")

        i += 1

    parameters.remove("mean_test_score")

    for parameter in parameters:
        temp1 = result.groupby(parameter)["mean_test_score"].agg(np.mean).reset_index()
        temp1 = temp1.sort_values(by=["mean_test_score"], ascending=False)

        if prints == True:
            print("\n", temp1)

    if prints == True:
        print(f"Best score: {rs_model.best_score_}")
        print(f"Best params: {rs_model.best_params_}")
        print(f"Best estimator: {rs_model.best_estimator_}")

    result.to_sql("hyper_scores", utils.engine, if_exists="replace", index=False)
    # result.to_csv('HyperMeanScores.csv')


def plot_roc_curve(y_test, preds):
    fpr, tpr, temp = roc_curve(y_test, preds)
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


def plot_precision_recall(y_test, preds):
    precision, recall, th = precision_recall_curve(y_test, preds)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="royalblue", label="PRED")
    ax.plot([0, 1], [0.01, 0.01], color="firebrick", linestyle="--", label="Baseline")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_title("Precision-Recall Curve")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    plt.legend(loc="best")
    plt.gcf().savefig("Precision_Recall_Curve.png", dpi=1200)


@utils.timerun
def find_trees(
    classifier, train, features, target, cv_folds=5, early_stopping_rounds=50
):
    xgb_param = classifier.get_xgb_params()
    xgtrain = xgb.DMatrix(train[features], label=target)

    cvresult = xgb.cv(
        xgb_param,
        xgtrain,
        num_boost_round=classifier.get_params()["n_estimators"],
        nfold=cv_folds,
        metrics="auc",
        early_stopping_rounds=early_stopping_rounds,
    )

    print(cvresult.shape[0])


if __name__ == "__main__":
    data = pd.read_csv("Train_Ready.csv")
    mask = data["A_Massey"] != 0
    data = data.loc[mask].reset_index(drop=True)

    outcome = data["Outcome"]

    data.drop(
        [
            "Outcome",
            "Date",
            "Home",
            "Away",
            "MOV",
            "A_W",
            "H_W",
            "A_L",
            "H_L",
            "A_W_PCT",
            "H_W_PCT",
            "A_PLUS_MINUS",
            "H_PLUS_MINUS",
        ],
        axis=1,
        inplace=True,
    )
    data = data[data.columns.drop(list(data.filter(regex="_RANK")))]

    # data = data[['A_Massey',
    #              'H_Massey',
    #               'H_NET_RATING',
    #               'A_NET_RATING',
    #               'A_PIE',
    #               'H_PIE',
    #               'A_TS_PCT',
    #               'H_TS_PCT',
    #               'A_FGM',
    #               'H_FGM',
    #               'A_DREB',
    #               'H_DREB']]

    xgb_model, X, y = build_model(data, outcome, True)
    # hyperparameter_tuning(X, y, prints = True)

    # xgb1 = XGBClassifier(
    #     learning_rate =0.01,
    #     n_estimators=5000,
    #     max_depth=4,
    #     min_child_weight=60,
    #     gamma=0,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective= 'multi:softmax',
    #     nthread=4,
    #     num_class=2,
    #     seed=27)

    # find_trees(xgb1, data, features, outcome)
