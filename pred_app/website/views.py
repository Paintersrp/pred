"""
Docstring
"""
from flask import Blueprint, render_template
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///pred.db")
views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home_page():
    """
    Home Funcstring
    """
    preds = pd.read_sql_table("today_preds", engine)
    preds.columns = [
        "Win%",
        "Net",
        "Massey",
        "Odds",
        "Away Team",
        "Time",
        "Home Team",
        "Odds",
        "Massey",
        "Net",
        "Win%",
    ]

    metrics = pd.read_sql_table("metric_scores", engine)

    return render_template(
        "home.html",
        pred_table=preds,
        metrics_table=[
            metrics.to_html(
                classes="table table-hover table-striped table-sm table-dark"
            )
        ],
        metric_headers=[""],
    )
