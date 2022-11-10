from flask import Blueprint, render_template, request, flash
from nba_api.stats.endpoints import leaguedashteamstats as ldts
from website import views
from .views import engine
import pandas as pd

auth = Blueprint("auth", __name__)

@auth.route("/odds", methods=["GET", "POST"])
def odds():
    odds_stats = pd.read_sql_table('odds_stats', engine)
    odds_stats.index = odds_stats.index + 1
    odds_stats.rename(columns={"Team": "Team Name"}, inplace=True)

    return render_template(
        "odds.html",
        table=[
            odds_stats.to_html(
                table_id="data",
                classes="table table-hover table-striped table-sm table-dark",
            )
        ],
        headers=[""],
    )

@auth.route("/stats", methods=["GET", "POST"])
def league():
    all_stats = pd.read_sql_table('all_stats', engine)
    all_stats.index = all_stats.index + 1
    all_stats.rename(columns={"Team": "Team Name"}, inplace=True)

    return render_template(
        "stats.html",
        table=[
            all_stats.to_html(
                table_id="data",
                classes="table table-hover table-striped table-sm table-dark",
            )
        ],
        headers=[""],
    )

@auth.route("/stats/west", methods=["GET", "POST"])
def west():
    all_stats = pd.read_sql_table('all_stats', engine)
    all_stats.rename(columns={"Team": "Team Name"}, inplace=True)

    mask = (all_stats['Conf'] == 'West')
    west_stats = all_stats.loc[mask].reset_index(drop=True)
    west_stats.index = west_stats.index + 1

    return render_template(
        "stats_west.html",
        table=[
            west_stats.to_html(
                table_id="data",
                classes="table table-hover table-striped table-sm table-dark",
            )
        ],
        headers=[""],
    )

@auth.route("/stats/east", methods=["GET", "POST"])
def east():
    all_stats = pd.read_sql_table('all_stats', engine)
    all_stats.rename(columns={"Team": "Team Name"}, inplace=True)

    mask = (all_stats['Conf'] == 'East')
    east_stats = all_stats.loc[mask].reset_index(drop=True)
    east_stats.index = east_stats.index + 1

    return render_template(
        "stats_east.html",
        table=[
            east_stats.to_html(
                table_id="data",
                classes="table table-hover table-striped table-sm table-dark",
            )
        ],
        headers=[""],
    )


@auth.route("/compare")
def logout():
    return render_template("rankings.html")


@auth.route("/teams", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        first = request.form.get("firstName")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        if len(email) < 4:
            flash("Email must be greater than 4 characters.", category="fail")

        elif len(first) < 2:
            flash("First Name must be greater than 2 characters.", category="fail")

        elif password1 != password2:
            flash("Passwords do not match.", category="fail")

        elif len(password1) < 6:
            flash("Password must be at least 6 characters.", category="fail")

        else:
            flash("Account created.", category="success")

    return render_template("sign_up.html")


@auth.route("/about")
def about():
    return views.home_page()
