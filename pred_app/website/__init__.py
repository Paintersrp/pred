from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .views import views
from .auth import auth

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, static_folder='static')
    app.config['SECRET_KEY'] = 'DqwE'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pred.db'
    db.init_app(app)

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app