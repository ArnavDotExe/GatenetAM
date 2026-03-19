from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.views import views
    app.register_blueprint(views, url_prefix='/')

    from app.pipeline.pipeline import pipeline_api
    app.register_blueprint(pipeline_api,  url_prefix="/api")

    return app
