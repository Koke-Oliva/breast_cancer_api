# src/main.py
from flask import Flask
from src.api.routes import bp as api_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp)  # registrar rutas desde routes.py
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)
