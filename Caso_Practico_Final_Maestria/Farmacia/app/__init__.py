from flask import Flask

app = Flask(__name__)

from app import routes  # Importa routes después de definir app
