import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from flask import Flask, send_from_directory
import os
from api.app import app as api_app

app = Flask(__name__)

# Mount the API app under /api
app.register_blueprint(api_app, url_prefix='/api')

# Serve static files from root directory
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True) 