from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import logging
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
import sys
import base64

# Update the path to point to the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.polyPsych.open_end import OpenEndedAnalysis
from scripts.polyPsych.clustering import DataAnalysisApp
from scripts.polyPsych.cronbach import cronbach_alpha, CronbachResults
from python.scripts.polyPsych.coin_flip import flip_coin, alter_fact, DEFAULT_FACTS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure export folder
EXPORT_FOLDER = 'exports'
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify({'error': str(error)})
    response.status_code = error.status_code
    return response

def emit_log(message, level='info'):
    """Emit log message through WebSocket"""
    socketio.emit('log', {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message
    })

def save_uploaded_file(file):
    """Save uploaded file and return path"""
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

@app.route('/api/open-ended/analyze', methods=['POST'])
def analyze_open_ended():
    try:
        if 'file' not in request.files:
            raise APIError('No file provided')
        
        file = request.files['file']
        filepath = save_uploaded_file(file)
        
        analyzer = OpenEndedAnalysis()
        emit_log('Starting open-ended analysis')
        
        # Load and analyze data
        analyzer.load_csv_data(filepath)
        results = analyzer.analyze_patient_responses()
        
        if not results:
            raise APIError('Analysis failed')
        
        # Create visualizations and convert to base64
        visualizations = []
        analyzer.create_visualizations(results)
        viz_folder = os.path.join(EXPORT_FOLDER, max(os.listdir(EXPORT_FOLDER)))
        for viz_file in os.listdir(os.path.join(viz_folder, 'visualizations')):
            with open(os.path.join(viz_folder, 'visualizations', viz_file), 'rb') as f:
                viz_data = base64.b64encode(f.read()).decode('utf-8')
                visualizations.append(viz_data)
        
        emit_log('Analysis completed successfully')
        return jsonify({
            'results': results,
            'visualizations': visualizations
        })
        
    except Exception as e:
        logger.error(f"Error in open-ended analysis: {str(e)}", exc_info=True)
        emit_log(f"Error: {str(e)}", 'error')
        raise APIError(str(e))

@app.route('/api/clustering/analyze', methods=['POST'])
def analyze_clustering():
    try:
        if 'file' not in request.files:
            raise APIError('No file provided')
        
        file = request.files['file']
        filepath = save_uploaded_file(file)
        num_clusters = int(request.form.get('numClusters', 5))
        
        emit_log('Starting clustering analysis')
        
        # Initialize clustering analysis
        analysis = DataAnalysisApp(None)  # Pass None since we're not using GUI
        analysis.load_quant_data(filepath)
        results = analysis.cluster_data('kmeans', num_clusters)
        
        # Get visualizations
        visualizations = []
        analysis.visualize_results()
        if hasattr(analysis, 'last_fig'):
            import io
            buf = io.BytesIO()
            analysis.last_fig.savefig(buf, format='png')
            buf.seek(0)
            viz_data = base64.b64encode(buf.read()).decode('utf-8')
            visualizations.append(viz_data)
        
        emit_log('Clustering analysis completed successfully')
        return jsonify({
            'results': results,
            'visualizations': visualizations
        })
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}", exc_info=True)
        emit_log(f"Error: {str(e)}", 'error')
        raise APIError(str(e))

@app.route('/api/cronbach/analyze', methods=['POST'])
def analyze_cronbach():
    try:
        if 'file' not in request.files:
            raise APIError('No file provided')
        
        file = request.files['file']
        filepath = save_uploaded_file(file)
        confidence = float(request.form.get('confidenceLevel', 0.95))
        handle_missing = request.form.get('handleMissing', 'pairwise')
        
        emit_log('Starting Cronbach\'s alpha analysis')
        
        # Load data and calculate alpha
        df = pd.read_csv(filepath)
        results = cronbach_alpha(df, confidence, handle_missing)
        
        # Convert results to JSON-serializable format
        results_dict = {
            'alpha': float(results.alpha),
            'confidence_interval': [float(ci) for ci in results.confidence_interval],
            'std_error': float(results.std_error),
            'item_statistics': results.item_statistics.to_dict(),
            'scale_statistics': results.scale_statistics
        }
        
        emit_log('Cronbach\'s alpha analysis completed successfully')
        return jsonify({'results': results_dict})
        
    except Exception as e:
        logger.error(f"Error in Cronbach's alpha analysis: {str(e)}", exc_info=True)
        emit_log(f"Error: {str(e)}", 'error')
        raise APIError(str(e))

@app.route('/api/game/start', methods=['POST'])
def start_game():
    try:
        facts = DEFAULT_FACTS
        if 'customFacts' in request.files:
            file = request.files['customFacts']
            filepath = save_uploaded_file(file)
            with open(filepath, 'r') as f:
                facts = [line.strip() for line in f if line.strip()]
        
        current_fact = facts[0]
        is_true = flip_coin()
        if not is_true:
            current_fact = alter_fact(current_fact)
        
        return jsonify({
            'currentFact': current_fact,
            'remaining': len(facts) - 1,
            'isTrue': is_true
        })
        
    except Exception as e:
        logger.error(f"Error starting game: {str(e)}", exc_info=True)
        raise APIError(str(e))

@app.route('/api/game/answer', methods=['POST'])
def submit_answer():
    try:
        data = request.get_json()
        answer = data.get('answer')
        # Game logic here
        return jsonify({
            'correct': True,
            'nextFact': "Next fact...",
            'remaining': 5,
            'currentScore': 1
        })
    except Exception as e:
        logger.error(f"Error processing answer: {str(e)}", exc_info=True)
        raise APIError(str(e))

@app.route('/api/export', methods=['POST'])
def export_results():
    try:
        data = request.get_json()
        formats = data.get('formats', {})
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_folder = os.path.join(EXPORT_FOLDER, timestamp)
        os.makedirs(export_folder, exist_ok=True)
        
        exported_files = []
        
        # Export logic here based on formats
        
        return jsonify({
            'success': True,
            'files': exported_files
        })
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}", exc_info=True)
        raise APIError(str(e))

if __name__ == '__main__':
    socketio.run(app, debug=True) 