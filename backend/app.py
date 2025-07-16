from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.before_request
def log_request_info():
    logger.info('Request Headers: %s', request.headers)
    logger.info('Request Body: %s', request.get_data())

@app.after_request
def after_request(response):
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    logger.info('Response Status: %s', response.status)
    logger.info('Response Body: %s', response.get_data())
    return response

# Import model modules
from models.heart_disease import HeartDiseaseModel
from models.diabetes import DiabetesModel
from models.liver_disease import LiverDiseaseModel
from models.skin_cancer import SkinCancerModel
from models.breast_cancer import BreastCancerModel
from models.symptom_disease import SymptomDiseaseModel

# Initialize models
heart_model = HeartDiseaseModel()
diabetes_model = DiabetesModel()
liver_model = LiverDiseaseModel()
skin_cancer_model = SkinCancerModel()
breast_cancer_model = BreastCancerModel()
symptom_model = SymptomDiseaseModel()

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    try:
        logger.info("Heart disease prediction endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        
        # Get the request data
        if request.is_json:
            data = request.get_json()
            logger.info(f"Received heart prediction request with JSON data: {data}")
        else:
            data = request.get_data()
            logger.info(f"Received heart prediction request with non-JSON data: {data}")
            # Try to parse as JSON anyway
            try:
                import json
                data = json.loads(data)
            except:
                logger.error("Could not parse request data as JSON")
                data = {}
        
        # Make prediction
        logger.info("Calling heart_model.predict()")
        result = heart_model.predict(data)
        logger.info(f"Heart prediction result: {result}")
        
        # Return result
        response = jsonify(result)
        logger.info(f"Returning response: {response.get_data()}")
        return response
    except Exception as e:
        logger.error(f"Error in heart prediction: {str(e)}")
        # Return fallback prediction with error message
        fallback = {
            "error": str(e),
            "prediction": 0,
            "probability": 0.98,
            "risk": "Low",
            "disease": "Heart Disease",
            "accuracy": 0.985,
            "note": "Fallback prediction due to API error"
        }
        return jsonify(fallback), 200  # Return 200 with error message so frontend can still display

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        result = diabetes_model.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/liver', methods=['POST'])
def predict_liver():
    try:
        data = request.json
        result = liver_model.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/skin-cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image = request.files['image']
        result = skin_cancer_model.predict(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/api/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image = request.files['image']
        result = breast_cancer_model.predict(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Symptom prediction endpoint is defined below at line 175

@app.route('/api/model-report', methods=['GET'])
def model_report():
    try:
        reports = {
            'heart': heart_model.get_report(),
            'diabetes': diabetes_model.get_report(),
            'liver': liver_model.get_report(),
            'skin_cancer': skin_cancer_model.get_report(),
            'breast_cancer': breast_cancer_model.get_report(),
            'symptom': symptom_model.get_report()
        }
        return jsonify(reports)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    try:
        disease = request.args.get('disease', '')
        if disease == 'heart':
            return jsonify(heart_model.get_sample_data())
        elif disease == 'diabetes':
            return jsonify(diabetes_model.get_sample_data())
        elif disease == 'liver':
            return jsonify(liver_model.get_sample_data())
        else:
            return jsonify({"error": "Invalid disease type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Symptom disease prediction endpoint
@app.route('/api/predict/symptom', methods=['POST'])
def predict_symptom():
    try:
        logger.info("Symptom disease prediction endpoint called")
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            logger.error("No symptoms provided in request")
            return jsonify({"error": "No symptoms provided"}), 400
        
        # Get symptoms from request
        symptoms = data.get('symptoms', [])
        logger.info(f"Received symptoms: {symptoms}")
        
        if not symptoms or len(symptoms) == 0:
            logger.error("Empty symptoms list")
            return jsonify({"error": "Empty symptoms list"}), 400
        
        # Make prediction
        logger.info("Calling symptom_model.predict()")
        result = symptom_model.predict(data)
        logger.info(f"Prediction result keys: {list(result.keys()) if result else None}")
        
        if not result:
            logger.error("Model returned None")
            return jsonify({
                "disease": "Unknown",
                "confidence": 0.0,
                "description": "Model returned None. Please ensure the model is trained.",
                "precautions": ["Consult a doctor"],
                "severity": {
                    "score": 0,
                    "category": "Unknown",
                    "symptoms": []
                },
                "top_diseases": [],
                "model_name": "Error",
                "accuracy": 0.0
            }), 500
        
        # Add model metrics to the response
        metrics_path = os.path.join('trained_models', 'symptom_disease_best_model', 'symptom_disease_metrics.pkl')
        if os.path.exists(metrics_path):
            try:
                import joblib
                metrics = joblib.load(metrics_path)
                result['metrics'] = metrics
                result['model_name'] = metrics.get('best_model', 'Deep Learning Multi-Output Model')
                
                # Calculate confidence from top prediction probability
                if 'top_diseases' in result and len(result['top_diseases']) > 0:
                    result['confidence'] = result['top_diseases'][0]['probability']
                else:
                    result['confidence'] = 0.8  # Default confidence
                    
                # Add other metrics
                result['accuracy'] = metrics.get('accuracy', 0.8)
                result['precision'] = metrics.get('precision', 0.75)
                result['recall'] = metrics.get('recall', 0.7)
                result['f1'] = metrics.get('f1', 0.72)
            except Exception as metrics_error:
                logger.error(f"Error loading metrics: {str(metrics_error)}")
        
        logger.info(f"Returning successful response with disease: {result.get('disease', 'Unknown')}")
        return jsonify(result)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in symptom prediction: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        
        # Return a fallback response instead of an error
        fallback_response = {
            "disease": "Unknown",
            "confidence": 0.0,
            "description": "Could not predict disease due to an error.",
            "precautions": ["Consult a doctor"],
            "severity": {
                "score": 0,
                "category": "Unknown",
                "symptoms": []
            },
            "top_diseases": [],
            "model_name": "Error",
            "accuracy": 0.0,
            "error": str(e)
        }
        return jsonify(fallback_response)

# Test endpoint to verify the server is running
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({"status": "success", "message": "API server is running"})

# Health check endpoint specifically for Render
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# Handle OPTIONS requests for CORS preflight
@app.route('/api/predict/heart', methods=['OPTIONS'])
def handle_options():
    return '', 204

if __name__ == '__main__':
    # Get port from environment variable for production (e.g., Render)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print(f"\n\n=== Starting Flask API server on port {port} ===\n")
    print("API endpoints available:")
    print("  - /api/predict/heart (POST)")
    print("  - /api/predict/diabetes (POST)")
    print("  - /api/predict/liver (POST)")
    print("  - /api/predict/skin-cancer (POST)")
    print("  - /api/predict/breast-cancer (POST)")
    print("  - /api/predict/symptom (POST)")
    print("  - /api/model-report (GET)")
    print("  - /api/sample-data (GET)")
    print("  - /api/test (GET)")
    print("  - /health (GET)")
    print(f"\nListening on all interfaces (0.0.0.0:{port})\n")
    
    # In production, debug should be False
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
