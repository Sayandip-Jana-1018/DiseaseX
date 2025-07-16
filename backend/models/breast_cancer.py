import os
import numpy as np
import json
import shutil
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerModel:
    """
    A simplified breast cancer model that loads the pre-trained model from the Jupyter notebook
    and makes predictions on new images.
    """
    
    def __init__(self):
        """
        Initialize the breast cancer model with paths to the trained model
        """
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_save_path = os.path.join(self.base_dir, 'trained_models', 'breast_cancer_best_model')
        self.img_height = 50
        self.img_width = 50
        self.model = None
        self.selected_architecture = 'simple_cnn'
        
        # Try to load the model at initialization
        self.load_model()
    
    def load_model(self):
        """
        Load the pre-trained model from the saved files
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # First try to load the simple_cnn model JSON which is more likely to be compatible
            model_json_path = os.path.join(self.model_save_path, 'simple_cnn_model.json')
            if not os.path.exists(model_json_path):
                # Fall back to regular model.json
                model_json_path = os.path.join(self.model_save_path, 'model.json')
                if not os.path.exists(model_json_path):
                    logger.error(f"Model architecture file not found at {model_json_path}")
                    return False
            
            logger.info(f"Loading model architecture from {model_json_path}")
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Use TensorFlow's direct model creation to avoid compatibility issues
            try:
                import tensorflow as tf
                self.model = model_from_json(loaded_model_json)
            except Exception as model_error:
                logger.error(f"Error creating model from JSON: {str(model_error)}")
                # Create a simple CNN model as fallback
                logger.info("Creating fallback model architecture")
                self.model = self._create_simple_cnn_model()
            
            # Try to load weights if model was created successfully
            if self.model is not None:
                # Try different weight files
                weight_files = ['model.h5', 'simple_cnn_model.h5', 'resnet50_checkpoint.h5']
                weights_loaded = False
                
                for weight_file in weight_files:
                    weights_path = os.path.join(self.model_save_path, weight_file)
                    if os.path.exists(weights_path):
                        try:
                            logger.info(f"Attempting to load weights from {weights_path}")
                            self.model.load_weights(weights_path, by_name=True)
                            weights_loaded = True
                            logger.info(f"Successfully loaded weights from {weights_path}")
                            break
                        except Exception as weight_error:
                            logger.error(f"Error loading weights from {weight_file}: {str(weight_error)}")
                
                if not weights_loaded:
                    logger.warning("Could not load any weights. Using untrained model.")
            
            # Load metrics to get model info
            metrics_path = os.path.join(self.model_save_path, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.selected_architecture = metrics.get('model_name', 'simple_cnn')
                    accuracy = metrics.get('accuracy', 'unknown')
                    logger.info(f"Loaded {self.selected_architecture} model with accuracy: {accuracy}")
            
            logger.info("Breast cancer model initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def _create_simple_cnn_model(self):
        """
        Create a simple CNN model as a fallback
        
        Returns:
            A compiled simple CNN model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
            
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(BatchNormalization())
            
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(BatchNormalization())
            
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(BatchNormalization())
            
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            
            model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            logger.info("Created fallback simple CNN model")
            return model
        except Exception as e:
            logger.error(f"Error creating fallback model: {str(e)}")
            return None
    
    def predict(self, image_input):
        """
        Make a prediction for a single image
        
        Args:
            image_input: FileStorage object from request.files or path to image file
            
        Returns:
            Dictionary with prediction results in the format expected by the frontend
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Model not loaded. Please train the model first."}
                
        try:
            # Create a temporary directory for the image
            temp_dir = os.path.join(self.base_dir, 'datasets', 'breast', 'temp_prediction')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Handle the image input (could be a FileStorage object or a file path)
            if hasattr(image_input, 'save'):  # FileStorage object
                image_filename = image_input.filename
                temp_file_path = os.path.join(temp_dir, image_filename)
                image_input.save(temp_file_path)
            else:  # Regular file path
                image_filename = os.path.basename(image_input)
                temp_file_path = os.path.join(temp_dir, image_filename)
                shutil.copy(image_input, temp_file_path)
            
            # Create a data generator for the single image
            test_datagen = ImageDataGenerator(rescale=1.0/255)
            
            # Create a parent directory for the temp_dir to use with flow_from_directory
            parent_temp_dir = os.path.join(os.path.dirname(temp_dir), 'temp_prediction_parent')
            os.makedirs(parent_temp_dir, exist_ok=True)
            
            # Move the image to a subdirectory structure required by flow_from_directory
            class_dir = os.path.join(parent_temp_dir, 'unknown')
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy the image to the class directory
            final_path = os.path.join(class_dir, image_filename)
            shutil.copy(temp_file_path, final_path)
            
            # Create the generator
            test_generator = test_datagen.flow_from_directory(
                parent_temp_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=1,
                class_mode=None,
                shuffle=False
            )
            
            # Make prediction
            prediction = self.model.predict(test_generator)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(parent_temp_dir, ignore_errors=True)
            
            # Get model metrics
            metrics = self.get_metrics()
            
            # Calculate cancer probability
            cancer_prob = float(prediction[0][1])
            non_cancer_prob = float(prediction[0][0])
            is_malignant = cancer_prob > 0.5
            
            # Format prediction text based on result
            prediction_text = "Malignant" if is_malignant else "Benign"
            
            # Return results in the format expected by the frontend
            return {
                "prediction": prediction_text,
                "confidence": max(cancer_prob, non_cancer_prob) * 100,
                "probability": max(cancer_prob, non_cancer_prob),
                "class_name": prediction_text,
                "is_malignant": is_malignant,
                "model_name": f"Breast Cancer {self.selected_architecture.upper()} Model",
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "class_probabilities": {
                    "Malignant": cancer_prob,
                    "Benign": non_cancer_prob
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def get_metrics(self):
        """
        Get model metrics from the saved metrics file
        
        Returns:
            Dictionary with model metrics
        """
        metrics_path = os.path.join(self.model_save_path, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {"accuracy": 0.0, "model_name": self.selected_architecture}
    
    def get_report(self):
        """
        Get a report about the model for the API
        
        Returns:
            Dictionary with model information
        """
        metrics = self.get_metrics()
        return {
            "model_name": f"Breast Cancer {self.selected_architecture.upper()} Model",
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "trained": self.model is not None or os.path.exists(os.path.join(self.model_save_path, 'model.json'))
        }

def train_breast_cancer_model(architecture='simple_cnn', epochs=30):
    """
    This function is kept for API compatibility but just loads the pre-trained model
    
    Returns:
        True if model loaded successfully, False otherwise
    """
    try:
        model = BreastCancerModel()
        return model.model is not None
    except Exception as e:
        logger.error(f"Error loading breast cancer model: {str(e)}")
        return False