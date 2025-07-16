import os
import pandas as pd
import numpy as np
import joblib
import json
import random
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .base_model import BaseModel

class LiverDiseaseModel(BaseModel):
    """Liver Disease prediction model using standard ML algorithms"""
    
    def __init__(self):
        super().__init__('liver_disease')
        self.features = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
            'Aspartate_Aminotransferase', 'Total_Protiens', 
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        self.target = 'Dataset'
        self.dataset_path = os.path.join('datasets', 'liver', 'liver.csv')
        
        # Model directory
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models', 'liver_disease_best_model')
        self.model_path = os.path.join(self.model_dir, 'model.joblib')
        self.metrics_path = os.path.join(self.model_dir, 'metrics.json')
        
        # Try to load pre-trained model
        if not self.load_model():
            print("Pre-trained liver disease model not found. Train the model first.")
            
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def preprocess_data(self, data):
        """
        Preprocess liver disease data
        
        Args:
            data: DataFrame containing liver disease data
            
        Returns:
            X: Features
            y: Target
        """
        print("Preprocessing liver disease data...")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print(f"Found {df.isnull().sum().sum()} missing values. Filling with median/mode.")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        # Convert Gender to numeric (Male=1, Female=0)
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
        # Extract features and target
        X = df[self.features]
        y = df[self.target]
        
        # Convert target to binary (1=liver disease, 2=no liver disease) -> (1=liver disease, 0=no liver disease)
        if y.max() == 2:
            y = y.map({1: 1, 2: 0})
        
        return X, y
    
    def train(self, data=None):
        """
        Train the liver disease model
        
        Args:
            data: DataFrame containing liver disease data. If None, load from dataset_path.
            
        Returns:
            Dictionary containing training metrics
        """
        print("=== Training Liver Disease Model ===")
        
        # Load data if not provided
        if data is None:
            try:
                data = pd.read_csv(self.dataset_path)
                print(f"Loaded data from {self.dataset_path} with {data.shape[0]} rows and {data.shape[1]} columns")
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        
        # Preprocess data
        X, y = self.preprocess_data(data)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Define models to try
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        # Define parameter grids for each model
        param_grids = {
            'LogisticRegression': {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear', 'saga']
            },
            'RandomForest': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            },
            'SVM': {
                'model__C': [0.1, 1, 10],
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf', 'linear']
            },
            'XGBoost': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1, 0.3],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0]
            }
        }
        
        # Create pipelines with scaling
        pipelines = {}
        for name, model in models.items():
            pipelines[name] = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        
        # Train and evaluate models
        best_accuracy = 0
        best_model_name = None
        best_model = None
        results = {}
        grid_results = {}
        
        # For each model type
        for model_name, pipeline in pipelines.items():
            print(f"\nTraining {model_name}...")
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model_for_type = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model_for_type.predict(X_test)
            y_prob = best_model_for_type.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Calculate additional metrics
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store grid search results for model comparison
            grid_results[model_name] = grid_search
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{model_name} results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            
            # Update best model if this one is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = best_model_for_type
        
        print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        # Save the best model
        if best_model is not None:
            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save model
            joblib.dump(best_model, self.model_path)
            print(f"Saved model to {self.model_path}")
            
            # Extract feature importances if available
            feature_importance = {}
            if hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['model'], 'feature_importances_'):
                importances = best_model.named_steps['model'].feature_importances_
                feature_importance = dict(zip(self.features, importances))
                
                # Print feature importances
                print("\nFeature Importance:")
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"{feature}: {importance:.4f}")
            
            # Prepare metrics for saving
            metrics = {
                'accuracy': float(best_accuracy),
                'precision': float(results[best_model_name]['precision']),
                'recall': float(results[best_model_name]['recall']),
                'f1': float(results[best_model_name]['f1']),
                'roc_auc': float(results[best_model_name]['roc_auc']),
                'rmse': float(results[best_model_name]['rmse']),
                'r2': float(results[best_model_name]['r2']),
                'model_comparison': {}
            }
            
            # Add model comparison data
            for model_name, model_results in results.items():
                metrics['model_comparison'][model_name] = {
                    'accuracy': float(model_results['accuracy'] * 100),
                    'precision': float(model_results['precision'] * 100),
                    'recall': float(model_results['recall'] * 100),
                    'f1': float(model_results['f1'] * 100),
                    'rmse': float(model_results['rmse']),
                    'r2': float(model_results['r2'])
                }
            
            # Add best model name
            metrics['best_model'] = best_model_name
            
            # Save metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved metrics to: {self.metrics_path}")
            
            # Update model and metrics
            self.model = best_model
            self.metrics = metrics
            
            return metrics
        else:
            print("No model was trained successfully.")
            return None
    
    def load_model(self):
        """
        Load the pre-trained model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                
                # Load metrics if available
                if os.path.exists(self.metrics_path):
                    with open(self.metrics_path, 'r') as f:
                        self.metrics = json.load(f)
                    print(f"Loaded model metrics from {self.metrics_path}")
                else:
                    self.metrics = None
                    print("No metrics file found.")
                
                print(f"Loaded pre-trained liver disease model from {self.model_path}")
                return True
            else:
                print(f"No pre-trained model found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, data):
        """
        Make predictions using the trained model
        
        Args:
            data: Dictionary or DataFrame containing features for prediction
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        try:
            # Convert dictionary to DataFrame if needed
            if isinstance(data, dict):
                # Map API field names to model feature names
                feature_mapping = {
                    'age': 'Age',
                    'gender': 'Gender',
                    'total_bilirubin': 'Total_Bilirubin',
                    'direct_bilirubin': 'Direct_Bilirubin',
                    'alkaline_phosphotase': 'Alkaline_Phosphotase',
                    'alamine_aminotransferase': 'Alamine_Aminotransferase',
                    'aspartate_aminotransferase': 'Aspartate_Aminotransferase',
                    'total_protiens': 'Total_Protiens',
                    'albumin': 'Albumin',
                    'albumin_globulin_ratio': 'Albumin_and_Globulin_Ratio'
                }
                
                # Create a new dictionary with mapped feature names
                mapped_data = {}
                for api_name, model_name in feature_mapping.items():
                    if api_name in data:
                        mapped_data[model_name] = data[api_name]
                
                # Create DataFrame from mapped data
                df = pd.DataFrame([mapped_data])
            else:
                # If already a DataFrame, make a copy
                df = data.copy()
            
            # Convert Gender to numeric if needed
            if 'Gender' in df.columns and df['Gender'].dtype == 'object':
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
            
            # Ensure all required features are present
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                print(f"Missing features: {missing_features}")
                return None
            
            # Extract features
            X = df[self.features]
            
            # Make prediction
            y_pred = self.model.predict(X)[0]
            y_prob = self.model.predict_proba(X)[0]
            
            # Determine confidence
            confidence = float(y_prob[1] if y_pred == 1 else y_prob[0])
            
            # Get feature importances if available
            feature_importance = {}
            if hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['model'], 'feature_importances_'):
                importances = self.model.named_steps['model'].feature_importances_
                feature_importance = dict(zip(self.features, importances))
            
            # Prepare result
            result = {
                'prediction': int(y_pred),
                'prediction_label': 'Liver Disease' if y_pred == 1 else 'No Liver Disease',
                'confidence': float(confidence),
                'feature_importance': feature_importance
            }
            
            # Add metrics if available
            if self.metrics is not None:
                result['metrics'] = self.metrics
            
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_sample_data(self):
        """
        Get sample data for form autofill
        
        Returns:
            Dictionary with sample values for each feature
        """
        return {
            "Age": random.randint(20, 80),
            "Gender": random.choice(["Male", "Female"]),
            "Total_Bilirubin": round(random.uniform(0.4, 2.0), 2),
            "Direct_Bilirubin": round(random.uniform(0.1, 0.5), 2),
            "Alkaline_Phosphotase": random.randint(100, 300),
            "Alamine_Aminotransferase": random.randint(10, 80),
            "Aspartate_Aminotransferase": random.randint(10, 80),
            "Total_Protiens": round(random.uniform(5.0, 8.0), 1),
            "Albumin": round(random.uniform(2.5, 5.0), 1),
            "Albumin_and_Globulin_Ratio": round(random.uniform(0.8, 2.0), 1)
        }