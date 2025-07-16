import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class BaseModel:
    """Base class for all disease prediction models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        
        # Set paths for model files
        model_dir = os.path.join('trained_models', f'{model_name}_best_model')
        self.model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        self.scaler_path = os.path.join(model_dir, f'{model_name}_scaler.pkl')
        self.metrics_path = os.path.join(model_dir, f'{model_name}_metrics.pkl')
        
    def preprocess_data(self, data, target_col):
        """Preprocess data: handle missing values, encode categorical variables, etc."""
        # To be implemented by subclasses
        pass
    
    def train(self, X, y, models_to_try, param_grids, balance_classes=False, fit_params=None):
        """Train multiple models and select the best one"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply custom class balancing if needed
        if balance_classes:
            try:
                print("Applying custom class balancing in base model...")
                
                # Convert to DataFrame/Series for easier manipulation
                X_train_df = pd.DataFrame(X_train_scaled)
                y_train_series = pd.Series(y_train)
                
                # Get class counts
                class_counts = y_train_series.value_counts()
                print(f"Original training class distribution: {class_counts.to_dict()}")
                
                # For multiclass problems, balance all classes
                if len(class_counts) > 2:  # Multiclass case
                    print("Multiclass balancing: Upsampling all minority classes...")
                    
                    # Find the size of the largest class
                    max_size = class_counts.max()
                    
                    # Combine features and target for resampling
                    train_df = pd.concat([X_train_df, y_train_series], axis=1)
                    target_col = train_df.columns[-1]
                    
                    # Create a balanced dataset
                    balanced_dfs = []
                    
                    # For each class
                    for class_val, count in class_counts.items():
                        # Get samples of this class
                        class_df = train_df[train_df[target_col] == class_val]
                        
                        # If this is a minority class, upsample it
                        if count < max_size:
                            class_df = resample(
                                class_df,
                                replace=True,
                                n_samples=max_size,
                                random_state=42
                            )
                        
                        balanced_dfs.append(class_df)
                    
                    # Combine all balanced classes
                    df_balanced = pd.concat(balanced_dfs)
                    
                    # Shuffle the data
                    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    # Split back into features and target
                    X_train_scaled = df_balanced.iloc[:, :-1].values
                    y_train = df_balanced.iloc[:, -1].values
                    
                    print(f"Class balancing applied. New training shape: {X_train_scaled.shape}")
                    print(f"New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                
                else:  # Binary classification case
                    # Identify majority and minority classes
                    majority_class = class_counts.idxmax()
                    minority_class = class_counts.idxmin()
                    n_majority = class_counts[majority_class]
                    n_minority = class_counts[minority_class]
                    
                    # Balance only if there's a significant imbalance
                    if n_majority / n_minority > 1.2:  # 20% threshold for imbalance
                        # Combine features and target for resampling
                        train_df = pd.concat([X_train_df, y_train_series], axis=1)
                        
                        # Separate by class
                        df_majority = train_df[train_df.iloc[:, -1] == majority_class]
                        df_minority = train_df[train_df.iloc[:, -1] == minority_class]
                        
                        # Upsample minority class
                        df_minority_upsampled = resample(
                            df_minority,
                            replace=True,
                            n_samples=len(df_majority),
                            random_state=42
                        )
                        
                        # Combine majority and upsampled minority
                        df_balanced = pd.concat([df_majority, df_minority_upsampled])
                        
                        # Shuffle the data
                        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
                        
                        # Split back into features and target
                        X_train_scaled = df_balanced.iloc[:, :-1].values
                        y_train = df_balanced.iloc[:, -1].values
                        
                        print(f"Class balancing applied. New training shape: {X_train_scaled.shape}")
                        print(f"New training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                    else:
                        print("Training classes are already relatively balanced. Skipping resampling.")
            except Exception as e:
                print(f"Error during class balancing: {str(e)}. Proceeding with original data.")
                print("Continuing without balancing...")
        
        
        # Train and evaluate each model
        best_score = 0
        best_model = None
        
        for name, model in models_to_try.items():
            print(f"Training {name}...")
            
            # Grid search for hyperparameter tuning
            model_fit_params = {} if fit_params is None else fit_params.get(name, {})
            grid_search = GridSearchCV(
                model, 
                param_grids[name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Pass fit_params to the fit method instead
            grid_search.fit(X_train_scaled, y_train, **model_fit_params)
            
            # Get best model
            model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Best params: {grid_search.best_params_}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                self.metrics = {
                    'model_name': name,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'best_params': grid_search.best_params_
                }
                
                # For binary classification, add ROC AUC
                if len(np.unique(y)) == 2:
                    try:
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        self.metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    except:
                        pass
                
                # Generate confusion matrix
                self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                
                # Try to generate SHAP values if possible
                try:
                    print(f"Generating SHAP values for {name} model...")
                    # Different handling based on model type
                    if name == 'DeepLearning':
                        print("Using KernelExplainer for deep learning model")
                        # For Keras models, we need to use a different approach
                        import tensorflow as tf
                        # Get the underlying Keras model
                        if hasattr(model, 'model'):
                            keras_model = model.model
                        else:
                            keras_model = model
                            
                        # Create a function that returns the output of the model
                        def model_predict(x):
                            return keras_model.predict(x)
                            
                        # Use KernelExplainer for Keras models with a very small sample
                        # to avoid memory issues
                        background = X_train_scaled[:10]  # Use a small subset of training data as background
                        explainer = shap.KernelExplainer(model_predict, background)
                        shap_values = explainer.shap_values(X_test_scaled[:10])  # Limit to 10 samples for efficiency
                        
                        # Handle different output shapes
                        if isinstance(shap_values, list):
                            # For multi-class, take the mean of absolute values across classes
                            feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                        else:
                            # For binary classification
                            feature_importance = np.abs(shap_values).mean(axis=0)
                    else:
                        print("Using TreeExplainer for tree-based model")
                        try:
                            # First try with TreeExplainer which is faster but may fail additivity check
                            explainer = shap.TreeExplainer(model, check_additivity=False)  # Disable additivity check
                            shap_values = explainer(X_test_scaled[:20])  # Limit to 20 samples for efficiency
                            feature_importance = np.abs(shap_values.values).mean(axis=0)
                        except Exception as tree_error:
                            print(f"TreeExplainer failed: {str(tree_error)}. Falling back to KernelExplainer.")
                            # Fallback to KernelExplainer which is slower but more robust
                            def model_predict(x):
                                return model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
                            
                            background = X_train_scaled[:10]
                            explainer = shap.KernelExplainer(model_predict, background)
                            shap_values = explainer.shap_values(X_test_scaled[:10])
                            feature_importance = np.abs(shap_values).mean(axis=0)
                    
                    # Store feature importance
                    self.metrics['feature_importance'] = dict(zip(
                        range(X.shape[1]),
                        feature_importance.tolist()
                    ))
                except Exception as e:
                    print(f"Could not generate SHAP values: {str(e)}")
                    # Don't let SHAP errors stop the training process
        
        # Save the best model
        self.model = best_model
        print(f"Best model: {self.metrics['model_name']} with accuracy: {self.metrics['accuracy']:.4f}")
        
        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return self.metrics
    
    def load_model(self):
        """Load trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def predict(self, data):
        """Make prediction for new data"""
        # To be implemented by subclasses
        pass
    
    def get_report(self):
        """Get model metrics"""
        if not self.metrics and os.path.exists(self.model_path):
            # Try to load metrics if available
            metrics_path = os.path.join('models', 'trained_models', f'{self.model_name}_metrics.pkl')
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
        
        return self.metrics
    
    def get_sample_data(self):
        """Get sample data for form autofill"""
        # To be implemented by subclasses
        pass
