import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tempfile
import os

# Custom KerasClassifier implementation since tensorflow.keras.wrappers.scikit_learn is deprecated
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, epochs=100, batch_size=32, verbose=1, 
                 validation_split=0.1, **kwargs):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.kwargs = kwargs
        self.model = None
        self.history = None
        self.classes_ = None
        
    def get_params(self, deep=True):
        # This method is required for GridSearchCV to work properly
        params = {
            'build_fn': self.build_fn,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'validation_split': self.validation_split
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        # This method is required for GridSearchCV to work properly
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def fit(self, X, y, **kwargs):
        # Store unique classes for scikit-learn compatibility
        self.classes_ = np.unique(y)
        
        # Convert y to one-hot encoding if needed for multi-class
        if len(self.classes_) > 2:
            y_encoded = to_categorical(y)
        else:
            y_encoded = y
            
        if self.build_fn is None:
            self.model = self.__call__(**self.kwargs)
        else:
            self.model = self.build_fn()
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping to prevent overfitting
        if 'patience' in self.kwargs:
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=self.kwargs.get('patience', 5),
                restore_best_weights=True
            ))
            
        # Model checkpoint to save best model - use .keras format instead of .h5
        temp_dir = tempfile.gettempdir()
        checkpoint_path = os.path.join(temp_dir, 'best_model.keras')
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ))
        
        self.history = self.model.fit(
            X, y_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks,
            validation_split=self.validation_split
        )
        return self
    
    def predict(self, X, **kwargs):
        preds = self.model.predict(X, **kwargs)
        if preds.ndim > 1 and preds.shape[1] > 1:
            # Multiclass case
            return np.argmax(preds, axis=1)
        else:
            # Binary case
            return (preds > 0.5).astype(int).flatten()
    
    def predict_proba(self, X, **kwargs):
        preds = self.model.predict(X, **kwargs)
        if preds.ndim > 1 and preds.shape[1] > 1:
            # Multiclass case - already probabilities
            return preds
        else:
            # Binary case - add second column for scikit-learn compatibility
            return np.column_stack([1 - preds.flatten(), preds.flatten()])
            
    # Methods needed for SHAP compatibility
    def __call__(self, X):
        """Needed for SHAP compatibility"""
        return self.predict_proba(X)
        
    @property
    def feature_names(self):
        """Return feature names if available"""
        if hasattr(self, '_feature_names'):
            return self._feature_names
        return None
        
    @feature_names.setter
    def feature_names(self, names):
        """Set feature names"""
        self._feature_names = names
        
    def score(self, X, y, **kwargs):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X, **kwargs))
