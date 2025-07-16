
import os
import numpy as np
import cv2
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from pathlib import Path

from .base_model import BaseModel

class BreastCancerModel(BaseModel):
    """
    Breast Cancer detection model using deep learning on mammogram images.
    This model compares multiple architectures (ResNet50, EfficientNetB0, MobileNetV2, DenseNet121)
    and selects the best performing one based on validation accuracy.
    """
    
    def __init__(self):
        super().__init__('breast_cancer')
        self.image_size = (224, 224)
        self.class_names = ['Benign', 'Malignant']
        
        # Define model paths
        self.model_dir = os.path.join('trained_models', 'breast_cancer_best_model')
        self.model_path = os.path.join(self.model_dir, 'model.h5')
        self.metrics_path = os.path.join(self.model_dir, 'metrics.json')
        self.history_path = os.path.join(self.model_dir, 'training_history.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Pre-trained breast cancer model loaded successfully from {self.model_path}")
                
                # Load metrics if available
                if os.path.exists(self.metrics_path):
                    with open(self.metrics_path, 'r') as f:
                        self.metrics = json.load(f)
                    print(f"Model metrics loaded from {self.metrics_path}")
            except Exception as e:
                print(f"Error loading pre-trained breast cancer model: {e}")
                self.model = None
                self.metrics = None
        else:
            print("Pre-trained breast cancer model not found. Train the model first.")
            self.model = None
            self.metrics = None
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize image
        img = cv2.resize(image, self.image_size)
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        # Expand dimensions to match model input shape
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def create_model(self, architecture='resnet50', learning_rate=0.001):
        """
        Create a deep learning model for breast cancer detection using transfer learning
        
        Args:
            architecture: Base model architecture ('resnet50', 'efficientnet', 'mobilenet', 'densenet')
            learning_rate: Initial learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
        # Choose base model based on architecture parameter
        if architecture == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=(*self.image_size, 3))
        elif architecture == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                      input_shape=(*self.image_size, 3))
        elif architecture == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                   input_shape=(*self.image_size, 3))
        elif architecture == 'densenet':
            base_model = DenseNet121(weights='imagenet', include_top=False, 
                                   input_shape=(*self.image_size, 3))
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create model architecture
        inputs = Input(shape=(*self.image_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        """
        Create a custom CNN model for breast cancer detection
        
        Args:
            learning_rate: Initial learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def plot_training_history(self, history, model_name, save_path=None):
        """
        Plot training history
        
        Args:
            history: Keras history object
            model_name: Name of the model for plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def evaluate_model(self, model, test_generator):
        """
        Evaluate model performance with detailed metrics
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred_prob = model.predict(test_generator)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        
        # Create confusion matrix plot
        cm_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        # Return metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
        }
        
        return metrics
    
    def train_model(self, dataset_path, batch_size=32, epochs=50, validation_split=0.2):
        """
        Train breast cancer model using transfer learning with multiple architectures
        
        Args:
            dataset_path: Path to dataset directory
            batch_size: Batch size for training
            epochs: Maximum number of epochs to train
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of evaluation metrics for the best model
        """
        print(f"\n{'='*50}")
        print("BREAST CANCER DETECTION MODEL TRAINING")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path {dataset_path} does not exist.")
            return None
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=validation_split,
            fill_mode='nearest'
        )
        
        # Test data generator (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        print("Creating data generators...")
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Update class names from directory
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes found: {self.class_names}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Define model architectures to try
        model_architectures = [
            ('resnet50', self.create_model('resnet50')),
            ('efficientnet', self.create_model('efficientnet')),
            ('mobilenet', self.create_model('mobilenet')),
            ('densenet', self.create_model('densenet'))
        ]
        
        # Train and compare models
        results = {}
        best_val_acc = 0
        best_model = None
        best_history = None
        best_model_name = ""
        
        for model_name, model in model_architectures:
            print(f"\n{'-'*50}")
            print(f"Training {model_name} model...")
            print(f"{'-'*50}")
            
            # Callbacks
            checkpoint_path = os.path.join(self.model_dir, f"{model_name}_checkpoint.h5")
            callbacks = [
                ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            ]
            
            # Train model
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_acc = model.evaluate(val_generator, verbose=0)
            print(f"{model_name} - Validation Accuracy: {val_acc:.4f}")
            
            # Plot training history
            history_plot_path = os.path.join(self.model_dir, f"{model_name}_training_history.png")
            self.plot_training_history(history, model_name, save_path=history_plot_path)
            
            # Save results
            results[model_name] = {
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            }
            
            # Update best model if improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_history = history
                best_model_name = model_name
        
        print(f"\n{'-'*50}")
        print(f"Best model: {best_model_name} with validation accuracy: {best_val_acc:.4f}")
        print(f"{'-'*50}")
        
        # Fine-tune the best model if it's a transfer learning model
        if best_model_name in ['resnet50', 'efficientnet', 'mobilenet', 'densenet']:
            print(f"\nFine-tuning {best_model_name} model...")
            
            # Unfreeze some layers
            for layer in best_model.layers:
                if hasattr(layer, 'layers'):
                    # Unfreeze the last 30% of the base model layers
                    num_layers = len(layer.layers)
                    unfreeze_from = int(num_layers * 0.7)
                    
                    for i, l in enumerate(layer.layers):
                        if i >= unfreeze_from:
                            l.trainable = True
            
            # Recompile with lower learning rate
            best_model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"Fine-tuning with {sum(1 for layer in best_model.layers if layer.trainable)} trainable layers")
            
            # Callbacks for fine-tuning
            fine_tune_checkpoint = os.path.join(self.model_dir, f"{best_model_name}_fine_tuned_checkpoint.h5")
            fine_tune_callbacks = [
                ModelCheckpoint(fine_tune_checkpoint, monitor='val_accuracy', save_best_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
            ]
            
            # Train with fine-tuning
            fine_tune_history = best_model.fit(
                train_generator,
                epochs=20,
                validation_data=val_generator,
                callbacks=fine_tune_callbacks,
                verbose=1
            )
            
            # Evaluate after fine-tuning
            val_loss, val_acc = best_model.evaluate(val_generator, verbose=0)
            print(f"After fine-tuning - Validation Accuracy: {val_acc:.4f}")
            
            # Plot fine-tuning history
            fine_tune_plot_path = os.path.join(self.model_dir, f"{best_model_name}_fine_tuning_history.png")
            self.plot_training_history(fine_tune_history, f"{best_model_name} (Fine-tuned)", save_path=fine_tune_plot_path)
            
            # Update best accuracy if improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_history = fine_tune_history
            
            # Update results
            results[f"{best_model_name}_fine_tuned"] = {
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            }
        
        # Save best model
        self.model = best_model
        save_model(best_model, self.model_path)
        print(f"Best model saved to {self.model_path}")
        
        # Create test generator for final evaluation
        test_generator = test_datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Evaluate best model on test data
        print("\nEvaluating best model on test data...")
        metrics = self.evaluate_model(best_model, test_generator)
        
        # Add additional information to metrics
        metrics.update({
            'model_name': best_model_name,
            'class_names': self.class_names,
            'training_time': float(time.time() - start_time),
            'architecture_comparison': results
        })
        
        # Save metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Model metrics saved to {self.metrics_path}")
        
        # Save training history
        joblib.dump(best_history.history, self.history_path)
        print(f"Training history saved to {self.history_path}")
        
        # Print final results
        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Best model: {best_model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Training time: {metrics['training_time']:.2f} seconds ({metrics['training_time']/60:.2f} minutes)")
        print(f"{'='*50}")
        
        self.metrics = metrics
        return metrics
    
    def predict(self, image_file):
        """
        Make breast cancer prediction from image
        
        Args:
            image_file: Image file object
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.model:
            return {"error": "Model not loaded. Please train the model first."}
        
        try:
            # Read image
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Preprocess image
            img = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(img)[0][0]
            
            # Binary classification (threshold at 0.5)
            is_malignant = prediction > 0.5
            
            # Get class name
            class_name = "Malignant" if is_malignant else "Benign"
            
            # Calculate probabilities
            malignant_prob = float(prediction)
            benign_prob = float(1 - prediction)
            
            result = {
                "prediction": class_name,
                "probability": float(prediction if is_malignant else 1 - prediction),
                "malignant_probability": malignant_prob,
                "benign_probability": benign_prob,
                "model_name": self.metrics.get('model_name', 'Breast Cancer Detection Model') if self.metrics else 'Breast Cancer Detection Model',
                "accuracy": float(self.metrics.get('accuracy', 0)) if self.metrics else 0
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_report(self):
        """
        Get model metrics
        
        Returns:
            Dictionary containing model metrics
        """
        if not self.metrics and os.path.exists(self.metrics_path):
            # Try to load metrics if available
            try:
                with open(self.metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            except:
                pass
        
        return self.metrics