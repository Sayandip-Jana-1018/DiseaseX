import os
import argparse
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from models.heart_disease import HeartDiseaseModel
from models.diabetes import DiabetesModel
from models.liver_disease import LiverDiseaseModel
from models.skin_cancer import SkinCancerModel
from models.breast_cancer import BreastCancerModel
from models.symptom_disease import SymptomDiseaseModel

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {len(gpus)}")
        print("Using TensorFlow with GPU acceleration")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU detected. Using CPU for training.")

def train_heart_disease_model(dataset_path=None):
    """Train heart disease model"""
    print("\n=== Training Heart Disease Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'heart', 'heart.csv')
    
    model = HeartDiseaseModel()
    
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Call train method instead of train_model
    metrics = model.train(data)
    
    end_time = time.time()
    print(f"Heart Disease Model trained in {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Best model: {metrics.get('model_name', 'Unknown')}")
    
    return metrics

def train_diabetes_model(dataset_path=None):
    """Train diabetes model"""
    print("\n=== Training Diabetes Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'diabetes', 'pima_diabetes.csv')
    
    model = DiabetesModel()
    
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Call train method instead of train_model
    metrics = model.train(data)
    
    end_time = time.time()
    print(f"Diabetes Model trained in {end_time - start_time:.2f} seconds")
    
    # Check if metrics is None (training failed)
    if metrics is None:
        print("Training failed. Check the error messages above.")
    else:
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Best model: {metrics.get('model_name', 'Unknown')}")
        print(f"Metrics saved to: {model.metrics_path}")
    
    return metrics

def train_liver_disease_model(dataset_path=None):
    """Train liver disease model"""
    print("\n=== Training Liver Disease Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'liver', 'liver.csv')
    
    model = LiverDiseaseModel()
    data = pd.read_csv(dataset_path)
    metrics = model.train(data)
    
    end_time = time.time()
    if metrics is None:
        print("Training failed. Check the error messages above.")
    else:
        print(f"Liver Disease Model trained in {end_time - start_time:.2f} seconds")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Best model: {metrics.get('model_name', 'Unknown')}")
        print(f"Metrics saved to: {model.metrics_path}")
    
    return metrics

def train_skin_cancer_model(dataset_path=None, metadata_path=None, subset_size=5000):
    """Train skin cancer model"""
    print("\n=== Training Skin Cancer Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'skin', 'HAM10000')
    
    if metadata_path is None:
        metadata_path = os.path.join('datasets', 'skin', 'HAM10000_metadata.csv')
    
    model = SkinCancerModel()
    metrics = model.train_model(dataset_path, metadata_path, subset_size)
    
    end_time = time.time()
    print(f"Skin Cancer Model trained in {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Best model: {metrics.get('model_name', 'Unknown')}")
    
    return metrics



def train_breast_cancer_model(dataset_path=None, architecture='simple_cnn', epochs=30):
    """Train breast cancer model"""
    print("\n=== Training Breast Cancer Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'breast')
    
    # Create model instance
    model = BreastCancerModel(dataset_path=dataset_path)
    
    # Prepare data and train model
    model.prepare_data()
    metrics = model.train_model(architecture=architecture, epochs=epochs)
    
    end_time = time.time()
    print(f"Breast Cancer Model trained in {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Best model: {metrics.get('model_name', 'Unknown')}")
    print(f"Model saved to: {model.model_save_path}")
    
    return metrics

def train_symptom_disease_model(dataset_path=None, symptom_severity_path=None, 
                              precaution_path=None, description_path=None):
    """Train symptom-disease model"""
    print("\n=== Training Symptom-Disease Model ===")
    start_time = time.time()
    
    if dataset_path is None:
        dataset_path = os.path.join('datasets', 'symptoms', 'dataset.csv')
    
    if symptom_severity_path is None:
        symptom_severity_path = os.path.join('datasets', 'symptoms', 'Symptom-severity.csv')
    
    if precaution_path is None:
        precaution_path = os.path.join('datasets', 'symptoms', 'symptom_precaution.csv')
    
    if description_path is None:
        description_path = os.path.join('datasets', 'symptoms', 'symptom_Description.csv')
    
    model = SymptomDiseaseModel()
    metrics = model.train_model(
        dataset_path, 
        symptom_severity_path, 
        precaution_path, 
        description_path
    )
    
    end_time = time.time()
    print(f"Symptom-Disease Model trained in {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Best model: {metrics.get('model_name', 'Unknown')}")
    
    return metrics

def train_all_models(subset_size=5000):
    """Train all models"""
    print("=== Training All Models ===")
    start_time = time.time()
    
    results = {}
    
    # Train models
    results['heart'] = train_heart_disease_model()
    results['diabetes'] = train_diabetes_model()
    results['liver'] = train_liver_disease_model()
    results['skin_cancer'] = train_skin_cancer_model(subset_size=subset_size)
    results['breast_cancer'] = train_breast_cancer_model()
    results['symptom'] = train_symptom_disease_model()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n=== Training Summary ===")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nModel Accuracies:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics.get('accuracy', 0):.4f} - {metrics.get('model_name', 'Unknown')}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train disease prediction models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'heart', 'diabetes', 'liver', 
                                'skin', 'breast', 'symptom'],
                        help='Model to train')
    parser.add_argument('--subset', type=int, default=5000,
                        help='Subset size for image datasets (skin, brain, breast)')
    parser.add_argument('--architecture', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet50', 'efficientnet', 'mobilenet'],
                        help='Model architecture for image models')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs for image models')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models(subset_size=args.subset)
    elif args.model == 'heart':
        train_heart_disease_model()
    elif args.model == 'diabetes':
        train_diabetes_model()
    elif args.model == 'liver':
        train_liver_disease_model()
    elif args.model == 'skin':
        train_skin_cancer_model(subset_size=args.subset)

    elif args.model == 'breast':
        train_breast_cancer_model(architecture=args.architecture, epochs=args.epochs)
    elif args.model == 'symptom':
        train_symptom_disease_model()
