# DiseaseX Backend API

This is the backend for the DiseaseX healthcare platform's multi-disease prediction system. It provides API endpoints for disease prediction and model metrics.

## Deployment Guide

### Local Development

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the models (if needed):
   ```bash
   python train_models.py
   ```

3. Run the Flask application:
   ```bash
   python run.py
   ```
   The API will be available at http://localhost:5000

### Production Deployment on Render

1. Create a new account on [Render](https://render.com/) if you don't have one

2. Create a new Web Service:
   - Connect your GitHub repository
   - Select the backend directory
   - Use the following settings:
     - **Name**: diseaseX-backend (or your preferred name)
     - **Environment**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python run.py`

3. Add the following environment variables:
   - `FLASK_ENV`: production
   - `PYTHONUNBUFFERED`: true

4. Deploy the service

5. Once deployed, note the URL (e.g., https://diseaseX-backend.onrender.com)

### Important Notes for Deployment

1. **Model Files**: Ensure all trained model files are included in your repository or uploaded to Render's persistent disk storage.

2. **Memory Requirements**: The machine learning models (especially TensorFlow) require significant memory. Choose an appropriate instance size on Render.

3. **Cold Starts**: Free tier on Render has cold starts. The first request after inactivity may take longer to respond.

4. **CORS Configuration**: The backend is already configured to accept requests from any origin. For production, you may want to restrict this to only your frontend domain.

## API Endpoints

### Disease Prediction
- `POST /api/predict/heart` - Heart disease prediction
- `POST /api/predict/diabetes` - Diabetes prediction
- `POST /api/predict/liver` - Liver disease prediction
- `POST /api/predict/skin-cancer` - Skin cancer prediction (requires image upload)
- `POST /api/predict/breast-cancer` - Breast cancer prediction (requires image upload)
- `POST /api/predict/symptom` - Symptom-based disease prediction

### Utility Endpoints
- `GET /api/model-report` - Get metrics for all models
- `GET /api/sample-data` - Get sample data for testing
- `GET /api/test` - Test if the API is running

## Connecting Frontend to Backend

When deploying your frontend to Vercel, add the following environment variable in the Vercel dashboard:
- `NEXT_PUBLIC_API_URL`: Your Render backend URL (e.g., https://diseaseX-backend.onrender.com)

```bash
python train_models.py --model all
```

You can also train individual models:

```bash
python train_models.py --model heart
python train_models.py --model diabetes
python train_models.py --model liver
python train_models.py --model skin --subset 5000
python train_models.py --model breast
python train_models.py --model symptom
```

3. Run the Flask server:

```bash
python run.py
```

By default, the server runs on `0.0.0.0:5000`. You can change the host and port:

```bash
python run.py --host 127.0.0.1 --port 8000
```

## API Endpoints

### Heart Disease Prediction

```
POST /api/predict/heart
```

Request body:
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Diabetes Prediction

```
POST /api/predict/diabetes
```

Request body:
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

### Liver Disease Prediction

```
POST /api/predict/liver
```

Request body:
```json
{
  "Age": 65,
  "Gender": 1,
  "Total_Bilirubin": 0.7,
  "Direct_Bilirubin": 0.1,
  "Alkaline_Phosphotase": 187,
  "Alamine_Aminotransferase": 16,
  "Aspartate_Aminotransferase": 18,
  "Total_Protiens": 6.8,
  "Albumin": 3.3,
  "Albumin_and_Globulin_Ratio": 0.9
}
```

### Skin Cancer Prediction

```
POST /api/predict/skin-cancer
```

The mapping is:

akiec → 'Actinic keratoses'
bcc → 'Basal cell carcinoma'
bkl → 'Benign keratosis-like lesions' (this is what you're seeing in the data sample)
df → 'Dermatofibroma'
nv → 'Melanocytic nevi'
mel → 'Melanoma'
vasc → 'Vascular lesion'

Request body:
- Form data with an image file named "image"

### Breast Cancer Prediction

```
POST /api/predict/breast-cancer
```

Request body:
- Form data with an image file named "image"

### Symptom-to-Disease Prediction

```
POST /api/predict/symptom
```

Request body:
```json
{
  "symptoms": ["headache", "fever", "cough"]
}
```

### Model Reports

```
GET /api/model-report
```

Returns metrics for all trained models.

### Sample Data

```
GET /api/sample-data?disease=heart
GET /api/sample-data?disease=diabetes
GET /api/sample-data?disease=liver
```

Returns sample data for form autofill.

## Dataset Usage and Training Time

The system uses the provided datasets for each disease model:

1. **Heart Disease**: Uses the heart.csv dataset (~300 records)
   - Training time: ~1-2 minutes
   - Accuracy target: 95%+

2. **Diabetes**: Uses the diabetes.csv dataset (~768 records)
   - Training time: ~1-2 minutes
   - Accuracy target: 95%+

3. **Liver Disease**: Uses the indian_liver_patient.csv dataset (~583 records)
   - Training time: ~1-2 minutes
   - Accuracy target: 95%+

4. **Skin Cancer**: Uses the HAM10000 dataset with subset selection
   - Default subset: 5,000 images (out of ~10,000)
   - Training time: ~10-15 minutes with GPU acceleration
   - Accuracy target: 95%+

5. **Breast Cancer**: Uses the provided breast cancer image dataset
   - Default subset: 5,000 images (if dataset is larger)
   - Training time: ~10-15 minutes with GPU acceleration
   - Accuracy target: 95%+

6. **Symptom-Disease**: Uses the symptom dataset files
   - Training time: ~1-2 minutes
   - Accuracy target: 95%+

**Total estimated training time**: ~25-40 minutes on an i7 13th gen with RTX 4050 GPU

## Hardware Optimization

The model training script is optimized for systems with NVIDIA GPUs. It's specifically tested on an i7 13th gen with RTX 4050 for optimal performance. The script automatically detects and utilizes available GPU resources for faster training.

## Model Selection

The training process automatically selects the best model for each disease by comparing:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Deep Neural Networks

For image-based models (skin cancer and breast cancer), the system compares:
- Custom CNN
- MobileNetV2
- EfficientNetB0
- ResNet50/DenseNet121

Each model is tuned using GridSearchCV to find the optimal hyperparameters, and the best model is selected based on validation accuracy.
