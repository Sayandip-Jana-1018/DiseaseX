# DiseaseX Healthcare Platform

DiseaseX is a comprehensive healthcare platform featuring multiple disease prediction models, including skin cancer detection, heart disease prediction, diabetes analysis, and more.

## Deployment Guide

This project consists of two main components:
1. **Frontend**: Next.js application deployed on Vercel
2. **Backend**: Flask API with machine learning models deployed on Render

### Frontend Deployment (Vercel)

1. **Prepare your repository**:
   - Push your code to GitHub

2. **Deploy to Vercel**:
   - Sign up/login to [Vercel](https://vercel.com)
   - Import your GitHub repository
   - Configure the project:
     - Framework preset: Next.js
     - Root directory: `./` (or specify if your frontend is in a subdirectory)
     - Build command: `npm run build` (default)
     - Output directory: `.next` (default)

3. **Set environment variables**:
   - `NEXT_PUBLIC_API_URL`: Your backend URL (e.g., https://diseaseX-backend.onrender.com)

4. **Deploy**:
   - Click "Deploy" and wait for the build to complete
   - Vercel will provide you with a deployment URL

### Backend Deployment (Render)

See detailed instructions in the [backend README](./backend/README.md).

## Local Development

### Frontend

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env.local` file with:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   python app.py
   ```

4. The API will be available at [http://localhost:5000](http://localhost:5000)

## Features

- Multiple disease prediction models
- Interactive UI with dark/light mode
- Responsive design for all devices
- Real-time predictions with detailed metrics
- Model comparison and explanations

## Technologies

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: Flask, TensorFlow, scikit-learn, OpenCV
- **Deployment**: Vercel (frontend), Render (backend)
