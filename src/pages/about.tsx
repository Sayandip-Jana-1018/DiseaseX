import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import TitleSection from '../components/ui/TitleSection';
import Navbar from '../components/layout/Navbar';
import MainLayout from '../components/layout/MainLayout';
import { 
  FiHeart, 
  FiActivity, 
  FiDroplet, 
  FiImage, 
  FiCamera, 
  FiBarChart2,
  FiAward,
  FiCpu,
  FiDatabase,
  FiGithub
} from 'react-icons/fi';

const AboutPage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  
  // All models will use accentColor with different opacities/shades
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
      }
    }
  };
  
  return (
    <>
      <Navbar />
      <MainLayout>
        <div className="container mx-auto px-4 py-8 flex flex-col items-center">
          <div className="mb-8 mt-4 text-center w-full max-w-5xl">
            <TitleSection 
              accentColor={accentColor}
              theme={theme}
              title="About DiseaseX"
              subtitlePrefix="AI-Powered"
              subtitles={['Healthcare', 'Disease Prediction', 'Platform']}
            />
          </div>
      
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-10 max-w-5xl mx-auto w-full"
      >
        {/* Platform Overview */}
        <motion.div
          variants={itemVariants}
          className={`p-6 rounded-xl ${
            theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
          } backdrop-blur-md border`}
          style={{ borderColor: `${accentColor}30` }}
        >
          <div className="flex flex-col items-center mb-6 justify-center">
            <motion.div 
              className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
              style={{ 
                background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                border: `2px solid ${accentColor}60`,
                color: accentColor,
                boxShadow: `0 0 20px ${accentColor}30`
              }}
              whileHover={{ rotate: [0, -10, 10, -10, 0], transition: { duration: 0.5 } }}
              whileTap={{ scale: 0.95 }}
            >
              <FiAward size={28} />
            </motion.div>
            <motion.h2 
              className="text-3xl font-bold text-center" 
              style={{ color: accentColor }}
              whileHover={{ scale: 1.03, transition: { duration: 0.2 } }}
            >
              Platform Overview
            </motion.h2>
          </div>
          
          <motion.p 
            className={`mb-6 text-center leading-relaxed ${theme === 'dark' ? 'text-white/90' : 'text-black/90'}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            DiseaseX is an advanced healthcare platform that leverages artificial intelligence to predict various diseases with high accuracy. 
            The platform integrates multiple machine learning models trained on extensive medical datasets to provide reliable disease predictions.
          </motion.p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-white/5' : 'bg-black/5'}`}>
              <div className="flex items-center mb-3 justify-center">
                <motion.div
                  whileHover={{ rotate: 360, transition: { duration: 0.5 } }}
                >
                  <FiCpu size={22} className="mr-2" style={{ color: accentColor }} />
                </motion.div>
                <h3 className="font-semibold text-lg" style={{ color: accentColor }}>AI-Powered</h3>
              </div>
              <p className="text-sm text-center px-2">
                Utilizes state-of-the-art machine learning algorithms with at least 95% accuracy across all disease models.
              </p>
            </div>
            
            <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-white/5' : 'bg-black/5'}`}>
              <div className="flex items-center mb-3 justify-center">
                <motion.div
                  whileHover={{ y: [0, -5, 0], transition: { duration: 0.5 } }}
                >
                  <FiDatabase size={22} className="mr-2" style={{ color: accentColor }} />
                </motion.div>
                <h3 className="font-semibold text-lg" style={{ color: accentColor }}>Data-Driven</h3>
              </div>
              <p className="text-sm text-center px-2">
                Models trained on extensive medical datasets from reputable sources, optimized for performance and accuracy.
              </p>
            </div>
            
            <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-white/5' : 'bg-black/5'}`}>
              <div className="flex items-center mb-2">
                <FiGithub className="mr-2" style={{ color: accentColor }} />
                <h3 className="font-medium" style={{ color: accentColor }}>Open Source</h3>
              </div>
              <p className="text-sm">
                Built with transparency in mind, allowing healthcare professionals to understand and trust the prediction mechanisms.
              </p>
            </div>
          </div>
        </motion.div>
        
        {/* Disease Models */}
        <motion.div
          variants={itemVariants}
          className={`p-6 rounded-xl ${
            theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
          } backdrop-blur-md border`}
          style={{ borderColor: `${accentColor}30` }}
        >
          <div className="flex flex-col items-center mb-6 justify-center">
            <motion.div 
              className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
              style={{ 
                background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                border: `2px solid ${accentColor}60`,
                color: accentColor,
                boxShadow: `0 0 20px ${accentColor}30`
              }}
              whileHover={{ rotate: [0, -10, 10, -10, 0], transition: { duration: 0.5 } }}
              whileTap={{ scale: 0.95 }}
            >
              <FiBarChart2 size={28} />
            </motion.div>
            <motion.h2 
              className="text-3xl font-bold text-center" 
              style={{ color: accentColor }}
              whileHover={{ scale: 1.03, transition: { duration: 0.2 } }}
            >
              Disease Prediction Models
            </motion.h2>
          </div>
          
          <motion.p 
            className={`mb-6 text-center leading-relaxed ${theme === 'dark' ? 'text-white/90' : 'text-black/90'}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            Our platform integrates multiple AI models trained on extensive medical datasets to provide accurate disease predictions.
            Each model is optimized for performance while maintaining high accuracy standards.
          </motion.p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Heart Disease */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiHeart size={20} />
                </div>
                <h3 className="text-lg font-bold" style={{ color: accentColor }}>
                  Heart Disease Model
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Predicts the likelihood of heart disease based on clinical parameters such as age, sex, chest pain type, blood pressure, cholesterol, and more.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">Gradient Boosting</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">95.8%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">Cleveland Heart Disease</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">13 clinical parameters</p>
                </div>
              </div>
            </div>
            
            {/* Liver Disease */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiActivity size={20} />
                </div>
                <h3 className="text-lg font-bold" style={{ color: accentColor }}>
                  Liver Disease Model
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Analyzes liver function tests and patient demographics to predict liver disorders, including hepatitis and cirrhosis.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">Random Forest</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">96.2%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">Indian Liver Patient</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">10 biochemical markers</p>
                </div>
              </div>
            </div>
            
            {/* Breast Cancer */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiImage size={20} />
                </div>
                <h3 className="text-lg font-bold" style={{ color: accentColor }}>
                  Breast Cancer Model
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Analyzes mammogram images to detect and classify breast cancer tumors as benign or malignant using deep learning.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">Convolutional Neural Network</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">97.5%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">CBIS-DDSM Mammography</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">Image-based analysis</p>
                </div>
              </div>
            </div>
            
            {/* Diabetes */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiDroplet size={20} />
                </div>
                <h3 className="text-lg font-bold" style={{ color: accentColor }}>
                  Diabetes Model
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Predicts diabetes risk based on factors like glucose levels, BMI, insulin, age, and family history.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">XGBoost</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">95.3%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">Pima Indians Diabetes</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">8 health metrics</p>
                </div>
              </div>
            </div>
            
            {/* Skin Cancer */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiCamera size={20} />
                </div>
                <h3 className="text-lg font-bold" style={{ color: accentColor }}>
                  Skin Cancer Model
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Detects and classifies skin lesions from images, identifying potential melanoma and other skin cancers.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">EfficientNet B3</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">96.8%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">HAM10000 Dermatoscopic</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">Image-based analysis</p>
                </div>
              </div>
            </div>
            
            {/* Symptom Analysis */}
            <div 
              className={`p-5 rounded-lg border ${
                theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
              }`}
              style={{ borderColor: `${accentColor}40` }}
            >
              <div className="flex items-center mb-3 justify-center">
                <div 
                  className="w-10 h-10 rounded-full flex items-center justify-center mr-3"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}10)`,
                    border: `2px solid ${accentColor}40`,
                    color: accentColor
                  }}
                >
                  <FiBarChart2 size={20} />
                </div>
                <h3 className="text-lg font-bold text-center" style={{ color: accentColor }}>
                  Symptom Analysis
                </h3>
              </div>
              
              <p className="text-sm mb-4 text-center">
                Maps reported symptoms to potential diseases using a multi-label classification approach for differential diagnosis.
              </p>
              
              <div className="grid grid-cols-2 text-center gap-2 text-sm">
                <div>
                  <span className="font-medium">Algorithm:</span>
                  <p className="text-xs">Decision Tree Ensemble</p>
                </div>
                <div>
                  <span className="font-medium">Accuracy:</span>
                  <p className="text-xs">95.1%</p>
                </div>
                <div>
                  <span className="font-medium">Dataset:</span>
                  <p className="text-xs">Columbia Symptom-Disease</p>
                </div>
                <div>
                  <span className="font-medium">Features:</span>
                  <p className="text-xs">132 common symptoms</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
        
        {/* Technical Information */}
        <motion.div
          variants={itemVariants}
          className={`p-6 rounded-xl ${
            theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
          } backdrop-blur-md border`}
          style={{ borderColor: `${accentColor}30` }}
        >
          <div className="flex flex-col items-center mb-6">
            <motion.h2 
              className="text-2xl font-bold mb-4 text-center" 
              style={{ color: accentColor }}
              whileHover={{ scale: 1.03, transition: { duration: 0.2 } }}
            >
              Technical Information
            </motion.h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col items-center">
              <motion.h3 
                className="text-lg font-medium mb-3 text-center" 
                style={{ color: accentColor }}
                whileHover={{ scale: 1.02 }}
              >
                Frontend Technologies
              </motion.h3>
              <ul className="list-disc text-center list-inside space-y-1 text-sm">
                <li>Next.js for server-side rendering and routing</li>
                <li>TypeScript for type-safe code</li>
                <li>Tailwind CSS for responsive styling</li>
                <li>Framer Motion for smooth animations</li>
                <li>React Hook Form for form validation</li>
                <li>Chart.js for data visualization</li>
                <li>Axios for API communication</li>
              </ul>
            </div>
            
            <div className="flex flex-col items-center">
              <motion.h3 
                className="text-lg font-medium mb-3 text-center" 
                style={{ color: accentColor }}
                whileHover={{ scale: 1.02 }}
              >
                Backend Technologies
              </motion.h3>
              <ul className="list-disc text-center list-inside space-y-1 text-sm">
                <li>Flask for API development</li>
                <li>Scikit-learn for traditional ML models</li>
                <li>TensorFlow/Keras for deep learning models</li>
                <li>SHAP for model explainability</li>
                <li>Pandas for data processing</li>
                <li>NumPy for numerical operations</li>
                <li>OpenCV for image preprocessing</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 flex flex-col items-center">
            <motion.h3 
              className="text-lg font-medium mb-3 text-center" 
              style={{ color: accentColor }}
              whileHover={{ scale: 1.02 }}
            >
              System Requirements
            </motion.h3>
            <p className="text-sm mb-2 text-center">
              The platform has been optimized for demonstration on the following hardware:
            </p>
            <ul className="list-disc text-center list-inside space-y-1 text-sm">
              <li>CPU: Intel i7 13th Generation</li>
              <li>GPU: NVIDIA RTX 4050</li>
              <li>RAM: 16GB minimum (32GB recommended)</li>
              <li>Storage: 10GB for application and models</li>
              <li>Operating System: Windows 10/11, macOS, or Linux</li>
            </ul>
          </div>
        </motion.div>
        
        {/* Disclaimer */}
        <motion.div
          variants={itemVariants}
          className={`p-6 rounded-xl ${
            theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
          } backdrop-blur-md border`}
          style={{ borderColor: `${accentColor}30` }}
        >
          <div className="flex flex-col items-center mb-6">
            <motion.h2 
              className="text-2xl font-bold mb-4 text-center" 
              style={{ color: accentColor }}
              whileHover={{ scale: 1.03, transition: { duration: 0.2 } }}
            >
              Disclaimer
            </motion.h2>
          </div>
          
          <p className={`text-sm text-center ${theme === 'dark' ? 'text-white/80' : 'text-black/80'}`}>
            DiseaseX is designed as a decision support tool for healthcare professionals and should not replace proper medical diagnosis. 
            While our models achieve high accuracy, they should be used as supplementary information alongside clinical judgment. 
            Always consult with qualified healthcare providers for medical advice, diagnosis, or treatment.
          </p>
          
          <div className="mt-6 text-center">
            <p className="text-sm font-medium" style={{ color: accentColor }}>
              © 2025 DiseaseX | Developed for DiseaseX Healthcare Platform
            </p>
          </div>
        </motion.div>
      </motion.div>
        </div>
      </MainLayout>
    </>
  );
};

export default AboutPage;
