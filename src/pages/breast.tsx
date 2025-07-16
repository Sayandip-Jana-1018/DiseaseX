import React, { useState, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { FiImage, FiLoader, FiUpload, FiX, FiInfo } from 'react-icons/fi';
import PredictionResult from '../components/ui/PredictionResult';
import { predictDisease } from '../utils/api';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import MainLayout from '../components/layout/MainLayout';
import TitleSection from '../components/ui/TitleSection';

// Define form data type
interface BreastCancerFormData {
  image: FileList;
}

// Define result data type
interface BreastCancerResult {
  prediction: number;
  confidence: number;
  probability: number;
  class_name: string;
  is_malignant: boolean;
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  class_probabilities: Record<string, number>;
  top_features?: Array<{
    name: string;
    importance: number;
  }>;
  error?: string;
}

const BreastCancerPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BreastCancerResult | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const { theme, accentColor } = useTheme();
  
  const { register, handleSubmit, formState: { errors }, reset } = useForm<BreastCancerFormData>();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const clearImage = () => {
    setPreviewImage(null);
    reset({ image: undefined });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const onSubmit = async (data: BreastCancerFormData) => {
    if (!data.image || data.image.length === 0) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', data.image[0]);
      
      const result = await predictDisease.breast(formData);
      
      if (result.error) {
        // Handle API error response
        alert(`Error: ${result.error}. Please try again or contact support.`);
        console.error('API returned error:', result.error);
        return;
      }
      
      console.log('Breast cancer prediction result:', result);
      setResult(result);
    } catch (error) {
      console.error('Error predicting breast cancer:', error);
      alert('An error occurred while analyzing the image. Please try again later.');
    } finally {
      setLoading(false);
    }
  };
  
  // Helper function to generate gradient based on accent color
  const getGradient = (opacity1 = '80', opacity2 = '40') => {
    return `linear-gradient(135deg, ${accentColor}${opacity1} 0%, ${accentColor}${opacity2} 100%)`;
  };
  
  // Helper function to get background color based on theme
  const getBgColor = (opacity = '30') => {
    return theme === 'dark' ? `rgba(0, 0, 0, 0.${opacity})` : `rgba(255, 255, 255, 0.${opacity})`;
  };
  
  return (
    <MainLayout>
      <div className="container mx-auto px-3 py-10 flex flex-col items-center">
        {/* Title Section - More compact */}
        <div className="mb-0 text-center w-full max-w-5xl">
          <TitleSection 
            accentColor={accentColor}
            theme={theme}
            title="Breast Cancer Detection"
            subtitlePrefix="AI-Powered"
            subtitles={["Mammogram Analysis"]}
          />
        </div>
        
        {/* Main Content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-5xl"
        >
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 120, damping: 14 }}
            className="bg-opacity-30 backdrop-blur-md rounded-xl p-6 mt-6"
            style={{ 
              backgroundColor: getBgColor('10'),
              borderLeft: `4px solid ${accentColor}80`,
              boxShadow: `0 10px 30px -5px ${accentColor}30`
            }}
          >
            {result ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left side - Results */}
                <div className="flex flex-col w-full">
                  <div className="p-6 rounded-lg border-2 flex flex-col w-full h-full" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    {previewImage && (
                      <div className="flex justify-center -mb-32">
                        <div className="relative rounded-lg overflow-hidden w-full max-w-xs border-2" 
                          style={{ 
                            borderColor: `${accentColor}80`,
                            boxShadow: `0 4px 20px -2px ${accentColor}40`
                          }}
                        >
                          <img 
                            src={previewImage} 
                            alt="Mammogram" 
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                    )}
                    
                    <PredictionResult
                      prediction={result.prediction.toString()}
                      confidence={result.confidence}
                      modelName={result.model_name}
                      metrics={{
                        accuracy: result.accuracy,
                        precision: result.precision,
                        recall: result.recall,
                        f1: result.f1,
                        r2: 0.95, // Default value if not provided by API
                        rmse: 0.05 // Default value if not provided by API
                      }}
                      topFeatures={result.top_features}
                      diseaseType="breast"
                    />
                    
                    <div className="flex justify-center -mt-32">
                      <motion.button
                        onClick={() => {
                          setResult(null);
                          clearImage();
                        }}
                        className="px-6 py-3 rounded-full font-medium text-white transition-all duration-300 flex items-center"
                        style={{ 
                          background: getGradient(),
                          boxShadow: `0 4px 15px ${accentColor}40`
                        }}
                        whileHover={{ scale: 1.05, boxShadow: `0 10px 25px ${accentColor}50` }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <FiImage className="mr-2" />
                        Try Another Image
                      </motion.button>
                    </div>
                  </div>
                </div>
                
                {/* Right side - Instructions */}
                <div className="flex flex-col w-full">
                  <div className="p-10 rounded-lg border-2 flex flex-col w-full h-full justify-center items-center" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`,
                      minHeight: '450px'
                    }}
                  >
                    <h2 className="text-2xl font-semibold mb-10 text-center" style={{ color: accentColor }}>Instructions</h2>
                    
                    <div className="space-y-8 w-full max-w-md mx-auto">
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">1</span>
                        </div>
                        <p className="text-center text-base">Upload a clear, high-quality mammogram image</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">2</span>
                        </div>
                        <p className="text-center text-base">Ensure the image is properly oriented and in focus</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">3</span>
                        </div>
                        <p className="text-center text-base">Click "Analyze Mammogram" to get AI-powered prediction results</p>
                      </div>
                      
                      <div className="mt-10 p-5 rounded-md text-center w-full" 
                        style={{ 
                          backgroundColor: `${accentColor}15`,
                          borderLeft: `4px solid ${accentColor}`
                        }}
                      >
                        <div className="flex items-center justify-center gap-2 mb-3">
                          <FiInfo size={18} style={{ color: accentColor }} />
                          <span className="font-semibold text-lg" style={{ color: accentColor }}>Important Note:</span>
                        </div>
                        <p className="text-base">This tool is for educational purposes only. Always consult a medical professional for proper diagnosis.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left side - Upload */}
                <div className="flex flex-col w-full">
                  <div className="p-6 rounded-lg border-2 flex flex-col w-full h-full" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    <h2 className="text-xl font-semibold mb-6 text-center" style={{ color: accentColor }}>Upload Mammogram Image</h2>
                    
                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                      <div className="flex flex-col items-center">
                        <motion.div 
                          className={`w-full mt-24 h-64 rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer overflow-hidden relative ${
                            theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
                          }`}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          style={{ 
                            borderColor: previewImage ? `${accentColor}80` : `${accentColor}40`,
                            boxShadow: previewImage ? `0 4px 20px -2px ${accentColor}30` : 'none'
                          }}
                          onClick={() => fileInputRef.current?.click()}
                        >
              {previewImage ? (
                <>
                  <img 
                    src={previewImage} 
                    alt="Preview" 
                    className="w-full h-full object-cover"
                  />
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      clearImage();
                    }}
                    className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
                  >
                    <FiX size={16} />
                  </button>
                </>
              ) : (
                <>
                  <FiUpload size={40} className="mb-2" style={{ color: accentColor }} />
                  <p className="text-center px-4">
                    <span className="font-medium" style={{ color: accentColor }}>Click to upload</span> or drag and drop
                    <br />
                    <span className="text-sm text-gray-500">PNG, JPG or JPEG (max. 5MB)</span>
                  </p>
                </>
              )}
                          <input
                            type="file"
                            className="hidden"
                            accept="image/png, image/jpeg, image/jpg"
                            {...register('image', { 
                              required: 'Image is required',
                              onChange: handleImageChange
                            })}
                            ref={(e) => {
                              register('image').ref(e);
                              fileInputRef.current = e;
                            }}
                          />
                        </motion.div>
                        
                        {errors.image && <p className="mt-2 text-sm text-red-500">{errors.image.message}</p>}
                        
                        <p className="mt-4 text-sm text-gray-500 max-w-md text-center">
                          Upload a clear mammogram image for breast cancer detection. For best results, ensure the image is high quality and properly oriented.
                        </p>
                      </div>
                      
                      <motion.div 
                        className="flex justify-center mt-8"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                      >
                        <motion.button
                          type="submit"
                          disabled={loading || !previewImage}
                          className={`px-6 py-3 rounded-full font-medium text-white transition-all duration-300 flex items-center justify-center ${
                            !previewImage ? 'opacity-50 cursor-not-allowed' : ''
                          }`}
                          style={{ 
                            background: getGradient(),
                            boxShadow: `0 4px 15px ${accentColor}40`
                          }}
                          whileHover={!previewImage ? {} : { scale: 1.05, boxShadow: `0 10px 25px ${accentColor}50` }}
                          whileTap={!previewImage ? {} : { scale: 0.98 }}
                        >
                          {loading ? (
                            <>
                              <FiLoader className="animate-spin mr-2" />
                              Analyzing Image...
                            </>
                          ) : (
                            <>
                              <FiImage className="mr-2" />
                              Analyze Mammogram
                            </>
                          )}
                        </motion.button>
                      </motion.div>
                    </form>
                  </div>
                </div>
                
                {/* Right side - Instructions */}
                <div className="flex flex-col w-full">
                  <div className="p-10 rounded-lg border-2 flex flex-col w-full h-full justify-center items-center" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`,
                      minHeight: '450px'
                    }}
                  >
                    <h2 className="text-2xl font-semibold mb-10 text-center" style={{ color: accentColor }}>Instructions</h2>
                    
                    <div className="space-y-8 w-full max-w-md mx-auto">
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">1</span>
                        </div>
                        <p className="text-center text-base">Upload a clear, high-quality mammogram image</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">2</span>
                        </div>
                        <p className="text-center text-base">Ensure the image is properly oriented and in focus</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-5">
                        <div className="rounded-full p-3 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '48px',
                            height: '48px'
                          }}
                        >
                          <span className="text-base text-white font-bold">3</span>
                        </div>
                        <p className="text-center text-base">Click "Analyze Mammogram" to get AI-powered prediction results</p>
                      </div>
                      
                      <div className="mt-10 p-5 rounded-md text-center w-full" 
                        style={{ 
                          backgroundColor: `${accentColor}15`,
                          borderLeft: `4px solid ${accentColor}`
                        }}
                      >
                        <div className="flex items-center justify-center gap-2 mb-3">
                          <FiInfo size={18} style={{ color: accentColor }} />
                          <span className="font-semibold text-lg" style={{ color: accentColor }}>Important Note:</span>
                        </div>
                        <p className="text-base">This tool is for educational purposes only. Always consult a medical professional for proper diagnosis.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      </div>
    </MainLayout>
  );
};

export default BreastCancerPage;
