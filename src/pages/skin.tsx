import React, { useState, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { FiCamera, FiLoader, FiUpload, FiX, FiInfo } from 'react-icons/fi';
import PredictionResult from '../components/ui/PredictionResult';
import { predictDisease } from '../utils/api';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import MainLayout from '../components/layout/MainLayout';
import TitleSection from '../components/ui/TitleSection';

// Define form data type
interface SkinCancerFormData {
  image: FileList;
}

// Define result data type
interface SkinCancerResult {
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

const SkinCancerPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SkinCancerResult | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const { theme, accentColor } = useTheme();
  
  const { register, handleSubmit, formState: { errors }, reset } = useForm<SkinCancerFormData>();
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
  
  const onSubmit = async (data: SkinCancerFormData) => {
    if (!data.image || data.image.length === 0) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', data.image[0]);
      
      const result = await predictDisease.skin(formData);
      
      if (result.error) {
        // Handle API error response
        alert(`Error: ${result.error}. Please try again or contact support.`);
        console.error('API returned error:', result.error);
        return;
      }
      
      console.log('Skin cancer prediction result:', result);
      setResult(result);
    } catch (error) {
      console.error('Error predicting skin cancer:', error);
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
      <div className="container mx-auto px-3 py-24 flex flex-col items-center">
        {/* Title Section - More compact */}
        <div className="mb-0 text-center w-full max-w-2xl">
          <TitleSection 
            accentColor={accentColor}
            theme={theme}
            title="Skin Cancer Detection"
            subtitlePrefix="AI-Powered"
            subtitles={["Dermatological Analysis"]}
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
            className={`rounded-lg overflow-hidden ${theme === 'dark' ? 'bg-black/30' : 'bg-white/70'} backdrop-blur-md border-2 p-4`}
            style={{ 
              borderColor: `${accentColor}40`,
              boxShadow: `0 8px 20px -4px ${accentColor}30`
            }}
          >
            {result ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Left side - Input Data */}
                <div className="flex flex-col w-full">
                  <div className="p-16 rounded-lg border-2 flex flex-col w-full h-full" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    <h2 className="text-lg font-semibold mb-16 text-center" style={{ color: accentColor }}>Input Data</h2>
                    {previewImage && (
                      <div className="flex justify-center mb-4">
                        <div className="relative rounded-md overflow-hidden w-128 h-128 border-2" 
                          style={{ 
                            borderColor: `${accentColor}60`,
                            boxShadow: `0 4px 12px -2px ${accentColor}30`
                          }}
                        >
                          <img 
                            src={previewImage} 
                            alt="Skin lesion" 
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                    )}
                    
                    <div className="mt-auto">
                      <motion.button
                        onClick={() => {
                          setResult(null);
                          clearImage();
                        }}
                        className="w-full px-4 py-2 rounded-md font-medium text-white transition-all duration-300"
                        style={{ 
                          background: getGradient('80', '60'),
                          boxShadow: `0 3px 10px ${accentColor}40`
                        }}
                        whileHover={{ scale: 1.03, boxShadow: `0 6px 15px ${accentColor}50` }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <FiCamera className="mr-1 inline" size={14} />
                        Analyze Another Image
                      </motion.button>
                    </div>
                  </div>
                </div>
                
                {/* Right side - Prediction Results */}
                <div className="flex flex-col w-full">
                  <div className="p-4 rounded-lg border-2 flex flex-col w-full h-full" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    <h2 className="text-lg font-semibold mb-4 text-center" style={{ color: accentColor }}>Prediction Results</h2>
                    <div className="space-y-4 overflow-auto">
                      
                      {/* Main Prediction Result */}
                      <div 
                        className="rounded-md p-3 mb-8 text-center"
                        style={{ 
                          background: getGradient('30', '10'),
                          boxShadow: `0 4px 12px -2px ${accentColor}20`
                        }}
                      >
                        <h3 className="text-xl font-bold" style={{ color: accentColor }}>
                          {result.class_name}
                        </h3>
                        <div className="flex items-center justify-center gap-1 text-sm mt-1">
                          <FiInfo size={14} style={{ color: accentColor }} />
                          <span>Confidence: {Math.round(result.confidence * 100)}%</span>
                        </div>
                      </div>
                      
                      {/* Performance Metrics */}
                      <div className="mt-8 p-4">
                        <h3 className="text-sm text-center font-medium mb-2" style={{ color: accentColor }}>Performance Metrics</h3>
                        <div 
                          className="grid grid-cols-3 gap-2 text-xs p-2 rounded-md"
                          style={{ backgroundColor: getBgColor('10') }}
                        >
                          <div className="flex justify-between items-center p-1 rounded" style={{ backgroundColor: getBgColor('20') }}>
                            <span>Accuracy:</span>
                            <span className="font-medium">{Math.round(result.accuracy * 100)}%</span>
                          </div>
                          <div className="flex justify-between items-center p-1 rounded" style={{ backgroundColor: getBgColor('20') }}>
                            <span>Precision:</span>
                            <span className="font-medium">{Math.round(result.precision * 100)}%</span>
                          </div>
                          <div className="flex justify-between items-center p-1 rounded" style={{ backgroundColor: getBgColor('20') }}>
                            <span>Recall:</span>
                            <span className="font-medium">{Math.round(result.recall * 100)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Model Comparison */}
                      <div className="mt-3 pb-8">
                        <h3 className="text-sm mt-4 text-center font-medium mb-2" style={{ color: accentColor }}>Model Comparison</h3>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="p-2 rounded" style={{ background: getGradient('20', '10') }}>
                            <p className="font-medium text-xs">ResNet</p>
                            <p className="text-xs">67%</p>
                          </div>
                          <div className="p-2 rounded" style={{ background: getGradient('20', '10') }}>
                            <p className="font-medium text-xs">EfficientNet</p>
                            <p className="text-xs">72%</p>
                          </div>
                          <div className="p-2 rounded" style={{ background: getGradient('20', '10') }}>
                            <p className="font-medium text-xs">MobileNet</p>
                            <p className="text-xs">75%</p>
                          </div>
                        </div>
                      </div>
                      
                      {/* Class Probabilities */}
                      <div className="mt-3">
                        <h3 className="text-sm text-center font-medium mb-2" style={{ color: accentColor }}>Class Probabilities</h3>
                        <div className="space-y-2 p-2 rounded-md" style={{ backgroundColor: `${accentColor}10` }}>
                          {Object.entries(result.class_probabilities || {}).map(([className, prob]) => (
                            <div key={className} className="flex items-center text-xs">
                              <div className="w-24 truncate">{className}</div>
                              <div className="flex-1 mx-2">
                                <div className="relative h-3 w-full bg-gradient-to-r from-gray-900 to-gray-100 rounded-full overflow-hidden">
                                  <div 
                                    className="absolute top-0 left-0 h-full"
                                    style={{ 
                                      width: `${Math.round(prob * 100)}%`,
                                      backgroundColor: className === result.class_name ? accentColor : `${accentColor}60`
                                    }}
                                  />
                                </div>
                              </div>
                              <div className="w-8 text-right">{Math.round(prob * 100)}%</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Left side - Upload area */}
                <div className="flex flex-col w-full">
                  <div className="p-4 rounded-lg border-2 flex flex-col w-full h-full" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    <h2 className="text-lg font-semibold mb-4 text-center" style={{ color: accentColor }}>Upload Image</h2>
                    
                    <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col h-full">
                      <motion.div 
                        className={`w-full h-48 mt-24 rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer overflow-hidden relative ${
                          theme === 'dark' ? 'bg-black/20' : 'bg-white/50'
                        }`}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                        style={{ borderColor: previewImage ? `${accentColor}80` : `${accentColor}40` }}
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
                              className="absolute top-2 right-2 text-white rounded-full p-1.5 transition-colors"
                              style={{ backgroundColor: accentColor }}
                            >
                              <FiX size={14} />
                            </button>
                          </>
                        ) : (
                          <>
                            <FiUpload size={28} className="mb-2" style={{ color: accentColor }} />
                            <p className="text-center px-2 text-sm">
                              <span className="font-medium" style={{ color: accentColor }}>Click to upload</span>
                              <br />
                              <span className="text-xs text-gray-500">PNG, JPG or JPEG (max. 5MB)</span>
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
                      
                      {errors.image && <p className="mt-2 text-xs" style={{ color: accentColor }}>{errors.image.message}</p>}
                      
                      <div className="mt-auto pt-4">
                        <motion.button
                          type="submit"
                          disabled={loading || !previewImage}
                          className={`w-full px-4 py-2.5 rounded-md font-medium text-white transition-all duration-300 flex items-center justify-center ${
                            !previewImage ? 'opacity-50 cursor-not-allowed' : ''
                          }`}
                          style={{ 
                            background: getGradient('90', '70'),
                            boxShadow: `0 3px 10px ${accentColor}40`
                          }}
                          whileHover={!previewImage ? {} : { scale: 1.03, boxShadow: `0 6px 15px ${accentColor}50` }}
                          whileTap={!previewImage ? {} : { scale: 0.98 }}
                        >
                          {loading ? (
                            <>
                              <FiLoader className="animate-spin mr-2" size={16} />
                              <span>Analyzing...</span>
                            </>
                          ) : (
                            <>
                              <FiCamera className="mr-2" size={16} />
                              <span>Analyze Skin Lesion</span>
                            </>
                          )}
                        </motion.button>
                      </div>
                    </form>
                  </div>
                </div>
                
                {/* Right side - Instructions */}
                <div className="flex flex-col w-full">
                  <div className="p-8 rounded-lg border-2 flex flex-col w-full h-full justify-center items-center" 
                    style={{ 
                      borderColor: `${accentColor}60`, 
                      backgroundColor: getBgColor('20'),
                      boxShadow: `0 4px 12px -2px ${accentColor}20`,
                      minHeight: '400px'
                    }}
                  >
                    <h2 className="text-xl font-semibold mb-8 text-center" style={{ color: accentColor }}>Instructions</h2>
                    
                    <div className="space-y-6 w-full max-w-md mx-auto">
                      <div className="flex flex-col items-center gap-4">
                        <div className="rounded-full p-2.5 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '36px',
                            height: '36px'
                          }}
                        >
                          <span className="text-sm text-white font-bold">1</span>
                        </div>
                        <p className="text-center text-base">Upload a clear, well-lit image of the skin lesion</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-4">
                        <div className="rounded-full p-2.5 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '36px',
                            height: '36px'
                          }}
                        >
                          <span className="text-sm text-white font-bold">2</span>
                        </div>
                        <p className="text-center text-base">Ensure the lesion is centered and in focus</p>
                      </div>
                      
                      <div className="flex flex-col items-center gap-4">
                        <div className="rounded-full p-2.5 flex-shrink-0 flex items-center justify-center" 
                          style={{ 
                            backgroundColor: accentColor,
                            width: '36px',
                            height: '36px'
                          }}
                        >
                          <span className="text-sm text-white font-bold">3</span>
                        </div>
                        <p className="text-center text-base">Click "Analyze Skin Lesion" to get AI-powered prediction results</p>
                      </div>
                      
                      <div className="mt-8 p-4 rounded-md text-center w-full" 
                        style={{ 
                          backgroundColor: `${accentColor}15`,
                          borderLeft: `3px solid ${accentColor}`
                        }}
                      >
                        <div className="flex items-center justify-center gap-2 mb-2">
                          <FiInfo size={16} style={{ color: accentColor }} />
                          <span className="font-semibold" style={{ color: accentColor }}>Important Note:</span>
                        </div>
                        <p>This tool is for educational purposes only. Always consult a dermatologist for proper diagnosis.</p>
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

export default SkinCancerPage;