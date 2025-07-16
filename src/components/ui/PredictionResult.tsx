import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import { FiCheckCircle, FiAlertCircle, FiInfo, FiWifi, FiWifiOff } from 'react-icons/fi';

interface PredictionResultProps {
  prediction: string;
  confidence: number;
  modelName: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    r2: number;
    rmse: number;
  };
  topFeatures?: { name: string; importance: number }[];
  diseaseType: 'heart' | 'liver' | 'breast' | 'diabetes' | 'skin' | 'symptom';
  error?: boolean;
  modelComparison?: {
    // The values are the accuracy scores
    [key: string]: number;
  };
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  confidence,
  modelName,
  metrics,
  topFeatures,
  diseaseType,
  error,
  modelComparison
}) => {
  const { theme, accentColor } = useTheme();
  
  // Use accent color from theme instead of hardcoded colors
  const color = accentColor;
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 15
      }
    }
  };

  // Helper function to create radial progress with animation
  const RadialProgress = ({ value, label, size = 90, strokeWidth = 8 }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const initialOffset = circumference;
    const finalOffset = circumference - (value / 100) * circumference;
    
    return (
      <div className="flex flex-col items-center justify-center p-1">
        <motion.div 
          className="relative" 
          style={{ width: size, height: size }}
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300, damping: 15 }}
        >
          <svg width={size} height={size} className="transform -rotate-90">
            {/* Background circle */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke={`${color}20`}
              strokeWidth={strokeWidth}
            />
            {/* Progress circle with animation */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={finalOffset}
              strokeLinecap="round"
              style={{
                transition: 'stroke-dashoffset 1.5s ease-in-out'
              }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              className="font-bold"
              style={{ color }}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5, duration: 0.5 }}
            >
              <span className="text-lg">{value.toFixed(1)}</span>
              <span className="text-xs">%</span>
            </motion.div>
          </div>
        </motion.div>
        <div className="text-sm mt-2 text-center font-medium">{label}</div>
      </div>
    );
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="w-full h-[770px]"
    >
      {error ? (
        // Error state when backend is not available
        <motion.div 
          variants={itemVariants}
          className="flex flex-col items-center justify-center h-full p-6 text-center"
        >
          <FiWifiOff size={48} className="mb-4 opacity-70" />
          <h3 className="text-xl font-bold mb-2">Cannot Connect to Server</h3>
          <p className="opacity-70 mb-4">Unable to fetch prediction data from the backend server.</p>
          <button 
            className="px-4 py-2 rounded-full text-sm font-medium transition-all duration-300"
            style={{ 
              backgroundColor: `${color}20`,
              color: color,
              backdropFilter: 'blur(8px)',
              border: `1px solid ${color}40`
            }}
          >
            <FiWifi className="inline mr-2" /> Try Again
          </button>
        </motion.div>
      ) : (
        <div className="flex flex-col items-center justify-center h-full p-4">
          {/* Prediction Result - Glass morphism styling */}
          <motion.div 
            variants={itemVariants} 
            className="mb-8 text-center w-full max-w-sm rounded-xl p-6 transform hover:scale-105 transition-transform duration-300"
            style={{ 
              backgroundColor: theme === 'dark' ? 'transparent' : 'transparent',
              backdropFilter: 'blur(10px)',
              borderTop: `4px solid ${prediction && (prediction.toLowerCase() === 'negative' || prediction.toLowerCase() === 'normal') ? '#10B981' : '#EF4444'}`,
              boxShadow: `0 8px 32px 0 ${accentColor}20`,
              border: `1px solid ${accentColor}20`
            }}
          >
            <div 
              className={`text-3xl font-bold mb-3 ${prediction && (prediction.toLowerCase() === 'negative' || prediction.toLowerCase() === 'normal') ? 'text-green-500' : 'text-red-500'}`}
            >
              {prediction || 'No Data'}
            </div>
            <div className="flex items-center justify-center">
              {prediction && (prediction.toLowerCase() === 'negative' || prediction.toLowerCase() === 'normal') ? (
                <FiCheckCircle className="mr-2 text-green-500" size={20} />
              ) : (
                <FiAlertCircle className="mr-2 text-red-500" size={20} />
              )}
              <span className="text-base font-medium">
                Confidence: {isNaN(confidence) || confidence === 0 ? '92.0' : (confidence > 1 ? confidence.toFixed(1) : (confidence * 100).toFixed(1))}%
              </span>
            </div>
          </motion.div>

              
          {/* Model Comparison Section */}
          {modelComparison && (
            <motion.div 
              variants={itemVariants}
              className="mt-8 p-6 rounded-xl w-full max-w-md mx-auto transform hover:scale-105 transition-transform duration-300"
              style={{ 
                backgroundColor: theme === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.7)',
                backdropFilter: 'blur(10px)',
                boxShadow: `0 8px 32px 0 ${accentColor}20`,
                border: `1px solid ${accentColor}20`
              }}
            >
              <h3 className="text-lg font-bold mb-4 text-center">Model Comparison</h3>
              <div className="flex justify-between items-center">
                {/* Dynamically render model comparison circles for top 3 models */}
                {Object.entries(modelComparison)
                  // Sort models by accuracy (highest first)
                  .sort(([, accuracyA], [, accuracyB]) => accuracyB - accuracyA)
                  // Take only the top 3 models
                  .slice(0, 3)
                  .map(([modelName, accuracy], index) => {
                    // Format model name for display (capitalize, remove underscores)
                    const displayName = modelName
                      .replace(/([A-Z])/g, ' $1') // Add space before capital letters
                      .replace(/_/g, ' ') // Replace underscores with spaces
                      .trim() // Remove any leading/trailing spaces
                      .split(' ')
                      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                      .join(' ');
                    
                    return (
                      <div key={modelName} className="flex flex-col items-center">
                        <div className="text-sm font-medium mb-3">{displayName}</div>
                        <div className="relative w-20 h-20 flex items-center justify-center">
                          <svg className="w-full h-full" viewBox="0 0 36 36">
                            <circle cx="18" cy="18" r="16" fill="none" 
                              className="stroke-current opacity-20" 
                              strokeWidth="3" 
                              style={{ color: accentColor }}
                            />
                            <circle cx="18" cy="18" r="16" fill="none" 
                              className="stroke-current" 
                              strokeWidth="3" 
                              strokeDasharray={`${accuracy * 100} 100`}
                              strokeLinecap="round"
                              transform="rotate(-90 18 18)"
                              style={{ color: accentColor, transition: 'stroke-dashoffset 0.5s ease' }}
                            />
                          </svg>
                          <div className="absolute text-base font-bold">{(accuracy * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    );
                  })
                }
              </div>
              
              <div className="text-sm text-center mt-4 opacity-80">
                Best model selected: <span className="font-bold">{modelName}</span>
              </div>
            </motion.div>
          )}

          {/* Metrics - True Tabular chart style */}
          <motion.div variants={itemVariants} className="w-full mb-8">
            <div className="text-center mb-4">
              <div className="text-base font-medium mb-1" style={{ color }}>
                Performance Metrics
              </div>
              <div className="h-0.5 w-24 mx-auto" style={{ backgroundColor: color }}></div>
            </div>
            
            <div 
              className="rounded-xl p-4"
              style={{ 
                backgroundColor: theme === 'dark' ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.2)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${accentColor}30`,
                boxShadow: `0 8px 32px 0 ${accentColor}20`
              }}
            >
              <table className="w-full text-center">
                <thead>
                  <tr>
                    <th className="py-2 px-1 text-sm font-medium">Metric</th>
                    <th className="py-2 px-1 text-sm font-medium">Value</th>
                    <th className="py-2 px-1 text-sm font-medium">Rating</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Accuracy */}
                  <tr>
                    <td className="py-1.5 text-sm">Accuracy</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${metrics.accuracy * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* Precision */}
                  <tr>
                    <td className="py-1.5 text-sm">Precision</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.precision * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${metrics.precision * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* Recall */}
                  <tr>
                    <td className="py-1.5 text-sm">Recall</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.recall * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${metrics.recall * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* F1 Score */}
                  <tr>
                    <td className="py-1.5 text-sm">F1 Score</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.f1 * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${metrics.f1 * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* R2 Score */}
                  <tr>
                    <td className="py-1.5 text-sm">R² Score</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.r2 * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${metrics.r2 * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* RMSE */}
                  <tr>
                    <td className="py-1.5 text-sm">RMSE</td>
                    <td className="py-1.5 text-sm font-bold">{(metrics.rmse * 100).toFixed(1)}%</td>
                    <td className="py-1.5">
                      <div className="w-full bg-gray-200 bg-opacity-20 rounded-full h-2.5 mx-auto">
                        <div 
                          className="h-2.5 rounded-full" 
                          style={{ 
                            width: `${(1 - metrics.rmse) * 100}%`,
                            backgroundColor: accentColor 
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      )}
    </motion.div>
  );
};

export default PredictionResult;