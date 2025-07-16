import React, { ReactNode, useState } from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import TitleSection from './TitleSection';

interface FormContainerProps {
  title: string;
  subtitle?: string;
  icon: ReactNode;
  iconColor?: string; // Made optional since we'll default to accentColor
  children: ReactNode;
  predictionResult?: ReactNode;
}

const FormContainer: React.FC<FormContainerProps> = ({
  title,
  subtitle = 'Prediction Model',
  icon,
  iconColor,
  children,
  predictionResult
}) => {
  const { theme, accentColor } = useTheme();
  
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

  // Find form and non-form children
  const formElements: React.ReactNode[] = [];
  const resultElements: React.ReactNode[] = [];
  
  React.Children.forEach(children, child => {
    if (React.isValidElement(child) && child.type === 'form') {
      formElements.push(child);
    } else {
      resultElements.push(child);
    }
  });

  return (
    <div className="container mx-auto px-4 py-8 flex flex-col items-center">
      <div className="mb-8 mt-4 text-center w-full max-w-3xl">
        <TitleSection 
          accentColor={accentColor}
          theme={theme}
          title={title}
          subtitlePrefix="AI-Powered"
          subtitles={[subtitle, 'Health Analysis', 'Prediction System']}
        />
      </div>
      
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-6xl mx-auto w-full"
      >
        <motion.div 
          variants={itemVariants}
          className={`rounded-xl overflow-hidden ${
            theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
          } backdrop-blur-md border p-6 md:p-8`}
          style={{ 
            borderColor: `${accentColor}30`,
            boxShadow: `0 10px 30px -5px ${accentColor}20`
          }}
        >
          <div className="flex flex-col items-center mb-8">
            <motion.div 
              className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
              style={{ 
                background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                border: `2px solid ${accentColor}60`,
                color: accentColor,
                boxShadow: `0 0 20px ${accentColor}30`
              }}
              initial={{ rotate: 0 }}
              animate={{ 
                rotate: [0, -3, 3, -3, 0],
                scale: [1, 1.03, 1, 1.03, 1]
              }}
              transition={{ 
                repeat: Infinity, 
                duration: 3,
                ease: "easeInOut"
              }}
              whileHover={{ scale: 1.1, rotate: 5 }}
              whileTap={{ scale: 0.95 }}
            >
              {icon}
            </motion.div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
            {/* Form Column */}
            <motion.div 
              variants={itemVariants}
              className="flex flex-col h-[870px]"
            >
              <div 
                className="p-6 rounded-xl border h-full flex flex-col" 
                style={{ 
                  borderColor: accentColor, 
                  backgroundColor: theme === 'dark' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.2)'
                }}
              >
                <h2 className="text-lg font-semibold mb-6 text-center" style={{ color: accentColor }}>Input Data</h2>
                <div className="flex-1">
                  {formElements.length > 0 ? formElements : children}
                </div>
              </div>
            </motion.div>
            
            {/* Prediction Result Column */}
            <motion.div 
              variants={itemVariants}
              className="flex flex-col h-full"
            >
              <div 
                className="p-6 rounded-xl border h-full flex flex-col" 
                style={{ 
                  borderColor: accentColor,
                  backgroundColor: theme === 'dark' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.2)'
                }}
              >
                <h2 className="text-lg font-semibold mb-6 text-center" style={{ color: accentColor }}>Prediction Results</h2>
                <div className="flex-1">
                  {resultElements.length > 0 ? resultElements : predictionResult || (
                    <div className="flex items-center justify-center h-full text-center opacity-50">
                      <div>Enter data and click predict to see results</div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default FormContainer;
