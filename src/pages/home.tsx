import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import TitleSection from '../components/ui/TitleSection';
import Navbar from '../components/layout/Navbar';
import MainLayout from '../components/layout/MainLayout';
import { useRouter } from 'next/router';
import { FiActivity, FiHeart, FiDroplet, FiThermometer, FiSun, FiInfo, FiCrosshair } from 'react-icons/fi';
import { RiVirusFill } from 'react-icons/ri';
import Image from 'next/image';

// Define model card data
const modelCards = [
  {
    id: 'symptom',
    title: 'Symptom to Disease',
    description: 'Predict possible diseases based on your symptoms',
    image: '/sample.webp',
    route: '/symptom'
  },
  {
    id: 'heart',
    title: 'Heart Disease',
    description: 'Check your heart health with our prediction model',
    image: '/heart.png',
    route: '/heart'
  },
  {
    id: 'liver',
    title: 'Liver Disease',
    description: 'Analyze liver function parameters to predict disease',
    image: '/liver.png',
    route: '/liver'
  },
  {
    id: 'diabetes',
    title: 'Diabetes',
    description: 'Predict diabetes risk based on health parameters',
    image: '/diabetes.png',
    route: '/diabetes'
  },
  {
    id: 'skin',
    title: 'Skin Cancer',
    description: 'Analyze skin lesions to detect potential skin cancer',
    image: '/skin.png',
    route: '/skin'
  },
  {
    id: 'breast',
    title: 'Breast Cancer',
    description: 'Analyze breast cancer risk based on health parameters',
    image: '/breast.png',
    route: '/breast'
  },
  {
    id: 'about',
    title: 'About',
    description: 'Learn more about our models and how they work',
    image: '/about.webp',
    route: '/about'
  }
];

const HomePage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const router = useRouter();

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

  const cardVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 12
      }
    },
    hover: {
      scale: 1.05,
      y: -10,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 10
      }
    },
    tap: {
      scale: 0.97,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 10
      }
    }
  };

  return (
    <>
      <Navbar />
      <MainLayout>
        <div className="container mx-auto px-4 py-8 flex flex-col items-center">
          <div className="mb-12 mt-4 text-center w-full max-w-3xl">
            <TitleSection 
              accentColor={accentColor}
              theme={theme}
              title="DiseaseX"
              subtitlePrefix="Your"
              subtitles={[
                'Disease Prediction System',
                'Health Analysis Platform',
                'Medical Assistant',
                'Diagnostic Helper'
              ]}
            />
          </div>

      {/* First row - 3 cards */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-8 max-w-7xl mx-auto w-full px-4 mb-8"
      >
        {modelCards.slice(0, 3).map((card) => (
          <motion.div
            key={card.id}
            className="relative cursor-pointer"
            variants={cardVariants}
            whileHover="hover"
            whileTap="tap"
            onClick={() => router.push(card.route)}
          >
            <div 
              className={`rounded-xl overflow-hidden h-full ${
                theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
              } backdrop-blur-md border transition-all duration-500`}
              style={{ 
                borderColor: `${accentColor}70`,
                boxShadow: `0 10px 30px -5px ${accentColor}50`
              }}
            >
              {/* Glow effect */}
              <div 
                className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-500 rounded-xl"
                style={{ 
                  background: `radial-gradient(circle at center, ${accentColor}40 0%, transparent 70%)`,
                }}
              />

              <div className="p-6 flex flex-col h-full items-center text-center">
                <motion.div 
                  className="w-20 h-20 rounded-full flex items-center justify-center mb-5"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                    border: `2px solid ${accentColor}60`,
                    color: accentColor,
                    boxShadow: `0 0 20px ${accentColor}30`
                  }}
                  initial={{ rotate: 0 }}
                  animate={{ 
                    rotate: [0, -5, 5, -5, 0],
                    scale: [1, 1.05, 1, 1.05, 1]
                  }}
                  transition={{ 
                    repeat: Infinity, 
                    duration: 3,
                    ease: "easeInOut"
                  }}
                  whileHover={{ 
                    scale: 1.2,
                    rotate: [0, -15, 15, -15, 0], 
                    transition: { duration: 0.8 } 
                  }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Image 
                    src={card.image} 
                    alt={`${card.title} icon`}
                    width={60}
                    height={60}
                    className="object-contain"
                    style={{ filter: theme === 'dark' ? 'brightness(1.2)' : 'none' }}
                  />
                </motion.div>
                
                <motion.h3 
                  className="text-2xl font-bold mb-3"
                  style={{ color: accentColor }}
                  whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
                >
                  {card.title}
                </motion.h3>
                
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'} max-w-xs mx-auto`}>
                  {card.description}
                </p>
                
                <div className="mt-auto pt-5">
                  <motion.div 
                    className="text-sm font-medium flex items-center justify-center gap-2 bg-opacity-20 rounded-full py-2 px-4"
                    style={{ 
                      color: accentColor,
                      backgroundColor: `${accentColor}20`,
                      border: `1px solid ${accentColor}40`
                    }}
                    whileHover={{ 
                      backgroundColor: `${accentColor}30`, 
                      scale: 1.05,
                      transition: { duration: 0.2 }
                    }}
                  >
                    <FiActivity className="animate-pulse" />
                    <span>Explore Model</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Second row - 4 cards */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate={{ opacity: 1, transition: { delay: 0.3, staggerChildren: 0.1 } }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-8 max-w-7xl mx-auto w-full px-4"
      >
        {modelCards.slice(3).map((card) => (
          <motion.div
            key={card.id}
            className="relative cursor-pointer"
            variants={cardVariants}
            whileHover="hover"
            whileTap="tap"
            onClick={() => router.push(card.route)}
          >
            <div 
              className={`rounded-xl overflow-hidden h-full ${
                theme === 'dark' ? 'bg-black/30' : 'bg-white/70'
              } backdrop-blur-md border transition-all duration-500`}
              style={{ 
                borderColor: `${accentColor}70`,
                boxShadow: `0 10px 30px -5px ${accentColor}50`
              }}
            >
              {/* Glow effect */}
              <div 
                className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-500 rounded-xl"
                style={{ 
                  background: `radial-gradient(circle at center, ${accentColor}40 0%, transparent 70%)`,
                }}
              />

              <div className="p-6 flex flex-col h-full items-center text-center">
                <motion.div 
                  className="w-20 h-20 rounded-full flex items-center justify-center mb-5"
                  style={{ 
                    background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                    border: `2px solid ${accentColor}60`,
                    color: accentColor,
                    boxShadow: `0 0 20px ${accentColor}30`
                  }}
                  initial={{ rotate: 0 }}
                  animate={{ 
                    rotate: [0, -5, 5, -5, 0],
                    scale: [1, 1.05, 1, 1.05, 1]
                  }}
                  transition={{ 
                    repeat: Infinity, 
                    duration: 3,
                    ease: "easeInOut"
                  }}
                  whileHover={{ 
                    scale: 1.2,
                    rotate: [0, -15, 15, -15, 0], 
                    transition: { duration: 0.8 } 
                  }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Image 
                    src={card.image} 
                    alt={`${card.title} icon`}
                    width={60}
                    height={60}
                    className="object-contain"
                    style={{ filter: theme === 'dark' ? 'brightness(1.2)' : 'none' }}
                  />
                </motion.div>
                
                <motion.h3 
                  className="text-2xl font-bold mb-3"
                  style={{ color: accentColor }}
                  whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
                >
                  {card.title}
                </motion.h3>
                
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'} max-w-xs mx-auto`}>
                  {card.description}
                </p>
                
                <div className="mt-auto pt-5">
                  <motion.div 
                    className="text-sm font-medium flex items-center justify-center gap-2 bg-opacity-20 rounded-full py-2 px-4"
                    style={{ 
                      color: accentColor,
                      backgroundColor: `${accentColor}20`,
                      border: `1px solid ${accentColor}40`
                    }}
                    whileHover={{ 
                      backgroundColor: `${accentColor}30`, 
                      scale: 1.05,
                      transition: { duration: 0.2 }
                    }}
                  >
                    <FiActivity className="animate-pulse" />
                    <span>Explore Model</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>
        </div>
      </MainLayout>
    </>
  );
};

export default HomePage;
