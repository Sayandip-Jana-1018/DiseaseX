import React from 'react';
import { useTheme } from '../../context/ThemeContext';
import { motion } from 'framer-motion';
import Navbar from './Navbar';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const { theme, accentColor } = useTheme();

  // Generate complementary colors for gradient based on accent color
  const getComplementaryColor = () => {
    // Simple complementary color calculation
    const hex = accentColor.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    
    // Adjust to create a nice gradient pair
    const adjustedR = Math.min(255, r + 50);
    const adjustedG = Math.max(0, g - 30);
    const adjustedB = Math.min(255, b + 70);
    
    return `#${adjustedR.toString(16).padStart(2, '0')}${adjustedG.toString(16).padStart(2, '0')}${adjustedB.toString(16).padStart(2, '0')}`;
  };

  return (
    <div 
      className="min-h-screen transition-all duration-500 relative overflow-hidden"
      style={{ 
        backgroundColor: theme === 'dark' ? '#0a0a0a' : '#f8f9fa',
        color: theme === 'dark' ? '#ffffff' : '#121212'
      }}
    >
      {/* Gradient background */}
      <div 
        className="absolute inset-0 opacity-20 transition-opacity duration-500"
        style={{ 
          background: `radial-gradient(circle at top right, ${getComplementaryColor()}80 0%, transparent 70%), 
                      radial-gradient(circle at bottom left, ${accentColor}80 0%, transparent 70%)`,
          filter: 'blur(80px)',
          zIndex: 0
        }}
      />
      
      {/* Decorative elements */}
      <motion.div 
        className="absolute top-20 right-[10%] w-64 h-64 rounded-full opacity-10"
        style={{ 
          background: `radial-gradient(circle, ${accentColor} 0%, transparent 70%)`,
          filter: 'blur(40px)',
        }}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.15, 0.1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
      
      <motion.div 
        className="absolute bottom-20 left-[5%] w-96 h-96 rounded-full opacity-10"
        style={{ 
          background: `radial-gradient(circle, ${getComplementaryColor()} 0%, transparent 70%)`,
          filter: 'blur(60px)',
        }}
        animate={{
          scale: [1, 1.1, 1],
          opacity: [0.1, 0.12, 0.1],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          repeatType: "reverse",
          delay: 2,
        }}
      />
      
      {/* Navbar */}
      <Navbar />
      
      {/* Content container with proper spacing for navbar */}
      <div className="relative z-10 pt-16"> {/* Added pt-16 for navbar spacing */}
        {children}
      </div>
      
      {/* Hidden scrollbar CSS */}
      <style jsx global>{`
        ::-webkit-scrollbar {
          width: 0px;
          background: transparent;
        }
        
        html {
          scrollbar-width: none;
          -ms-overflow-style: none;
        }
        
        body {
          overflow-y: scroll;
        }
      `}</style>
    </div>
  );
};

export default MainLayout;
