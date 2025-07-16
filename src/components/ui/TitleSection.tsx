import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { gsap } from 'gsap';
import { TextPlugin } from 'gsap/dist/TextPlugin';

// Register GSAP plugins
if (typeof window !== 'undefined') {
  gsap.registerPlugin(TextPlugin);
}

interface TitleSectionProps {
  accentColor: string;
  theme: string;
  title?: string;
  subtitlePrefix?: string;
  subtitles?: string[];
}

const TitleSection: React.FC<TitleSectionProps> = ({ 
  accentColor, 
  theme, 
  title = "MedX - AI Medical Assistant",
  subtitlePrefix = "I am",
  subtitles = [
    'Personalized Medical Assistant',
    'AI Doctor',
    '24/7 Availability',
    'Your Health Partner'
  ]
}) => {
  const textRef = useRef<HTMLDivElement>(null);

  // Text typing animation with GSAP
  useEffect(() => {
    if (typeof window === 'undefined' || !textRef.current) return;
    
    let currentIndex = 0;
    
    const animateText = () => {
      gsap.to(textRef.current, {
        duration: 0.5,
        text: { value: '', padSpace: true },
        ease: 'none',
        onComplete: () => {
          currentIndex = (currentIndex + 1) % subtitles.length;
          gsap.to(textRef.current, {
            duration: 1.5,
            text: { value: subtitles[currentIndex], padSpace: true },
            ease: 'none',
            onComplete: () => {
              gsap.delayedCall(2, animateText);
            }
          });
        }
      });
    };
    
    gsap.to(textRef.current, {
      duration: 1.5,
      text: { value: subtitles[0], padSpace: true },
      ease: 'none',
      onComplete: () => {
        gsap.delayedCall(2, animateText);
      }
    });
    
    return () => {
      gsap.killTweensOf(textRef.current);
    };
  }, []);

  return (
    <motion.div 
      className="text-center mb-10"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div 
        className="inline-block relative mb-2"
        whileHover={{ scale: 1.05 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        {/* Decorative elements around title */}
        <motion.div 
          className="absolute -top-6 -left-6 w-12 h-12 opacity-40"
          style={{ 
            borderWidth: '2px 0 0 2px',
            borderStyle: 'solid',
            borderColor: accentColor,
            borderRadius: '30% 70% 70% 30% / 30% 30% 70% 70%'
          }}
          animate={{ rotate: 360, scale: [1, 1.1, 1] }}
          transition={{ 
            rotate: { duration: 20, repeat: Infinity, ease: "linear" },
            scale: { duration: 3, repeat: Infinity, ease: "easeInOut" }
          }}
        />
        <motion.div 
          className="absolute -bottom-6 -right-6 w-12 h-12 opacity-40"
          style={{ 
            borderWidth: '0 2px 2px 0',
            borderStyle: 'solid',
            borderColor: accentColor,
            borderRadius: '70% 30% 30% 70% / 70% 70% 30% 30%'
          }}
          animate={{ rotate: -360, scale: [1, 1.1, 1] }}
          transition={{ 
            rotate: { duration: 25, repeat: Infinity, ease: "linear" },
            scale: { duration: 4, repeat: Infinity, ease: "easeInOut" }
          }}
        />
        
        <h2 
          className="text-5xl md:text-6xl font-bold relative z-10"
          style={{ 
            backgroundImage: `linear-gradient(135deg, #fff 0%, ${accentColor} 50%, #fff 100%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textShadow: `0 0 5px rgba(255, 255, 255, 0.1), 0 0 10px ${accentColor}40`
          }}
        >
          {title}
        </h2>
      </motion.div>
      
      {/* Animated typing effect */}
      <motion.div className="flex justify-center items-center text-xl md:text-2xl mt-4 h-10">
        <span className="mr-2 opacity-80">{subtitlePrefix}</span>
        <div 
          ref={textRef} 
          className="font-semibold"
          style={{ color: accentColor }}
        >
          {subtitles[0]}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default TitleSection;
