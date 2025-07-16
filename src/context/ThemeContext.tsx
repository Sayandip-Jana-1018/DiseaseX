import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';

type Theme = 'light' | 'dark';

interface ColorPalette {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  error: string;
  success: string;
  warning: string;
  info: string;
}

interface ThemeContextType {
  theme: Theme;
  colors: ColorPalette;
  accentColor: string;
  toggleTheme: () => void;
  setAccentColor: (color: string) => void;
  getThemeColor: (colorName: keyof ColorPalette) => string;
}

// Predefined color palettes with improved aesthetics
const darkPalette: ColorPalette = {
  primary: '#0a0a0a',      // Deeper black for better contrast
  secondary: '#1a1a1a',    // Slightly lighter for layering
  accent: '#00d4ff',       // Brighter cyan for better visibility
  background: '#080808',   // Nearly black background for depth
  surface: '#1e1e1e',      // Dark surface for cards/elements
  text: '#ffffff',         // Pure white text
  textSecondary: '#cccccc', // Lighter secondary text for better readability
  border: '#333333',       // Dark borders
  error: '#ff4d6a',        // Brighter error color
  success: '#4ade80',      // Vibrant success green
  warning: '#ffb74d',      // Softer warning orange
  info: '#38bdf8'          // Bright info blue
};

const lightPalette: ColorPalette = {
  primary: '#ffffff',      // Pure white
  secondary: '#f8f8f8',    // Slightly off-white for layering
  accent: '#0099cc',       // Deeper blue for better contrast on light backgrounds
  background: '#f0f0f0',   // Softer background
  surface: '#ffffff',      // White surface for cards/elements
  text: '#111111',         // Nearly black text for better contrast
  textSecondary: '#555555', // Darker secondary text for better readability
  border: '#e6e6e6',       // Subtle borders
  error: '#dc2626',        // Strong error red
  success: '#16a34a',      // Rich success green
  warning: '#f59e0b',      // Warm warning amber
  info: '#0284c7'          // Deep info blue
};

// Generate a color palette based on accent color with harmonizing colors
const generatePalette = (baseTheme: ColorPalette, accentColor: string): ColorPalette => {
  // Convert hex to HSL for easier manipulation
  const hexToHSL = (hex: string): {h: number, s: number, l: number} => {
    // Remove the # if present
    hex = hex.replace(/^#/, '');
    
    // Parse the RGB values
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;
    
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0, s = 0;
    const l = (max + min) / 2;
    
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      
      h /= 6;
    }
    
    return { h: h * 360, s: s * 100, l: l * 100 };
  };
  
  // Convert HSL back to hex
  const hslToHex = (h: number, s: number, l: number): string => {
    h /= 360;
    s /= 100;
    l /= 100;
    
    let r, g, b;
    
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };
      
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    
    const toHex = (x: number) => {
      const hex = Math.round(x * 255).toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };
    
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  };
  
  // Get HSL values of the accent color
  const accentHSL = hexToHSL(accentColor);
  
  // Create complementary colors based on the accent
  const complementary = hslToHex((accentHSL.h + 180) % 360, accentHSL.s, accentHSL.l);
  const analogous1 = hslToHex((accentHSL.h + 30) % 360, accentHSL.s, accentHSL.l);
  const analogous2 = hslToHex((accentHSL.h - 30 + 360) % 360, accentHSL.s, accentHSL.l);
  
  // Adjust error, success, warning colors to harmonize with the accent
  return {
    ...baseTheme,
    accent: accentColor,
    info: analogous1,
    success: baseTheme === darkPalette ? 
      hslToHex((accentHSL.h + 90) % 360, Math.min(accentHSL.s + 10, 100), Math.min(accentHSL.l + 5, 100)) :
      hslToHex((accentHSL.h + 90) % 360, Math.min(accentHSL.s + 5, 100), Math.max(accentHSL.l - 10, 20)),
    warning: baseTheme === darkPalette ? 
      hslToHex((accentHSL.h + 150) % 360, Math.min(accentHSL.s, 100), Math.min(accentHSL.l + 10, 100)) :
      hslToHex((accentHSL.h + 150) % 360, Math.min(accentHSL.s, 100), Math.max(accentHSL.l - 5, 30)),
  };
};

const defaultAccentColor = '#00d4ff'; // Brighter cyan as default

const defaultContext: ThemeContextType = {
  theme: 'dark',
  colors: darkPalette,
  accentColor: defaultAccentColor,
  toggleTheme: () => {},
  setAccentColor: () => {},
  getThemeColor: () => '#000000',
};

const ThemeContext = createContext<ThemeContextType>(defaultContext);

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: ReactNode;
}

// Enhanced predefined accent colors for better visual appeal
export const accentColors = [
  '#00d4ff', // Bright Cyan
  '#3b82f6', // Vibrant Blue
  '#6366f1', // Electric Indigo
  '#a855f7', // Rich Purple
  '#ec4899', // Vibrant Pink
  '#ef4444', // Bright Red
  '#f97316', // Vivid Orange
  '#10b981', // Emerald Green
  '#eab308', // Amber Yellow
  '#14b8a6', // Teal
  '#06b6d4', // Cyan
  '#0ea5e9', // Sky Blue
];

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('dark');
  const [accentColor, setAccentColor] = useState<string>(defaultAccentColor);
  const [colors, setColors] = useState<ColorPalette>(
    generatePalette(theme === 'dark' ? darkPalette : lightPalette, accentColor)
  );

  useEffect(() => {
    // Check if theme preference is stored in localStorage
    const savedTheme = localStorage.getItem('theme') as Theme;
    const savedAccentColor = localStorage.getItem('accentColor');

    if (savedTheme) {
      setTheme(savedTheme);
    }

    if (savedAccentColor) {
      setAccentColor(savedAccentColor);
    }
  }, []);

  // Update colors when theme or accent color changes
  useEffect(() => {
    const basePalette = theme === 'dark' ? darkPalette : lightPalette;
    const newColors = generatePalette(basePalette, accentColor);
    setColors(newColors);
    
    // Apply theme to document body
    document.body.classList.toggle('dark-theme', theme === 'dark');
    document.body.classList.toggle('light-theme', theme === 'light');
    
    // Set CSS variables for colors
    Object.entries(newColors).forEach(([key, value]) => {
      document.documentElement.style.setProperty(`--color-${key}`, value);
    });
    
    // Set additional CSS variables for gradients and effects
    document.documentElement.style.setProperty('--accent-gradient', 
      `linear-gradient(135deg, ${newColors.accent}, ${newColors.info})`);
    
    document.documentElement.style.setProperty('--accent-glow', 
      `0 0 20px ${newColors.accent}80, 0 0 40px ${newColors.accent}40`);
      
    document.documentElement.style.setProperty('--surface-gradient', 
      theme === 'dark' ? 
        `linear-gradient(180deg, ${newColors.surface}, ${newColors.primary})` : 
        `linear-gradient(180deg, ${newColors.primary}, ${newColors.surface})`);
        
    // Apply accent color to all elements with data-accent-color attribute
    document.querySelectorAll('[data-accent-color="true"]').forEach(element => {
      if (element instanceof HTMLElement) {
        element.style.backgroundColor = accentColor;
      }
    });
    
    // Apply gradient backgrounds with accent color
    document.querySelectorAll('[data-accent-gradient="true"]').forEach(element => {
      if (element instanceof HTMLElement) {
        element.style.background = `linear-gradient(135deg, ${accentColor}22 0%, ${theme === 'dark' ? '#000000' : '#ffffff'} 100%)`;
      }
    });
  }, [theme, accentColor]);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Force immediate update of all theme-dependent elements
    const event = new CustomEvent('themechange', { detail: { theme: newTheme } });
    document.dispatchEvent(event);
  };

  const handleSetAccentColor = (color: string) => {
    setAccentColor(color);
    localStorage.setItem('accentColor', color);
    
    // Force immediate update of all accent-color-dependent elements
    const event = new CustomEvent('accentcolorchange', { detail: { color } });
    document.dispatchEvent(event);
  };
  
  const getThemeColor = (colorName: keyof ColorPalette): string => {
    return colors[colorName];
  };

  return (
    <ThemeContext.Provider
      value={{
        theme,
        colors,
        accentColor,
        toggleTheme,
        setAccentColor: handleSetAccentColor,
        getThemeColor,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};
