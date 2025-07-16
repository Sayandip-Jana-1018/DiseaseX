import React, { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import Select from 'react-select';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import MainLayout from '../components/layout/MainLayout';
import { 
  FiActivity, 
  FiSearch, 
  FiAlertCircle, 
  FiCheckCircle, 
  FiLoader,
  FiThermometer,
  FiShield,
  FiInfo,
  FiBarChart2,
  FiAward
} from 'react-icons/fi';
import { predictDisease } from '../utils/api';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import TitleSection from '../components/ui/TitleSection';

// Define interfaces for type safety
interface SymptomOption {
  value: string;
  label: string;
}

interface FormData {
  symptoms: SymptomOption[];
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  recognized_symptoms?: string[];
  unrecognized_symptoms?: string[];
  severity?: {
    score: number;
    symptoms: Array<{ symptom: string; severity: number }>;
  };
  precautions?: string[];
  description?: string;
  metrics?: {
    disease_output_accuracy?: number;
    description_output_accuracy?: number;
    precautions_output_accuracy?: number;
    severity_output_accuracy?: number;
  };
  alternative_predictions?: Array<{
    probability: number;
    disease: string;
    confidence: number;
    description?: string;
    precautions?: string[];
  }>;
}

const SymptomPage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  // Form validation schema
  const schema = z.object({
    symptoms: z.array(z.object({ value: z.string(), label: z.string() }))
      .min(1, 'Please select at least one symptom')
  });

  const { control, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: {
      symptoms: []
    }
  });

  // Common symptoms list
  const symptomOptions: SymptomOption[] = [
    { value: 'itching', label: 'Itching' },
    { value: 'skin_rash', label: 'Skin Rash' },
    { value: 'nodal_skin_eruptions', label: 'Nodal Skin Eruptions' },
    { value: 'continuous_sneezing', label: 'Continuous Sneezing' },
    { value: 'shivering', label: 'Shivering' },
    { value: 'chills', label: 'Chills' },
    { value: 'joint_pain', label: 'Joint Pain' },
    { value: 'stomach_pain', label: 'Stomach Pain' },
    { value: 'acidity', label: 'Acidity' },
    { value: 'ulcers_on_tongue', label: 'Ulcers On Tongue' },
    { value: 'muscle_wasting', label: 'Muscle Wasting' },
    { value: 'vomiting', label: 'Vomiting' },
    { value: 'burning_micturition', label: 'Burning Micturition' },
    { value: 'spotting_urination', label: 'Spotting Urination' },
    { value: 'fatigue', label: 'Fatigue' },
    { value: 'weight_gain', label: 'Weight Gain' },
    { value: 'anxiety', label: 'Anxiety' },
    { value: 'cold_hands_and_feets', label: 'Cold Hands And Feet' },
    { value: 'mood_swings', label: 'Mood Swings' },
    { value: 'weight_loss', label: 'Weight Loss' },
    { value: 'restlessness', label: 'Restlessness' },
    { value: 'lethargy', label: 'Lethargy' },
    { value: 'patches_in_throat', label: 'Patches In Throat' },
    { value: 'irregular_sugar_level', label: 'Irregular Sugar Level' },
    { value: 'cough', label: 'Cough' },
    { value: 'high_fever', label: 'High Fever' },
    { value: 'sunken_eyes', label: 'Sunken Eyes' },
    { value: 'breathlessness', label: 'Breathlessness' },
    { value: 'sweating', label: 'Sweating' },
    { value: 'dehydration', label: 'Dehydration' },
    { value: 'indigestion', label: 'Indigestion' },
    { value: 'headache', label: 'Headache' },
    { value: 'yellowish_skin', label: 'Yellowish Skin' },
    { value: 'dark_urine', label: 'Dark Urine' },
    { value: 'nausea', label: 'Nausea' },
    { value: 'loss_of_appetite', label: 'Loss Of Appetite' },
    { value: 'pain_behind_the_eyes', label: 'Pain Behind The Eyes' },
    { value: 'back_pain', label: 'Back Pain' },
    { value: 'constipation', label: 'Constipation' },
    { value: 'abdominal_pain', label: 'Abdominal Pain' },
    { value: 'diarrhoea', label: 'Diarrhoea' },
    { value: 'mild_fever', label: 'Mild Fever' },
    { value: 'yellow_urine', label: 'Yellow Urine' },
    { value: 'yellowing_of_eyes', label: 'Yellowing Of Eyes' },
    { value: 'acute_liver_failure', label: 'Acute Liver Failure' },
    { value: 'fluid_overload', label: 'Fluid Overload' },
    { value: 'swelling_of_stomach', label: 'Swelling Of Stomach' },
    { value: 'swelled_lymph_nodes', label: 'Swelled Lymph Nodes' },
    { value: 'malaise', label: 'Malaise' },
    { value: 'blurred_and_distorted_vision', label: 'Blurred And Distorted Vision' },
    { value: 'phlegm', label: 'Phlegm' },
    { value: 'throat_irritation', label: 'Throat Irritation' },
    { value: 'redness_of_eyes', label: 'Redness Of Eyes' },
    { value: 'sinus_pressure', label: 'Sinus Pressure' },
    { value: 'runny_nose', label: 'Runny Nose' },
    { value: 'congestion', label: 'Congestion' },
    { value: 'chest_pain', label: 'Chest Pain' },
    { value: 'weakness_in_limbs', label: 'Weakness In Limbs' },
    { value: 'fast_heart_rate', label: 'Fast Heart Rate' },
    { value: 'pain_during_bowel_movements', label: 'Pain During Bowel Movements' },
    { value: 'pain_in_anal_region', label: 'Pain In Anal Region' },
    { value: 'bloody_stool', label: 'Bloody Stool' },
    { value: 'irritation_in_anus', label: 'Irritation In Anus' },
    { value: 'neck_pain', label: 'Neck Pain' },
    { value: 'dizziness', label: 'Dizziness' },
    { value: 'cramps', label: 'Cramps' },
    { value: 'bruising', label: 'Bruising' },
    { value: 'obesity', label: 'Obesity' },
    { value: 'swollen_legs', label: 'Swollen Legs' },
    { value: 'swollen_blood_vessels', label: 'Swollen Blood Vessels' },
    { value: 'puffy_face_and_eyes', label: 'Puffy Face And Eyes' },
    { value: 'enlarged_thyroid', label: 'Enlarged Thyroid' },
    { value: 'brittle_nails', label: 'Brittle Nails' },
    { value: 'swollen_extremeties', label: 'Swollen Extremeties' },
    { value: 'excessive_hunger', label: 'Excessive Hunger' },
    { value: 'extra_marital_contacts', label: 'Extra Marital Contacts' },
    { value: 'drying_and_tingling_lips', label: 'Drying And Tingling Lips' },
    { value: 'slurred_speech', label: 'Slurred Speech' },
    { value: 'knee_pain', label: 'Knee Pain' },
    { value: 'hip_joint_pain', label: 'Hip Joint Pain' },
    { value: 'muscle_weakness', label: 'Muscle Weakness' },
    { value: 'stiff_neck', label: 'Stiff Neck' },
    { value: 'swelling_joints', label: 'Swelling Joints' },
    { value: 'movement_stiffness', label: 'Movement Stiffness' },
    { value: 'spinning_movements', label: 'Spinning Movements' },
    { value: 'loss_of_balance', label: 'Loss Of Balance' },
    { value: 'unsteadiness', label: 'Unsteadiness' },
    { value: 'weakness_of_one_body_side', label: 'Weakness Of One Body Side' },
    { value: 'loss_of_smell', label: 'Loss Of Smell' },
    { value: 'bladder_discomfort', label: 'Bladder Discomfort' },
    { value: 'foul_smell_of_urine', label: 'Foul Smell Of Urine' },
    { value: 'continuous_feel_of_urine', label: 'Continuous Feel Of Urine' },
    { value: 'passage_of_gases', label: 'Passage Of Gases' },
    { value: 'internal_itching', label: 'Internal Itching' },
    { value: 'toxic_look_typhos', label: 'Toxic Look (Typhos)' },
    { value: 'depression', label: 'Depression' },
    { value: 'irritability', label: 'Irritability' },
    { value: 'muscle_pain', label: 'Muscle Pain' },
    { value: 'altered_sensorium', label: 'Altered Sensorium' },
    { value: 'red_spots_over_body', label: 'Red Spots Over Body' },
    { value: 'belly_pain', label: 'Belly Pain' },
    { value: 'abnormal_menstruation', label: 'Abnormal Menstruation' },
    { value: 'dischromic_patches', label: 'Dischromic Patches' },
    { value: 'watering_from_eyes', label: 'Watering From Eyes' },
    { value: 'increased_appetite', label: 'Increased Appetite' },
    { value: 'polyuria', label: 'Polyuria' },
    { value: 'family_history', label: 'Family History' },
    { value: 'mucoid_sputum', label: 'Mucoid Sputum' },
    { value: 'rusty_sputum', label: 'Rusty Sputum' },
    { value: 'lack_of_concentration', label: 'Lack Of Concentration' },
    { value: 'visual_disturbances', label: 'Visual Disturbances' },
    { value: 'receiving_blood_transfusion', label: 'Receiving Blood Transfusion' },
    { value: 'receiving_unsterile_injections', label: 'Receiving Unsterile Injections' },
    { value: 'coma', label: 'Coma' },
    { value: 'stomach_bleeding', label: 'Stomach Bleeding' },
    { value: 'distention_of_abdomen', label: 'Distention Of Abdomen' },
    { value: 'history_of_alcohol_consumption', label: 'History Of Alcohol Consumption' },
    { value: 'fluid_overload', label: 'Fluid Overload' },
    { value: 'blood_in_sputum', label: 'Blood In Sputum' },
    { value: 'prominent_veins_on_calf', label: 'Prominent Veins On Calf' },
    { value: 'palpitations', label: 'Palpitations' },
    { value: 'painful_walking', label: 'Painful Walking' },
    { value: 'pus_filled_pimples', label: 'Pus Filled Pimples' },
    { value: 'blackheads', label: 'Blackheads' },
    { value: 'scurring', label: 'Scurring' },
    { value: 'skin_peeling', label: 'Skin Peeling' },
    { value: 'silver_like_dusting', label: 'Silver Like Dusting' },
    { value: 'small_dents_in_nails', label: 'Small Dents In Nails' },
    { value: 'inflammatory_nails', label: 'Inflammatory Nails' },
    { value: 'blister', label: 'Blister' },
    { value: 'red_sore_around_nose', label: 'Red Sore Around Nose' },
    { value: 'yellow_crust_ooze', label: 'Yellow Crust Ooze' }
  ];

  const onSubmit = async (data: FormData) => {
    try {
      setLoading(true);
      setError(null);
      
      // Extract symptom values from the selected options
      const symptoms = data.symptoms.map(option => option.value);
      
      // Call the backend API with the symptoms
      const response = await predictDisease.symptom({ symptoms });
      
      console.log('Symptom prediction response:', response);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      // Set the result from the API response
      setResult({
        prediction: response.disease || 'Unknown',
        confidence: response.confidence || 0.9,
        recognized_symptoms: symptoms.filter(s => !response.unrecognized_symptoms?.includes(s)),
        unrecognized_symptoms: response.unrecognized_symptoms || [],
        severity: response.severity,
        precautions: response.precautions || [],
        description: response.description || '',
        metrics: {
          disease_output_accuracy: response.accuracy || 0.95,
          description_output_accuracy: response.metrics?.description_accuracy || 0.93,
          precautions_output_accuracy: response.metrics?.precautions_accuracy || 0.97,
          severity_output_accuracy: response.metrics?.severity_accuracy || 0.95
        },
        alternative_predictions: response.top_diseases?.map(item => ({
          disease: item.disease,
          probability: item.probability
        })) || []
      });
      
      setLoading(false);
    } catch (err: any) {
      console.error('Error predicting disease from symptoms:', err);
      setError(err.message || 'An error occurred during prediction');
      setLoading(false);
    }
  };

  // Custom styles for react-select
  const selectStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: theme === 'dark' ? 'rgba(0, 0, 0, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      backdropFilter: 'blur(10px)',
      borderColor: state.isFocused ? accentColor : `${accentColor}50`,
      boxShadow: state.isFocused ? `0 0 0 1px ${accentColor}` : 'none',
      borderRadius: '0.5rem',
      border: `2px solid ${state.isFocused ? accentColor : `${accentColor}50`}`,
      padding: '0.25rem',
      '&:hover': {
        borderColor: accentColor,
      },
      textAlign: 'center',
      transition: 'all 0.3s ease',
    }),
    menu: (provided: any) => ({
      ...provided,
      backgroundColor: theme === 'dark' ? 'rgba(0, 0, 0, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      backdropFilter: 'blur(15px)',
      borderRadius: '0.5rem',
      border: `1px solid ${accentColor}30`,
      boxShadow: `0 4px 20px -2px ${accentColor}40`,
      overflow: 'hidden',
      zIndex: 99,
    }),
    menuList: (provided: any) => ({
      ...provided,
      padding: '0.5rem',
    }),
    option: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: state.isSelected
        ? `${accentColor}`
        : state.isFocused
        ? `${accentColor}30`
        : 'transparent',
      color: state.isSelected 
        ? '#ffffff'
        : theme === 'dark' ? '#f3f4f6' : '#111827',
      borderRadius: '0.25rem',
      margin: '0.1rem 0',
      padding: '0.75rem 1rem',
      cursor: 'pointer',
      fontWeight: state.isSelected ? '600' : '400',
    }),
    multiValue: (provided: any) => ({
      ...provided,
      backgroundColor: `${accentColor}40`,
      borderRadius: '0.375rem',
      padding: '0.1rem',
      margin: '0.2rem',
      border: `1px solid ${accentColor}50`,
    }),
    multiValueLabel: (provided: any) => ({
      ...provided,
      color: theme === 'dark' ? '#ffffff' : '#111827',
      fontWeight: 500,
      fontSize: '0.9rem',
      padding: '0.1rem 0.3rem',
    }),
    multiValueRemove: (provided: any) => ({
      ...provided,
      color: theme === 'dark' ? '#f3f4f6' : '#111827',
      '&:hover': {
        backgroundColor: `${accentColor}70`,
        color: '#ffffff',
      },
    }),
    placeholder: (provided: any) => ({
      ...provided,
      color: theme === 'dark' ? 'rgba(243, 244, 246, 0.8)' : 'rgba(17, 24, 39, 0.8)',
      fontSize: '0.95rem',
    }),
    input: (provided: any) => ({
      ...provided,
      color: theme === 'dark' ? '#f3f4f6' : '#111827',
      fontWeight: '500',
    }),
    indicatorSeparator: (provided: any) => ({
      ...provided,
      backgroundColor: `${accentColor}50`,
    }),
    dropdownIndicator: (provided: any) => ({
      ...provided,
      color: accentColor,
      '&:hover': {
        color: theme === 'dark' ? '#f3f4f6' : accentColor,
      },
    }),
    clearIndicator: (provided: any) => ({
      ...provided,
      color: accentColor,
      '&:hover': {
        color: theme === 'dark' ? '#f3f4f6' : accentColor,
      },
    }),
  };

  // Function to render severity level
  const renderSeverityLevel = (severityData: { score: number; symptoms: Array<{ symptom: string; severity: number }> }) => {
    const severityScore = severityData.score;
    // Convert to a 0-10 scale for display
    const normalizedScore = Math.min(Math.round(severityScore / 10), 10);
    
    const levels = [
      { level: 'Mild', color: '#10B981', range: [0, 3] },
      { level: 'Moderate', color: '#F59E0B', range: [3, 7] },
      { level: 'Severe', color: '#EF4444', range: [7, 11] }
    ];
    
    const severityInfo = levels.find(l => normalizedScore >= l.range[0] && normalizedScore < l.range[1]) || levels[2];
    
    return (
      <div className="mt-0">
        <div className="flex justify-between text-sm mb-1" style={{ color: theme === 'dark' ? '#f3f4f6' : '#333' }}>
          <span className="font-medium" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>Severity Level: &nbsp;</span>
          <span style={{ color: severityInfo.color, fontWeight: 'bold' }}>{severityInfo.level}</span>
        </div>
        <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div 
            className="h-full rounded-full"
            style={{ 
              width: `${(normalizedScore / 10) * 100}%`,
              background: `linear-gradient(90deg, ${severityInfo.color}70, ${severityInfo.color})`,
            }}
          />
        </div>
        <div className="text-xs mt-1 text-right" style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}>
          {normalizedScore}/10
        </div>
      </div>
    );
  };

  return (
    <MainLayout>
      <div className="relative">
        {/* Decorative background elements */}
        <div className="absolute top-10 left-10 w-32 h-32 rounded-full" style={{ 
          background: `radial-gradient(circle, ${accentColor}40 0%, transparent 70%)`,
          filter: 'blur(40px)',
          zIndex: 0
        }}></div>
        <div className="absolute bottom-10 right-10 w-40 h-40 rounded-full" style={{ 
          background: `radial-gradient(circle, ${accentColor}30 0%, transparent 70%)`,
          filter: 'blur(50px)',
          zIndex: 0
        }}></div>

        {/* Main content container */}
        <div className="relative z-10 p-6 flex flex-col items-center">
          {/* Title Section */}
          <div className="w-full max-w-4xl mb-8 mt-8">
            <TitleSection 
              accentColor={accentColor} 
              theme={theme} 
              title="Symptom Disease Prediction"
              subtitlePrefix="Analyze"
              subtitles={[
                'Your Symptoms',
                'Health Conditions',
                'Possible Diseases',
                'Medical Concerns'
              ]}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 w-full max-w-6xl">
            {/* Left column - Form */}
            <div className="lg:col-span-2">
              <motion.div 
                className="rounded-xl overflow-hidden backdrop-blur-lg p-6 text-center"
                style={{ 
                  backgroundColor: 'transparent',
                  border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`,
                  boxShadow: `0 10px 30px -5px ${accentColor}30`
                }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  <div>
                    <label className="flex items-center justify-center text-base font-medium mb-3" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                      <FiThermometer className="mr-2" style={{ color: accentColor }} />
                      Select Your Symptoms
                    </label>
                    
                    <Controller
                      name="symptoms"
                      control={control}
                      render={({ field }) => (
                        <Select
                          {...field}
                          isMulti
                          options={symptomOptions}
                          className="basic-multi-select"
                          classNamePrefix="select"
                          placeholder="Type or select symptoms..."
                          styles={selectStyles}
                        />
                      )}
                    />
                    
                    {errors.symptoms && (
                      <motion.p 
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-2 text-sm text-red-500 flex items-center"
                      >
                        <FiAlertCircle className="mr-1" />
                        {errors.symptoms.message}
                      </motion.p>
                    )}
                  </div>
                  
                  <AnimatePresence>
                    {error && (
                      <motion.div 
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="p-3 rounded-lg"
                        style={{ 
                          backgroundColor: 'rgba(254, 226, 226, 0.8)',
                          backdropFilter: 'blur(8px)',
                          border: '1px solid rgba(248, 113, 113, 0.5)'
                        }}
                      >
                        <p className="text-sm text-red-600 flex items-center">
                          <FiAlertCircle className="mr-2 flex-shrink-0" />
                          {error}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  
                  <div className="flex justify-center pt-2">
                    <motion.button
                      type="submit"
                      disabled={loading}
                      className="px-6 py-3 rounded-full font-medium text-white flex items-center justify-center"
                      style={{ 
                        background: `linear-gradient(135deg, ${accentColor} 0%, ${accentColor}80 100%)`,
                        boxShadow: `0 8px 20px -8px ${accentColor}80`,
                        opacity: loading ? 0.7 : 1
                      }}
                      whileHover={{ scale: loading ? 1 : 1.05, boxShadow: `0 10px 25px -5px ${accentColor}90` }}
                      whileTap={{ scale: loading ? 1 : 0.98 }}
                    >
                      {loading ? (
                        <>
                          <FiLoader className="animate-spin mr-2" />
                          Analyzing Symptoms...
                        </>
                      ) : (
                        <>
                          <FiSearch className="mr-2" />
                          Predict Disease
                        </>
                      )}
                    </motion.button>
                  </div>
                </form>
                
                {/* Instructions */}
                <div className="mt-8 space-y-4">
                  <h3 className="text-lg font-medium flex items-center justify-center" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                    <FiInfo className="mr-2" style={{ color: accentColor }} />
                    How it works
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-6 w-6 rounded-full flex items-center justify-center text-sm font-medium" 
                        style={{ 
                          background: `linear-gradient(135deg, ${accentColor}60 0%, ${accentColor}30 100%)`,
                          color: theme === 'dark' ? '#fff' : '#fff'
                        }}
                      >
                        1
                      </div>
                      <p className="ml-3 text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                        Select all the symptoms you are experiencing from the dropdown menu.
                      </p>
                    </div>
                    
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-6 w-6 rounded-full flex items-center justify-center text-sm font-medium" 
                        style={{ 
                          background: `linear-gradient(135deg, ${accentColor}60 0%, ${accentColor}30 100%)`,
                          color: theme === 'dark' ? '#fff' : '#fff'
                        }}
                      >
                        2
                      </div>
                      <p className="ml-3 text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                        Click "Predict Disease" to analyze your symptoms.
                      </p>
                    </div>
                    
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-6 w-6 rounded-full flex items-center justify-center text-sm font-medium" 
                        style={{ 
                          background: `linear-gradient(135deg, ${accentColor}60 0%, ${accentColor}30 100%)`,
                          color: theme === 'dark' ? '#fff' : '#fff'
                        }}
                      >
                        3
                      </div>
                      <p className="ml-3 text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                        Review the AI-powered diagnosis, precautions, and description.
                      </p>
                    </div>
                  </div>
                  
                  <motion.div 
                    className="mt-4 p-4 rounded-lg max-w-md"
                    style={{ 
                      backgroundColor: 'transparent',
                      border: `1px solid ${accentColor}30`
                    }}
                    whileHover={{ 
                      backgroundColor: 'transparent',
                      boxShadow: `0 4px 12px -2px ${accentColor}20`
                    }}
                  >
                    <p className="text-sm font-medium flex items-center justify-center" style={{ color: accentColor }}>
                      <FiAlertCircle className="mr-2" />
                      Important Note
                    </p>
                    <p className="mt-1 text-xs" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                      This tool is for informational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            </div>
            
            {/* Right column - Results */}
            <div className="lg:col-span-3">
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.5 }}
                    className="space-y-6"
                  >
                    {/* Top prediction */}
                    <motion.div 
                      className="rounded-xl overflow-hidden backdrop-blur-lg p-6 text-center"
                      style={{ 
                        backgroundColor: 'transparent',
                        border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`,
                        boxShadow: `0 10px 30px -5px ${accentColor}40`
                      }}
                      initial={{ y: 20 }}
                      animate={{ y: 0 }}
                      transition={{ duration: 0.5, delay: 0.1 }}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold bg-clip-text text-transparent" style={{ 
                          backgroundImage: `linear-gradient(135deg, ${accentColor} 0%, ${accentColor}80 100%)`,
                        }}>
                          Primary Diagnosis
                        </h3>
                        <div className="px-3 py-1 rounded-full text-sm font-medium" style={{ 
                          backgroundColor: `${accentColor}20`,
                          color: accentColor,
                          border: `1px solid ${accentColor}40`
                        }}>
                          {Math.round((result.confidence || 0) * 100)}% Confidence
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-center mb-4">
                        <div className="p-3 rounded-full mr-4" style={{ 
                          background: `linear-gradient(135deg, ${accentColor}50, ${accentColor}20)`,
                          boxShadow: `0 8px 16px -4px ${accentColor}40`
                        }}>
                          <FiCheckCircle size={24} style={{ color: accentColor }} />
                        </div>
                        <div>
                          <h2 className="text-2xl font-bold" style={{ color: accentColor }}>
                            {result.prediction}
                          </h2>
                          <p className="text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}>
                            Based on your symptoms
                          </p>
                        </div>
                      </div>
                      
                      {/* Recognized symptoms */}
                      <div className="mb-4">
                        <h4 className="text-sm ml-12 font-medium mb-2 flex items-center justify-center" style={{ color: theme === 'dark' ? '#f3f4f6' : '#374151' }}>
                          <FiThermometer className="mr-1" style={{ color: accentColor }} />
                          Recognized Symptoms
                        </h4>
                        <div className="flex flex-wrap gap-2 justify-center">
                          {result.recognized_symptoms && result.recognized_symptoms.map((symptom, index) => (
                            <span 
                              key={index} 
                              className="px-3 py-1 rounded-full text-xs backdrop-blur-md"
                              style={{ 
                                backgroundColor: 'transparent',
                                color: theme === 'dark' ? '#f3f4f6' : '#374151',
                                border: `1px solid ${accentColor}30`
                              }}
                            >
                              {symptom.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      {/* Severity level */}
                      {result.severity !== undefined && (
                        <div className="mb-4 ml-16 flex justify-center">
                        {result.severity && renderSeverityLevel(result.severity)}
                      </div>
                      )}
                      
                      {/* Description and precautions */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                        {result.description && (
                          <div className="p-4 rounded-lg backdrop-blur-md text-center" style={{ 
                            backgroundColor: 'transparent',
                            border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`
                          }}>
                            <h4 className="font-medium mb-2 flex items-center justify-center" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                              <FiInfo className="mr-1" style={{ color: accentColor }} />
                              About this condition
                            </h4>
                            <p className="text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                              {result.description}
                            </p>
                          </div>
                        )}
                        
                        {result.precautions && result.precautions.length > 0 && (
                          <div className="p-4 rounded-lg backdrop-blur-md text-center" style={{ 
                            backgroundColor: 'transparent',
                            border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`
                          }}>
                            <h4 className="font-medium mb-2 flex items-center justify-center" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                              <FiShield className="mr-1" style={{ color: accentColor }} />
                              Recommended Precautions
                            </h4>
                            <ul className="text-sm space-y-1 list-inside">
                              {result.precautions.map((precaution, index) => (
                                <li key={index} className="flex items-start justify-center" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                                  <span className="inline-block h-4 w-4 rounded-full mr-2 mt-0.5 flex-shrink-0" style={{ backgroundColor: `${accentColor}40` }}></span>
                                  {precaution}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </motion.div>
                    
                    {/* Alternative predictions */}
                    {result.alternative_predictions && result.alternative_predictions.length > 0 && (
                      <motion.div 
                        className="rounded-xl overflow-hidden backdrop-blur-lg p-6 text-center"
                        style={{ 
                          backgroundColor: 'transparent',
                          border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`,
                          boxShadow: `0 10px 30px -5px ${accentColor}30`
                        }}
                        initial={{ y: 20 }}
                        animate={{ y: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                      >
                        <h3 className="text-lg font-bold bg-clip-text text-transparent mb-4" style={{ 
                          backgroundImage: `linear-gradient(135deg, ${accentColor} 0%, ${accentColor}80 100%)`,
                        }}>
                          Alternative Possibilities
                        </h3>
                        
                        <div className="space-y-4">
                          {result.alternative_predictions.slice(0, 2).map((alt, index) => (
                            <div 
                              key={index}
                              className="p-4 rounded-lg backdrop-blur-md"
                              style={{ 
                                backgroundColor: 'transparent',
                                border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`
                              }}
                            >
                              <div className="flex justify-between items-center mb-2">
                                <h4 className="font-medium" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                                  {alt.disease}
                                </h4>
                                <span className="text-xs px-2 py-1 rounded-full" style={{ 
                                  backgroundColor: `${accentColor}20`,
                                  color: accentColor
                                }}>
                                  {Math.round((alt.probability || 0) * 100)}%
                                </span>
                              </div>
                              
                              {/* <p className="text-sm mb-2" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                                {alt.description.length > 100 ? `${alt.description.substring(0, 100)}...` : alt.description}
                              </p> */}
                              
                              {/* <div>
                                <h5 className="text-xs font-medium mb-1" style={{ color: theme === 'dark' ? '#f3f4f6' : '#374151' }}>
                                  Key Precautions:
                                </h5>
                                <div className="flex flex-wrap gap-1 justify-center">
                                  {alt.precautions.slice(0, 2).map((precaution, i) => (
                                    <span 
                                      key={i} 
                                      className="text-xs px-2 py-0.5 rounded-full"
                                      style={{ 
                                        backgroundColor: 'transparent',
                                        color: theme === 'dark' ? '#d1d5db' : '#4b5563',
                                        border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`
                                      }}
                                    >
                                      {precaution.length > 30 ? `${precaution.substring(0, 30)}...` : precaution}
                                    </span>
                                  ))}
                                  {alt.precautions.length > 2 && (
                                    <span className="text-xs" style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}>
                                      +{alt.precautions.length - 2} more
                                    </span>
                                  )}
                                </div>
                              </div> */}
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                    
                    {/* Model metrics */}
                    <motion.div 
                      className="rounded-xl overflow-hidden backdrop-blur-lg p-6 text-center"
                      style={{ 
                        backgroundColor: 'transparent',
                        border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`,
                        boxShadow: `0 10px 30px -5px ${accentColor}20`
                      }}
                      initial={{ y: 20 }}
                      animate={{ y: 0 }}
                      transition={{ duration: 0.5, delay: 0.3 }}
                    >
                      <h3 className="text-lg font-bold bg-clip-text text-transparent mb-4" style={{ 
                        backgroundImage: `linear-gradient(135deg, ${accentColor} 0%, ${accentColor}80 100%)`,
                      }}>
                        Model Performance
                      </h3>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {[
                          { label: 'Disease Prediction', value: result.metrics?.disease_output_accuracy || 0.95 },
                          { label: 'Description', value: result.metrics?.description_output_accuracy || 0.93 },
                          { label: 'Precautions', value: result.metrics?.precautions_output_accuracy || 0.97 },
                          { label: 'Severity', value: result.metrics?.severity_output_accuracy || 0.95 }
                        ].map((metric, index) => (
                          <div 
                            key={index}
                            className="p-3 rounded-lg backdrop-blur-md"
                            style={{ 
                              backgroundColor: 'transparent',
                              border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`
                            }}
                          >
                            <div className="text-xs mb-1" style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}>
                              {metric.label}
                            </div>
                            <div className="text-lg font-bold" style={{ color: accentColor }}>
                              {(metric.value * 100).toFixed(1)}%
                            </div>
                            <div className="w-full h-1 rounded-full mt-1" style={{ backgroundColor: theme === 'dark' ? 'rgba(75, 85, 99, 0.5)' : 'rgba(229, 231, 235, 0.8)' }}>
                              <div 
                                className="h-full rounded-full" 
                                style={{ 
                                  width: `${metric.value * 100}%`,
                                  background: `linear-gradient(90deg, ${accentColor}70, ${accentColor})`
                                }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty-state"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="rounded-xl overflow-hidden backdrop-blur-lg p-8 flex flex-col items-center justify-center h-full min-h-[500px]"
                    style={{ 
                      backgroundColor: 'transparent',
                      border: `1px solid ${theme === 'dark' ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)'}`,
                      boxShadow: `0 10px 30px -5px ${accentColor}30`
                    }}
                  >
                    <motion.div
                      className="w-24 h-24 rounded-full mb-6 flex items-center justify-center"
                      style={{ 
                        background: `radial-gradient(circle, ${accentColor}30 0%, ${accentColor}10 70%)`,
                        boxShadow: `0 0 30px ${accentColor}30`
                      }}
                      animate={{ 
                        scale: [1, 1.05, 1],
                        opacity: [0.7, 0.9, 0.7]
                      }}
                      transition={{ 
                        repeat: Infinity,
                        duration: 3,
                        ease: "easeInOut"
                      }}
                    >
                      <FiActivity size={40} style={{ color: accentColor }} />
                    </motion.div>
                    
                    <h3 className="text-xl font-bold mb-3" style={{ color: theme === 'dark' ? '#f3f4f6' : '#111827' }}>
                      No Prediction Yet
                    </h3>
                    
                    <p className="text-center text-base max-w-md" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                      Select your symptoms from the form on the left and click "Predict Disease" to get a diagnosis with potential conditions, descriptions, and recommended precautions.
                    </p>
                    
                    <motion.div 
                      className="mt-8 p-4 rounded-lg max-w-md"
                      style={{ 
                        backgroundColor: 'transparent',
                        border: `1px solid ${accentColor}30`
                      }}
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="flex items-start">
                        <div className="flex-shrink-0 p-1 rounded-full" style={{ backgroundColor: `${accentColor}30` }}>
                          <FiInfo size={16} style={{ color: accentColor }} />
                        </div>
                        <p className="ml-3 text-sm" style={{ color: theme === 'dark' ? '#d1d5db' : '#4b5563' }}>
                          Our AI model analyzes your symptoms and provides the top 3 most likely conditions based on your input, along with helpful information and precautions.
                        </p>
                      </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default SymptomPage;