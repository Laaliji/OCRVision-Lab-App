import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { ProcessStep } from '../../types/types';

interface ProcessStepperProps {
  steps: ProcessStep[];
  currentStep: number;
  progress: number;
  onStepClick?: (stepNumber: number) => void;
}

const ProcessStepper: React.FC<ProcessStepperProps> = ({ steps, currentStep, progress, onStepClick }) => {
  const handleStepClick = (stepNumber: number) => {
    // Only allow clicking on completed steps or the current step
    if (onStepClick && (stepNumber <= currentStep)) {
      onStepClick(stepNumber);
    }
  };
  
  return (
    <div className="mb-10">
      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-6">
        <div 
          className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500 ease-in-out" 
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      
      {/* Steps */}
      <div className="flex justify-between">
        {steps.map(step => (
          <div 
            key={step.id}
            className={`flex flex-col items-center cursor-pointer transition-colors ${
              step.id <= currentStep ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'
            }`}
            onClick={() => handleStepClick(step.id)}
          >
            <div className={`
              w-10 h-10 flex items-center justify-center rounded-full mb-2
              ${step.isActive ? 'bg-indigo-600 text-white' : 
                step.isCompleted ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-500'}
              transition-colors duration-300
            `}>
              {step.isCompleted ? (
                <FontAwesomeIcon icon={faCheck} />
              ) : (
                step.id
              )}
            </div>
            <div className="text-center">
              <p className={`text-sm font-medium ${
                step.isActive ? 'text-indigo-600' : 
                step.isCompleted ? 'text-green-600' : 'text-gray-500'
              }`}>
                {step.title}
              </p>
              <p className="text-xs text-gray-500 hidden md:block">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessStepper; 