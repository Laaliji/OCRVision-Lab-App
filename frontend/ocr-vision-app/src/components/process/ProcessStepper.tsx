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
  return (
    <div className="mb-12">
      {/* Progress Bar */}
      <div className="relative mb-16">
        <div className="w-full h-2 bg-gray-200 rounded-full">
          <div 
            className="h-full bg-indigo-500 rounded-full transition-all duration-300" 
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Step Indicators */}
        <div className="absolute -top-2 left-0 right-0">
          <div className="flex justify-between">
            {steps.map((step) => (
              <div 
                key={step.id} 
                className="relative w-24 text-center"
                onClick={() => onStepClick && onStepClick(step.id)}
                style={{ cursor: onStepClick ? 'pointer' : 'default' }}
              >
                <div className={`
                  w-6 h-6 mx-auto rounded-full flex items-center justify-center text-sm font-bold mb-2 relative z-10
                  ${step.isCompleted 
                    ? 'bg-green-500 text-white' 
                    : step.isActive 
                      ? 'bg-indigo-500 text-white' 
                      : 'bg-gray-300 text-white'}
                `}>
                  {step.isCompleted ? <FontAwesomeIcon icon={faCheck} /> : step.id}
                </div>
                <p className={`
                  text-xs font-medium mt-2
                  ${step.isActive || step.isCompleted ? 'text-gray-700' : 'text-gray-500'}
                `}>
                  {step.title}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessStepper; 