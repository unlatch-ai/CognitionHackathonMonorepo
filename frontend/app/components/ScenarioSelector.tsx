"use client";

import React, { useState } from 'react';

interface Scenario {
  id: string;
  name: string;
  description: string;
  file: string;
}

interface ScenarioSelectorProps {
  scenarios: Scenario[];
  selectedScenario: string;
  onScenarioChange: (scenarioId: string) => void;
  className?: string;
}

export function ScenarioSelector({ 
  scenarios, 
  selectedScenario, 
  onScenarioChange, 
  className = "" 
}: ScenarioSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  const currentScenario = scenarios.find(s => s.id === selectedScenario);

  const handleSelect = (scenarioId: string) => {
    onScenarioChange(scenarioId);
    setIsOpen(false);
  };

  return (
    <div className={`relative ${className}`}>
      {/* Dropdown Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm hover:bg-white/10 transition-colors"
      >
        <div className="flex flex-col items-start">
          <span className="text-white">{currentScenario?.name || 'Select Scenario'}</span>
          {currentScenario && (
            <span className="text-white/50 text-xs mt-1">{currentScenario.description}</span>
          )}
        </div>
        <svg 
          className={`w-4 h-4 text-white/70 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
          />
          
          {/* Menu */}
          <div className="absolute top-full left-0 right-0 mt-2 bg-black border border-white/20 rounded-lg shadow-2xl z-20 max-h-80 overflow-y-auto">
            {scenarios.map((scenario) => (
              <button
                key={scenario.id}
                onClick={() => handleSelect(scenario.id)}
                className={`w-full px-4 py-3 text-left hover:bg-white/10 transition-colors border-b border-white/10 last:border-b-0 ${
                  selectedScenario === scenario.id ? 'bg-white/5' : ''
                }`}
              >
                <div className="flex flex-col">
                  <span className="text-white font-mono text-sm">{scenario.name}</span>
                  <span className="text-white/50 text-xs mt-1 font-mono">{scenario.description}</span>
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
