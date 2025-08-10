"use client";

import React, { useState, useEffect } from 'react';

interface ModelConfig {
  hasThinking: boolean;
  baseDelay: number;
  varianceRange: number;
  thinkingDelay?: number;
}

interface ScenarioConfig {
  scenarios: {
    [key: string]: {
      name: string;
      description: string;
    };
  };
  models: string[];
  modelConfig: {
    [key: string]: ModelConfig;
  };
}

interface AvailableCombinations {
  models: string[];
  scenarios: string[];
}

interface ScenarioSelectorProps {
  selectedScenario: string;
  selectedModel: string;
  onSelectionChange: (scenario: string, model: string) => void;
  className?: string;
}

export function ScenarioSelector({ 
  selectedScenario, 
  selectedModel,
  onSelectionChange, 
  className = "" 
}: ScenarioSelectorProps) {
  const [scenarioConfig, setScenarioConfig] = useState<ScenarioConfig | null>(null);
  const [isScenarioOpen, setIsScenarioOpen] = useState(false);
  const [isModelOpen, setIsModelOpen] = useState(false);

  // Load scenario configuration
  useEffect(() => {
    fetch('/terminal-scenarios/scenarios.json')
      .then(res => res.json())
      .then(data => setScenarioConfig(data))
      .catch(err => console.error('Failed to load scenarios config:', err));
  }, []);

  const handleScenarioSelect = (scenarioId: string) => {
    onSelectionChange(scenarioId, selectedModel);
    setIsScenarioOpen(false);
  };

  const handleModelSelect = (model: string) => {
    onSelectionChange(selectedScenario, model);
    setIsModelOpen(false);
  };

  if (!scenarioConfig) {
    return (
      <div className={`${className}`}>
        <div className="text-white/50 text-sm font-mono">Loading scenarios...</div>
      </div>
    );
  }

  const currentScenario = scenarioConfig.scenarios[selectedScenario];
  const currentModel = selectedModel;

  return (
    <div className={`${className}`}>
      <div className="grid grid-cols-2 gap-4">
        {/* Scenario Dropdown */}
        <div className="relative">
          <label className="block text-white/70 text-xs font-mono mb-2">Scenario</label>
        <button
          onClick={() => setIsScenarioOpen(!isScenarioOpen)}
          className="w-full flex items-center justify-between px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm hover:bg-white/10 transition-colors"
        >
          <div className="flex flex-col items-start">
            <span className="text-white">{currentScenario?.name || `Scenario ${selectedScenario}`}</span>
            {currentScenario && (
              <span className="text-white/50 text-xs mt-1">{currentScenario.description}</span>
            )}
          </div>
          <svg 
            className={`w-4 h-4 text-white/70 transition-transform ${isScenarioOpen ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Scenario Dropdown Menu */}
        {isScenarioOpen && (
          <>
            <div 
              className="fixed inset-0 z-10" 
              onClick={() => setIsScenarioOpen(false)}
            />
            <div className="absolute top-full left-0 right-0 mt-2 bg-black border border-white/20 rounded-lg shadow-2xl z-20 max-h-60 overflow-y-auto">
              {Object.entries(scenarioConfig.scenarios).map(([id, scenario]) => (
                <button
                  key={id}
                  onClick={() => handleScenarioSelect(id)}
                  className={`w-full px-4 py-3 text-left hover:bg-white/10 transition-colors border-b border-white/10 last:border-b-0 ${
                    selectedScenario === id ? 'bg-white/5' : ''
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

      {/* Model Dropdown */}
      <div className="relative">
        <label className="block text-white/70 text-xs font-mono mb-2">Model</label>
        <button
          onClick={() => setIsModelOpen(!isModelOpen)}
          className="w-full flex items-center justify-between px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm hover:bg-white/10 transition-colors"
        >
          <div className="flex flex-col items-start">
            <span className="text-white capitalize">{currentModel}</span>
            <span className="text-white/50 text-xs mt-1 font-mono">AI Model</span>
          </div>
          <svg 
            className={`w-4 h-4 text-white/70 transition-transform ${isModelOpen ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Model Dropdown Menu */}
        {isModelOpen && (
          <>
            <div 
              className="fixed inset-0 z-10" 
              onClick={() => setIsModelOpen(false)}
            />
            <div className="absolute top-full left-0 right-0 mt-2 bg-black border border-white/20 rounded-lg shadow-2xl z-20 max-h-60 overflow-y-auto">
              {scenarioConfig.models.map((model) => (
                <button
                  key={model}
                  onClick={() => handleModelSelect(model)}
                  className={`w-full px-4 py-3 text-left hover:bg-white/10 transition-colors border-b border-white/10 last:border-b-0 ${
                    selectedModel === model ? 'bg-white/5' : ''
                  }`}
                >
                  <span className="text-white font-mono text-sm capitalize">{model}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
      </div>
    </div>
  );
}
