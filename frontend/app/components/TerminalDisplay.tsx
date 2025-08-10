"use client";

import React, { useState, useEffect, useRef } from 'react';
import { ScenarioSelector } from './ScenarioSelector';

interface Command {
  id: number;
  command: string;
  full_prompt: string;
  response: string;
}

interface TerminalData {
  system_prompt: string;
  goal: string;
  commands: Command[];
}

interface TerminalDisplayProps {
  className?: string;
}

export function TerminalDisplay({ className = "" }: TerminalDisplayProps) {
  const [data, setData] = useState<TerminalData | null>(null);
  const [selectedScenario, setSelectedScenario] = useState("1");
  const [selectedModel, setSelectedModel] = useState("gpt4");
  const [modelConfig, setModelConfig] = useState<any>(null);
  const [displayedCommands, setDisplayedCommands] = useState<Command[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [typingCommand, setTypingCommand] = useState("");
  const [currentTypingOutput, setCurrentTypingOutput] = useState("");
  const [userScrolled, setUserScrolled] = useState(false);
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [agentCompleted, setAgentCompleted] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const agentTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Load model configuration
  useEffect(() => {
    fetch('/terminal-scenarios/scenarios.json')
      .then(res => res.json())
      .then(data => setModelConfig(data.modelConfig))
      .catch(err => console.error('Failed to load model config:', err));
  }, []);

  // Handle selection changes
  const handleSelectionChange = (scenario: string, model: string) => {
    setSelectedScenario(scenario);
    setSelectedModel(model);
  };

  // Generate realistic delay based on model configuration
  const getModelDelay = (isThinkingPhase = false) => {
    if (!modelConfig || !modelConfig[selectedModel]) {
      return 1500; // Default fallback
    }
    
    const config = modelConfig[selectedModel];
    
    if (isThinkingPhase && config.hasThinking && config.thinkingDelay) {
      // Variable thinking times: 30% short (500-1200ms), 50% medium (1200-2500ms), 20% long (2500-4000ms)
      const rand = Math.random();
      if (rand < 0.3) {
        // Short thinking
        return 500 + Math.floor(Math.random() * 700);
      } else if (rand < 0.8) {
        // Medium thinking
        return 1200 + Math.floor(Math.random() * 1300);
      } else {
        // Long thinking
        return 2500 + Math.floor(Math.random() * 1500);
      }
    }
    
    // Normal command delay with variance
    const variance = Math.floor(Math.random() * config.varianceRange) - (config.varianceRange / 2);
    return Math.max(500, config.baseDelay + variance);
  };

  // Load scenario data based on model and scenario selection
  useEffect(() => {
    const fileName = `${selectedModel}_${selectedScenario}.json`;
    const filePath = `/terminal-scenarios/${fileName}`;
    
    fetch(filePath)
      .then((res) => res.json())
      .then((scenarioData) => {
        setData(scenarioData);
        // Reset state when scenario changes
        setDisplayedCommands([]);
        setIsAgentRunning(false);
        setAgentCompleted(false);
        setIsTyping(false);
        setIsThinking(false);
        setTypingCommand("");
        setCurrentTypingOutput("");
      })
      .catch((err) => console.log(`Scenario data not found for ${fileName}:`, err));
  }, [selectedScenario, selectedModel]);

  // Auto-scroll to bottom unless user has manually scrolled up
  const scrollToBottom = () => {
    if (terminalRef.current && !userScrolled) {
      terminalRef.current.scrollTo({
        top: terminalRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  // Handle manual scroll detection
  const handleScroll = () => {
    if (terminalRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = terminalRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 5; // 5px tolerance
      setUserScrolled(!isAtBottom);
    }
  };

  // Typing animation function
  const startTypingAnimation = (command: string, onComplete: () => void) => {
    if (!command) {
      console.warn("Empty command passed to typing animation");
      onComplete();
      return;
    }
    
    setIsTyping(true);
    setTypingCommand("");
    
    let currentChar = 0;
    typingIntervalRef.current = setInterval(() => {
      try {
        if (currentChar <= command.length) {
          const currentText = command.slice(0, currentChar);
          setTypingCommand(currentText);
          currentChar++;
        } else {
          // Typing finished
          if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
          }
          setIsTyping(false);
          setTimeout(onComplete, 300);
        }
      } catch (error) {
        console.error("Error in typing animation:", error);
        setTypingCommand(command);
        setIsTyping(false);
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
        }
        setTimeout(onComplete, 300);
      }
    }, 25);
  };

  // Start live agent execution
  const startAgent = () => {
    if (!data?.commands || data.commands.length === 0 || isAgentRunning) return;
    
    setIsAgentRunning(true);
    setAgentCompleted(false);
    setDisplayedCommands([]);
    setTypingCommand("");
    setCurrentTypingOutput("");
    
    let commandIndex = 0;
    
    const executeNextCommand = () => {
      if (commandIndex >= data.commands.length) {
        setIsAgentRunning(false);
        setAgentCompleted(true);
        setTypingCommand(""); // Clear the last command when done
        return;
      }
      
      const command = data.commands[commandIndex];
      
      // Check if model has thinking capability
      const config = modelConfig?.[selectedModel];
      const shouldThink = config?.hasThinking;
      
      if (shouldThink) {
        // Show thinking phase
        setIsThinking(true);
        const thinkingDelay = getModelDelay(true);
        
        agentTimeoutRef.current = setTimeout(() => {
          setIsThinking(false);
          // Start typing animation after thinking
          startTypingAnimation(command.command, () => {
            // After typing is complete, add to displayed commands and clear typing state
            const newCommand = { ...command };
            setDisplayedCommands(prev => [...prev, newCommand]);
            setTypingCommand(""); // Clear typing command immediately to prevent duplicates
            
            commandIndex++;
            
            // Wait with realistic model-based delay before executing next command
            const nextDelay = getModelDelay();
            agentTimeoutRef.current = setTimeout(() => {
              executeNextCommand();
            }, nextDelay);
          });
        }, thinkingDelay);
      } else {
        // Direct typing animation without thinking phase
        startTypingAnimation(command.command, () => {
          // After typing is complete, add to displayed commands and clear typing state
          const newCommand = { ...command };
          setDisplayedCommands(prev => [...prev, newCommand]);
          setTypingCommand(""); // Clear typing command immediately to prevent duplicates
          
          commandIndex++;
          
          // Wait with realistic model-based delay before executing next command
          const nextDelay = getModelDelay();
          agentTimeoutRef.current = setTimeout(() => {
            executeNextCommand();
          }, nextDelay);
        });
      }
    };
    
    executeNextCommand();
  };

  // Stop agent execution
  const stopAgent = () => {
    setIsAgentRunning(false);
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
    }
    if (agentTimeoutRef.current) {
      clearTimeout(agentTimeoutRef.current);
    }
  };

  // Reset terminal
  const resetTerminal = () => {
    stopAgent();
    setDisplayedCommands([]);
    setTypingCommand("");
    setCurrentTypingOutput("");
    setAgentCompleted(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
      if (agentTimeoutRef.current) {
        clearTimeout(agentTimeoutRef.current);
      }
    };
  }, []);

  // Auto-scroll when content changes
  useEffect(() => {
    scrollToBottom();
  }, [displayedCommands, typingCommand]);

  if (!data) {
    return (
      <div className={`bg-black border border-white/10 rounded-lg p-6 font-mono ${className}`}>
        <div className="text-white/50 text-center">Loading terminal data...</div>
      </div>
    );
  }

  const maxCommands = data.commands?.length || 0;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Scenario Selector */}
      <ScenarioSelector
        selectedScenario={selectedScenario}
        selectedModel={selectedModel}
        onSelectionChange={handleSelectionChange}
      />
      
      <div className="bg-black border border-white/10 rounded-lg overflow-hidden flex flex-col h-[70vh]">
        {/* Terminal Header */}
        <div className="bg-white/5 border-b border-white/10 px-4 py-3 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-3 h-3 rounded-full bg-red-500/70"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/70"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/70"></div>
            </div>
            <span className="text-white/70 text-sm font-mono ml-4">Terminal</span>
          </div>
          <div className="text-white/50 text-xs font-mono">
            {/* {currentTypingOutput === -1 ? '0' : currentTypingOutput + 1} / {maxCommands} */}
          </div>
        </div>

      {/* System Info */}
      <div className="p-4 border-b border-white/5 flex-shrink-0">
        <div className="text-white/60 text-sm font-mono mb-2">
          <span className="text-white/40">System:</span> {data.system_prompt}
        </div>
        <div className="text-white/60 text-sm font-mono">
          <span className="text-white/40">Goal:</span> {data.goal}
        </div>
      </div>

      {/* Terminal Content */}
      <div 
        ref={terminalRef}
        onScroll={handleScroll}
        className="p-4 flex-1 min-h-0 overflow-y-auto bg-black scrollbar-hide"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        <div className="space-y-3">
          {/* Completed Commands */}
          {displayedCommands.map((cmd, index) => (
            <div key={cmd.id} className="space-y-1">
              {/* Command Input */}
              <div className="flex items-center space-x-2">
                <span className="text-green-400 font-mono text-sm">$</span>
                <span className="text-white font-mono text-sm">{cmd.command}</span>
              </div>
              
              {/* Command Output (show if response exists, even if empty) */}
              {cmd.response !== undefined && (
                <div className="ml-4 text-white/80 font-mono text-sm whitespace-pre-wrap">
                  {cmd.response || ''}
                </div>
              )}
              
              {/* Separator line */}
              <div className="h-px bg-white/10 my-3"></div>
            </div>
          ))}
          
          {/* Thinking Indicator */}
          {isThinking && (
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="text-blue-400 font-mono text-sm">🤔</span>
                <span className="text-blue-300 font-mono text-sm animate-pulse">
                  Thinking...
                </span>
              </div>
            </div>
          )}
          
          {/* Current Command with Typing Animation */}
          {(isTyping || typingCommand) && !isThinking && (
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="text-green-400 font-mono text-sm">$</span>
                <span className="text-white font-mono text-sm">
                  {typingCommand}
                  {isTyping && <span className="animate-pulse">|</span>}
                </span>
              </div>
            </div>
          )}
          
          {/* Cursor */}
          {!isTyping && !isThinking && (
            <div className="flex items-center space-x-2">
              <span className="text-green-400 font-mono text-sm">$</span>
              <span className="text-white/50 font-mono text-sm animate-pulse">_</span>
            </div>
          )}
        </div>
      </div>

      {/* History Progress Bar */}
      {maxCommands > 0 && (
        <div className="bg-white/5 border-t border-white/10 p-4 flex-shrink-0">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <span className="text-white/50 text-xs font-mono whitespace-nowrap">
                Run Agent
              </span>
              <button
                onClick={isAgentRunning ? stopAgent : startAgent}
                className={`px-3 py-1 text-xs font-mono rounded border transition-colors ${
                  isAgentRunning 
                    ? 'bg-red-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30' 
                    : 'bg-green-500/20 border-green-500/50 text-green-400 hover:bg-green-500/30'
                }`}
                disabled={maxCommands === 0}
              >
                {isAgentRunning ? '⏸ Stop' : '▶ Run'}
              </button>
            </div>
            <div className="flex-1 flex items-center space-x-3">
              <button
                onClick={resetTerminal}
                className="px-2 py-1 text-xs font-mono rounded border border-blue-500/50 bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
              >
                🔄 Reset
              </button>
            </div>
          </div>
        </div>
      )}

        <style jsx>{`
          /* Hide scrollbar */
          .scrollbar-hide::-webkit-scrollbar {
            display: none;
          }
        `}</style>
      </div>
      
      {/* Disclaimer */}
      <div className="absolute bottom-2 right-2">
        <span className="text-white/30 text-xs font-mono">*display only</span>
      </div>
    </div>
  );
}
