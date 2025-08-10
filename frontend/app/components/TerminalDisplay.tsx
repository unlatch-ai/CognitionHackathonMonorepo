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

interface Scenario {
  id: string;
  name: string;
  description: string;
  file: string;
}

interface TerminalDisplayProps {
  className?: string;
}

export function TerminalDisplay({ className = "" }: TerminalDisplayProps) {
  const [currentCommandIndex, setCurrentCommandIndex] = useState(-1); // Start at -1 for initial state
  const [displayedCommands, setDisplayedCommands] = useState<Command[]>([]);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [userScrolled, setUserScrolled] = useState(false);
  const [typingCommand, setTypingCommand] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showOutput, setShowOutput] = useState(false);
  const [data, setData] = useState<TerminalData | null>(null);
  const [selectedScenario, setSelectedScenario] = useState("react-setup");
  const terminalRef = useRef<HTMLDivElement>(null);
  const autoPlayIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Available scenarios
  const scenarios: Scenario[] = [
    {
      id: "react-setup",
      name: "React TypeScript Setup",
      description: "Setting up a new React project with TypeScript and dependencies",
      file: "/terminal-scenarios/react-setup.json"
    },
    {
      id: "python-setup",
      name: "Python ML Project",
      description: "Creating a Python machine learning project with virtual environment",
      file: "/terminal-scenarios/python-setup.json"
    },
    {
      id: "docker-deployment",
      name: "Docker Deployment",
      description: "Containerizing and deploying a web application with Docker",
      file: "/terminal-scenarios/docker-deployment.json"
    },
    {
      id: "git-workflow",
      name: "Git Workflow",
      description: "Complete Git workflow with branching, commits, and merging",
      file: "/terminal-scenarios/git-workflow.json"
    }
  ];

  // Load scenario data
  useEffect(() => {
    const currentScenario = scenarios.find(s => s.id === selectedScenario);
    if (currentScenario) {
      fetch(currentScenario.file)
        .then((res) => res.json())
        .then((scenarioData) => {
          setData(scenarioData);
          setCurrentCommandIndex(-1); // Reset to initial state
        })
        .catch((err) => console.log("Scenario data not found:", err));
    }
  }, [selectedScenario]);

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
  const startTypingAnimation = (command: string) => {
    if (!command) {
      console.warn("Empty command passed to typing animation");
      return;
    }
    
    setIsTyping(true);
    setShowOutput(false);
    setTypingCommand("");
    
    let currentChar = 0;
    typingIntervalRef.current = setInterval(() => {
      try {
        if (currentChar <= command.length) {
          const currentText = command.slice(0, currentChar);
          setTypingCommand(currentText);
          currentChar++;
        } else {
          // Typing finished, show output after a brief delay
          if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
          }
          setTimeout(() => {
            setIsTyping(false);
            setShowOutput(true);
          }, 300);
        }
      } catch (error) {
        console.error("Error in typing animation:", error);
        // Fallback: complete the typing immediately
        setTypingCommand(command);
        setIsTyping(false);
        setShowOutput(true);
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
        }
      }
    }, 25); // 25ms per character for faster, smooth demo typing
  };

  useEffect(() => {
    // Clear any existing typing animation
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = null;
    }
    
    if (data?.commands && data.commands.length > 0) {
      // Show commands up to the current index, or empty array for initial state
      if (currentCommandIndex === -1) {
        setDisplayedCommands([]);
        setIsTyping(false);
        setShowOutput(false);
        setTypingCommand("");
      } else {
        // Show completed commands (all previous commands)
        const completedCommands = data.commands.slice(0, currentCommandIndex);
        setDisplayedCommands(completedCommands);
        
        // Start typing animation for current command
        const currentCommand = data.commands[currentCommandIndex];
        if (currentCommand && currentCommand.command) {
          startTypingAnimation(currentCommand.command);
        }
      }
    }
  }, [currentCommandIndex, data]);

  // Auto-scroll to bottom when commands change or output is shown
  useEffect(() => {
    scrollToBottom();
  }, [displayedCommands, userScrolled, showOutput, typingCommand]);

  // Auto-play functionality
  const startAutoPlay = () => {
    if (isAutoPlaying || !data?.commands) return;
    
    setIsAutoPlaying(true);
    setCurrentCommandIndex(-1); // Start from initial state
    
    autoPlayIntervalRef.current = setInterval(() => {
      setCurrentCommandIndex(prev => {
        if (prev >= (data.commands.length - 1)) {
          setIsAutoPlaying(false);
          if (autoPlayIntervalRef.current) {
            clearInterval(autoPlayIntervalRef.current);
          }
          return prev;
        }
        return prev + 1;
      });
    }, 1500); // 1.5 seconds between commands
  };

  const stopAutoPlay = () => {
    setIsAutoPlaying(false);
    if (autoPlayIntervalRef.current) {
      clearInterval(autoPlayIntervalRef.current);
      autoPlayIntervalRef.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (autoPlayIntervalRef.current) {
        clearInterval(autoPlayIntervalRef.current);
      }
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

  const goToPrevious = () => {
    if (currentCommandIndex > -1) {
      setCurrentCommandIndex(currentCommandIndex - 1);
      setUserScrolled(false);
      stopAutoPlay();
    }
  };

  const goToNext = () => {
    if (currentCommandIndex < (data?.commands.length || 0) - 1) {
      setCurrentCommandIndex(currentCommandIndex + 1);
      setUserScrolled(false);
      stopAutoPlay();
    }
  };

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
        scenarios={scenarios}
        selectedScenario={selectedScenario}
        onScenarioChange={setSelectedScenario}
      />
      
      <div className="bg-black border border-white/10 rounded-lg overflow-hidden">
        {/* Terminal Header */}
        <div className="bg-white/5 border-b border-white/10 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-3 h-3 rounded-full bg-red-500/70"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/70"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/70"></div>
            </div>
            <span className="text-white/70 text-sm font-mono ml-4">Terminal</span>
          </div>
          <div className="text-white/50 text-xs font-mono">
            {currentCommandIndex === -1 ? '0' : currentCommandIndex + 1} / {maxCommands}
          </div>
        </div>

      {/* System Info */}
      <div className="p-4 border-b border-white/5">
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
        className="p-4 h-[400px] overflow-y-auto bg-black scrollbar-hide"
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
          
          {/* Current Command with Typing Animation */}
          {currentCommandIndex >= 0 && data?.commands[currentCommandIndex] && (
            <div className="space-y-1">
              {/* Typing Command Input */}
              <div className="flex items-center space-x-2">
                <span className="text-green-400 font-mono text-sm">$</span>
                <span className="text-white font-mono text-sm">
                  {typingCommand}
                  {isTyping && <span className="animate-pulse">|</span>}
                </span>
              </div>
              
              {/* Command Output (shown after typing is complete) */}
              {showOutput && data.commands[currentCommandIndex].response !== undefined && (
                <div className="ml-4 text-white/80 font-mono text-sm whitespace-pre-wrap">
                  {data.commands[currentCommandIndex].response || ''}
                </div>
              )}
              
              {/* Separator line if output is shown */}
              {showOutput && <div className="h-px bg-white/10 my-3"></div>}
            </div>
          )}
          
          {/* Cursor */}
          <div className="flex items-center space-x-2">
            <span className="text-green-400 font-mono text-sm">$</span>
            <span className="text-white/50 font-mono text-sm animate-pulse">_</span>
          </div>
        </div>
      </div>

      {/* History Progress Bar */}
      {maxCommands > 0 && (
        <div className="bg-white/5 border-t border-white/10 p-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <span className="text-white/50 text-xs font-mono whitespace-nowrap">
                Command History
              </span>
              <button
                onClick={isAutoPlaying ? stopAutoPlay : startAutoPlay}
                className={`px-3 py-1 text-xs font-mono rounded border transition-colors ${
                  isAutoPlaying 
                    ? 'bg-red-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30' 
                    : 'bg-green-500/20 border-green-500/50 text-green-400 hover:bg-green-500/30'
                }`}
                disabled={maxCommands === 0}
              >
                {isAutoPlaying ? '⏸ Stop' : '▶ Auto Play'}
              </button>
            </div>
            <div className="flex-1 flex items-center space-x-3">
              <button
                onClick={goToPrevious}
                disabled={currentCommandIndex === -1 || isAutoPlaying}
                className="px-2 py-1 text-xs font-mono rounded border border-white/30 text-white/70 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                ← Prev
              </button>
              
              <div className="flex-1 relative">
                <div 
                  className="w-full h-2 bg-white/10 rounded-lg"
                  style={{
                    background: currentCommandIndex === -1 
                      ? 'rgba(255,255,255,0.1)' 
                      : `linear-gradient(to right, #22c55e 0%, #22c55e ${((currentCommandIndex + 1) / maxCommands) * 100}%, rgba(255,255,255,0.1) ${((currentCommandIndex + 1) / maxCommands) * 100}%, rgba(255,255,255,0.1) 100%)`
                  }}
                />
                <div className="flex justify-between text-xs text-white/40 font-mono mt-1">
                  <span>0</span>
                  <span>{maxCommands}</span>
                </div>
              </div>
              
              <button
                onClick={goToNext}
                disabled={currentCommandIndex === maxCommands - 1 || isAutoPlaying}
                className="px-2 py-1 text-xs font-mono rounded border border-white/30 text-white/70 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                Next →
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
    </div>
  );
}
