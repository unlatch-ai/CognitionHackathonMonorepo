"use client";

import { useState, useEffect } from "react";
import { ChevronDown } from "lucide-react";

interface Citation {
  number: number;
  url: string;
}

interface TextItem {
  text: string;
  citation: Citation | null;
  sourceName?: string;
}

interface AnimatedTextProps {
  texts: TextItem[];
  onComplete?: () => void;
}

export function AnimatedText({ texts, onComplete }: AnimatedTextProps) {
  const [displayedTexts, setDisplayedTexts] = useState<string[]>([]);
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [showContent, setShowContent] = useState(false);
  const [animationStarted, setAnimationStarted] = useState(false);

  const startAnimation = () => {
    setAnimationStarted(true);
    setCurrentTextIndex(0);
    setDisplayedTexts([]);
    setShowContent(false);
  };

  const handleComplete = () => {
    const nextSection = document.getElementById("content");
    if (nextSection) {
      nextSection.scrollIntoView({ behavior: "smooth" });
    }
    onComplete?.();
  };

  const resetAnimation = () => {
    setCurrentTextIndex(0);
    setDisplayedTexts([]);
    setShowContent(false);
    setAnimationStarted(false);
  };

  useEffect(() => {
    if (animationStarted && currentTextIndex < texts.length) {
      const currentItem = texts[currentTextIndex];
      let charIndex = 0;
      let lastTime = performance.now();
      const targetInterval = 50;

      const animate = (currentTime: number) => {
        const deltaTime = currentTime - lastTime;

        if (deltaTime >= targetInterval) {
          if (charIndex <= currentItem.text.length) {
            const newText = currentItem.text.slice(0, charIndex);
            setDisplayedTexts(prev => {
              const updated = [...prev];
              updated[currentTextIndex] = newText;
              return updated;
            });
            charIndex++;
            lastTime = currentTime;
          } else {
            setTimeout(() => {
              if (currentTextIndex < texts.length - 1) {
                setCurrentTextIndex((prev) => prev + 1);
              } else {
                setShowContent(true);
                onComplete?.();
              }
            }, 1500);
            return;
          }
        }

        requestAnimationFrame(animate);
      };

      const animationFrame = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationFrame);
    }
  }, [currentTextIndex, texts, onComplete, animationStarted]);

  return (
    <div className="relative flex h-screen flex-col items-center justify-between p-4">
      {showContent && (
        <button
          onClick={resetAnimation}
          type="button"
          className="absolute right-8 top-8 animate-fade-in cursor-pointer rounded-full p-2 opacity-0 transition-all duration-500 hover:bg-white/5"
          aria-label="Reset animation"
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="transform opacity-50 transition-transform duration-300 hover:opacity-70"
            aria-hidden="true"
          >
            <path
              d="M21 8C20.0908 6.35127 18.7747 4.97439 17.1743 4.03841C15.5739 3.10244 13.7618 2.64513 11.9285 2.71489C10.0951 2.78465 8.3218 3.37811 6.79657 4.43157C5.27135 5.48504 4.05321 6.95736 3.27879 8.67C2.50438 10.3826 2.20079 12.2733 2.39811 14.1455C2.59543 16.0176 3.28641 17.8017 4.39907 19.3077C5.51174 20.8137 7.00965 21.9843 8.73924 22.6878C10.4688 23.3913 12.3652 23.6014 14.2077 23.2963"
              strokeLinecap="round"
            />
            <path d="M21 2V8H15" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}

      {!animationStarted ? (
        <div className="flex flex-col items-center justify-center h-full">
          <button
            onClick={startAnimation}
            className="font-mono text-2xl md:text-3xl px-8 py-4 border border-white/20 rounded-lg bg-white/5 hover:bg-white/10 transition-all duration-300 hover:scale-105"
          >
            Unmap Rogue SLMs.
          </button>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="max-w-4xl font-mono text-lg duration-700 animate-fade-in md:text-2xl space-y-4">
            <ul className="space-y-4 text-left">
              {texts.map((item, index) => (
                <li key={index} className={`flex items-start space-x-3 ${index <= currentTextIndex ? 'animate-fade-in' : 'opacity-0'}`}>
                  <span className="text-white/70 mt-1">•</span>
                  <div className="flex-1">
                    <span>{displayedTexts[index] || ''}</span>
                    {item.citation && displayedTexts[index] === item.text && (
                      <a
                        href={item.citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex text-sm align-super ml-1 text-white/70 hover:text-white transition-colors"
                      >
                        [{item.citation.number}]
                      </a>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      
      <div className="w-full max-w-4xl px-4 pb-8">
        <div className="border-t border-white/10 pt-4">
          <p className="mb-2 font-mono text-xs text-white/70">Sources:</p>
          <div className="space-y-1">
            {texts.filter(item => item.citation).map((item) => (
              <p key={item.citation!.number} className="text-xs font-mono text-white/50">
                [{item.citation!.number}] {" "}
                <a
                  href={item.citation!.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white/70 transition-colors"
                >
                  {item.sourceName || item.text.substring(0, 60)}...
                </a>
              </p>
            ))}
          </div>
        </div>
      </div>

      {showContent && (
        <button
          onClick={handleComplete}
          type="button"
          className="group absolute bottom-8 animate-fade-in cursor-pointer rounded-full p-4 opacity-0 transition-all duration-500 hover:bg-white/5"
          aria-label="Scroll to content"
        >
          <div className="relative animate-slide-in-from-top">
            <ChevronDown
              className="transform transition-transform duration-300 group-hover:translate-y-1 opacity-95 group-hover:opacity-100"
              size={40}
              aria-hidden="true"
            />
          </div>
        </button>
      )}
    </div>
  );
}
