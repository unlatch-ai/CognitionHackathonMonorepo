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
}

interface AnimatedTextProps {
  texts: TextItem[];
  onComplete?: () => void;
}

export function AnimatedText({ texts, onComplete }: AnimatedTextProps) {
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [displayText, setDisplayText] = useState("");
  const [showContent, setShowContent] = useState(false);

  const handleComplete = () => {
    const nextSection = document.getElementById("content");
    if (nextSection) {
      nextSection.scrollIntoView({ behavior: "smooth" });
    }
    onComplete?.();
  };

  const resetAnimation = () => {
    setCurrentTextIndex(0);
    setDisplayText("");
    setShowContent(false);
  };

  useEffect(() => {
    if (currentTextIndex < texts.length) {
      const currentItem = texts[currentTextIndex];
      let charIndex = 0;
      let lastTime = performance.now();
      const targetInterval = 50;

      const animate = (currentTime: number) => {
        const deltaTime = currentTime - lastTime;

        if (deltaTime >= targetInterval) {
          if (charIndex <= currentItem.text.length) {
            setDisplayText(currentItem.text.slice(0, charIndex));
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
  }, [currentTextIndex, texts, onComplete]);

  return (
    <div className="relative flex h-screen flex-col items-center justify-center p-4">
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

      <div className="max-w-3xl text-center font-mono text-lg duration-700 animate-fade-in md:text-2xl">
        {displayText}
        {currentTextIndex < texts.length &&
          texts[currentTextIndex].citation &&
          displayText === texts[currentTextIndex].text && (
            <a
              href={texts[currentTextIndex].citation?.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex text-sm align-super ml-0.5 text-white/70 hover:text-white transition-colors"
            >
              [{texts[currentTextIndex].citation?.number}]
            </a>
          )}
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
