"use client";

import { useState, useEffect } from "react";

export function useScrollBehavior() {
  const [hasScrolled, setHasScrolled] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setHasScrolled(entry.isIntersecting);
        });
      },
      {
        threshold: 0.5,
      },
    );

    const contentSection = document.getElementById("content");
    if (contentSection) {
      observer.observe(contentSection);
    }

    return () => {
      if (contentSection) {
        observer.unobserve(contentSection);
      }
    };
  }, []);

  return { hasScrolled };
}
