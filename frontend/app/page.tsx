"use client";

import React from "react";
import { AnimatedText } from "./components/AnimatedText";
import { Header } from "./components/Header";
import { CitationList } from "./components/CitationList";
import { TerminalDisplay } from "./components/TerminalDisplay";
import { useScrollBehavior } from "./hooks/useScrollBehavior";

interface Citation {
  number: number;
  url: string;
}

interface TextItem {
  text: string;
  citation: Citation | null;
  sourceName?: string;
}

const texts: TextItem[] = [
  {
    text: "Small Language Models (SLMs) Can Now Easily Run on Commodity Hardware, and Easily Accessible",
    citation: null,
    sourceName: "Trust me bro",
  },
  {
    text: "SLMs Can Be Smarter Than Large Language Models with Fine Tuning/RL",
    citation: {
      number: 1,
      url: "https://arxiv.org/pdf/2505.23723",
    },
    sourceName: "arXiv: ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering",
  },
  {
    text: "SLMs are The Future of Agentic AI",
    citation: {
      number: 2,
      url: "https://arxiv.org/pdf/2506.02153",
    },
    sourceName: "arXiv: Small Language Models are the Future of Agentic AI",
  },
];

const citations = [
  {
    number: 1,
    url: "https://arxiv.org/pdf/2505.23723",
    title: "ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering"
  },
  {
    number: 2,
    url: "https://arxiv.org/pdf/2506.02153",
    title: "Small Language Models are the Future of Agentic AI"
  }
];

export default function HomePage() {
  const { hasScrolled } = useScrollBehavior();
  const [galleryImages, setGalleryImages] = React.useState<string[]>([]);
  const [recentEvents, setRecentEvents] = React.useState<any[]>([]);
  const [terminalData, setTerminalData] = React.useState<any>(null);

  React.useEffect(() => {
    fetch("/api/gallery")
      .then((res) => res.json())
      .then((data) => setGalleryImages(data));
      
    fetch("/api/recent-events")
      .then((res) => res.json())
      .then((data) => setRecentEvents(data));
      
    fetch("/sample-terminal-data.json")
      .then((res) => res.json())
      .then((data) => setTerminalData(data))
      .catch((err) => console.log("Terminal data not found:", err));
  }, []);

  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      <Header isVisible={hasScrolled} />

      <div className="snap-y snap-mandatory h-screen overflow-y-auto">
        <div className="snap-start h-screen w-full">
          <AnimatedText texts={texts} />
        </div>

        {/* Interview with Anthropic Safety Researcher Section */}
        <div className="snap-start h-screen w-full relative overflow-hidden bg-transparent">
          {/* Geometric Background - Desktop */}
          <div className="absolute inset-0 opacity-40 md:block hidden">
            <div className="absolute inset-0 animate-rotate-bg mix-blend-plus-lighter">
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
                role="presentation"
                className="stroke-white/30"
              >
                <title>Geometric Background Pattern</title>
                <desc>
                  A rotating pattern of concentric hexagons creating a subtle background effect
                </desc>
                <path d="M50 0L93.3013 25V75L50 100L6.69873 75V25L50 0Z" strokeWidth="1" />
                <path d="M50 20L79.2487 35V65L50 80L20.7513 65V35L50 20Z" strokeWidth="1" />
                <path d="M50 40L65.9808 50V70L50 80L34.0192 70V50L50 40Z" strokeWidth="1" />
              </svg>
            </div>
          </div>

          <div className="relative z-10 h-full flex flex-col justify-center items-center px-4 md:px-8">
            <div className="w-full max-w-4xl mx-auto text-center">
              <div className="grid md:grid-cols-2 gap-8 md:gap-12 items-center">
                <div className="order-2 md:order-1">
                  <h1 className="text-3xl md:text-4xl font-mono mb-12 tracking-tight">
                    Interview with Anthropic Safety Researcher.
                  </h1>

                  <ul className="text-base md:text-lg font-mono space-y-6 mb-12 list-none text-left">
                    {[
                      <>Interviewed <a href="https://x.com/aengus_lynch1" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">aengus lynch</a>, Red Team at Anthropic</>,
                      "Identified most potent attack vectors, influenced our project",
                      "Gained insights into real-world AI safety challenges and mitigation strategies",
                    ].map((item, index) => (
                      <li
                        key={index}
                        className="flex items-start space-x-3 relative overflow-hidden"
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-white/30 mt-2 flex-shrink-0" />
                        <span>{item}</span>
                        <div
                          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                          style={{
                            transform: "translateX(-100%)",
                            animation: "shimmer 2s infinite",
                          }}
                        />
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="order-1 md:order-2">
                  <div className="relative">
                    <img
                      src="/gallery/interview.png"
                      alt="Interview with Anthropic Safety Researcher"
                      className="w-full max-w-md mx-auto rounded-lg border border-white/20"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Agent Terminal Section */}
        <div className="snap-start h-screen w-full relative overflow-hidden bg-black">
          <div className="h-full flex flex-col px-4 md:px-8 py-8">
            <div className="w-full max-w-6xl mx-auto flex flex-col h-full">
              <div className="text-center mb-6 flex-shrink-0">
                <h2 className="text-2xl md:text-4xl font-mono mb-3 tracking-tight">
                  Agent Terminal
                </h2>
                <p className="text-base md:text-lg font-mono text-white/70 max-w-2xl mx-auto">
                  Agent Malicious Tests
                </p>
              </div>
              <div className="flex-1 min-h-0">
                <TerminalDisplay className="w-full h-full" />
              </div>
            </div>
          </div>
        </div>

        <div
          id="content"
          className="snap-start h-screen w-full relative overflow-hidden bg-transparent"
        >
          {/* Geometric Background - Desktop */}
          <div className="absolute inset-0 opacity-40 md:block hidden">
            <div className="absolute inset-0 animate-rotate-bg mix-blend-plus-lighter">
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
                role="presentation"
                className="stroke-white/30"
              >
                <title>Geometric Background Pattern</title>
                <desc>
                  A rotating pattern of concentric hexagons creating a subtle background effect
                </desc>
                <path d="M50 0L93.3013 25V75L50 100L6.69873 75V25L50 0Z" strokeWidth="1" />
                <path d="M50 20L79.2487 35V65L50 80L20.7513 65V35L50 20Z" strokeWidth="1" />
                <path d="M50 40L65.1962 45V55L50 60L34.8038 55V45L50 40Z" strokeWidth="1" />
              </svg>
            </div>
          </div>
          {/* Simplified Background - Mobile */}
          <div className="absolute inset-0 opacity-40 block md:hidden">
            <div className="absolute inset-0 animate-rotate-bg mix-blend-plus-lighter">
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
                role="presentation"
                className="stroke-white/30"
              >
                <path d="M50 0L93.3013 25V75L50 100L6.69873 75V25L50 0Z" strokeWidth="1" />
                <path d="M50 20L79.2487 35V65L50 80L20.7513 65V35L50 20Z" strokeWidth="1" />
                <path d="M50 40L65.1962 45V55L50 60L34.8038 55V45L50 40Z" strokeWidth="1" />
              </svg>
            </div>
          </div>
          <div className="absolute inset-0 overflow-y-auto">
            <div className="min-h-full flex flex-col items-center justify-center p-4">
              <div
                className={`max-w-2xl w-full transition-opacity duration-500 pt-16 md:pt-20 ${hasScrolled ? "opacity-100" : "opacity-0"}`}
              >
                <div className="grid md:grid-cols-2 gap-12 mb-16">
                  <div>
                    <h2 className="text-2xl md:text-3xl font-mono mb-8 tracking-tight">
                      Upcoming Events
                    </h2>
                    <div className="border border-white/10 rounded-lg p-6 bg-white/5">
                      <iframe
                        src="https://lu.ma/embed/calendar/cal-R6hvwORXDHgOA9q/events"
                        width="100%"
                        height="450"
                        frameBorder="0"
                        style={{
                          border: "1px solid rgba(255, 255, 255, 0.1)",
                          borderRadius: "4px",
                          backgroundColor: "transparent"
                        }}
                        allowFullScreen
                        aria-hidden="false"
                        tabIndex={0}
                        className="w-full"
                      />
                    </div>
                  </div>

                  <div>
                    <h2 className="text-2xl md:text-3xl font-mono mb-8 tracking-tight">
                      Recent Events
                    </h2>
                    <div className="space-y-6">
                      {recentEvents.length > 0 ? (
                        recentEvents.map((event) => (
                          <div key={event.id} className="border border-white/10 rounded-lg p-6 bg-white/5">
                            <div className="flex justify-between items-start mb-3">
                              <h3 className="font-mono text-lg">{event.title}</h3>
                              <span className="font-mono text-sm text-white/70">{event.date}</span>
                            </div>
                            <p className="font-mono text-sm text-white/70 mb-3">
                              {event.description}
                            </p>
                            <div className="mb-3">
                              {event.photos && event.photos.length > 0 ? (
                                <div className="aspect-[4/3] bg-white/5 rounded border border-white/10 flex items-center justify-center overflow-hidden">
                                  {/* eslint-disable-next-line @next/next/no-img-element */}
                                  <img src={event.photos[0]} alt={`${event.title} photo`} className="object-cover w-full h-full" />
                                </div>
                              ) : (
                                <div className="aspect-[4/3] bg-white/5 rounded border border-white/10 flex items-center justify-center">
                                  <span className="font-mono text-xs text-white/40">Photo</span>
                                </div>
                              )}
                            </div>
                            <p className="font-mono text-xs text-white/50">
                              {event.attendees} • {event.outcome}
                            </p>
                          </div>
                        ))
                      ) : (
                        <div className="text-center text-white/50 font-mono">Loading events...</div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="mb-16 h-px w-full bg-white/10" />

                <h1 className="text-3xl md:text-4xl font-mono mb-12 tracking-tight">
                  Building real connections in SF.
                </h1>

                <ul className="text-base md:text-lg font-mono space-y-6 mb-12 list-none">
                  {[
                    "Casual meetups.",
                    "Lowkey poker nights.",
                    "Weekly tech events for builders and creators.",
                    "Meet real people, not just LinkedIn profiles.",
                  ].map((item, index) => (
                    <li
                      key={item}
                      className="flex items-center space-x-3 relative overflow-hidden"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-white/30" />
                      <span>{item}</span>
                      <div
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-sheen mix-blend-plus-lighter"
                        style={{ animationDelay: `${(index + 1) * 200}ms` }}
                      />
                    </li>
                  ))}
                </ul>

                <p className="text-lg md:text-xl font-mono mb-10 tracking-tight">
                  Join us at Unmap.ai for advanced AI agent security testing and malicious behavior detection.
                </p>

                <div className="mb-16 h-px w-full bg-white/10" />


                <div className="mb-16">
                  <h2 className="text-2xl md:text-3xl font-mono mb-8 tracking-tight">
                    Event Gallery
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                    {galleryImages.length > 0 ? (
                      galleryImages.map((src, idx) => (
                        <div key={src} className="aspect-square bg-white/5 rounded-lg border border-white/10 flex items-center justify-center overflow-hidden">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={src} alt={`Event photo ${idx + 1}`} className="object-cover w-full h-full" />
                        </div>
                      ))
                    ) : (
                      <div className="col-span-2 md:col-span-3 text-center text-white/50 font-mono">No photos found in gallery.</div>
                    )}
                  </div>
                  <p className="font-mono text-sm text-white/50">
                    Moments from Unmap.ai.
                  </p>
                </div>

                <div className="mb-16 h-px w-full bg-white/10" />

                <div className="space-y-4">
                  <p className="text-base md:text-lg font-mono">
                    RSVP: Contact Olsen or Kevin, you know how to reach us.
                  </p>
                  <p className="font-mono text-sm text-white/50">
                    Events are invite-only.
                  </p>
                </div>

                <div className="mb-6 h-px w-full bg-white/10" />

                <CitationList citations={citations} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
