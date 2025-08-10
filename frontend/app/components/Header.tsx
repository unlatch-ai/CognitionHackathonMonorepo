"use client";

interface HeaderProps {
  isVisible: boolean;
}

export function Header({ isVisible }: HeaderProps) {
  return (
    <header
      className={`fixed left-0 top-0 z-40 flex w-full items-center justify-between border-b border-white/10 px-6 py-4 backdrop-blur-md transition-opacity duration-500 ${
        isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
    >
      <div className="flex items-center">
        <span className="font-mono text-lg">Unmap.ai</span>
      </div>
      <nav className="flex items-center space-x-6 font-mono text-sm">
        <button
          className="text-white/70 transition-colors hover:text-white"
          onClick={() => {
            const teamElement = document.getElementById('team');
            if (teamElement) {
              teamElement.scrollIntoView({ 
                behavior: 'smooth',
                block: 'center'
              });
            }
          }}
        >
          Team
        </button>
      </nav>
    </header>
  );
}
