interface Citation {
  number: number;
  url: string;
  title: string;
}

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  return (
    <div className="border-t border-white/10 pt-6">
      <p className="mb-3 font-mono text-xs text-white/70">Sources:</p>
      <div className="space-y-1.5">
        {citations.map((citation) => (
          <p key={citation.number} className="text-xs font-mono text-white/50">
            [{citation.number}]{" "}
            <a
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white/70 transition-colors"
            >
              {citation.title}
            </a>
          </p>
        ))}
      </div>
      <p className="text-[10px] font-mono text-white/30 mt-6">
        © 2025 okay.events. All rights reserved.
      </p>
    </div>
  );
}
