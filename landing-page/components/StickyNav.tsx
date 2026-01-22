import React, { useState, useEffect } from 'react';
import { Github } from 'lucide-react';

const StickyNav: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 border-b ${scrolled ? 'bg-[#050505]/90 backdrop-blur-xl border-white/10 py-4' : 'bg-transparent border-transparent py-6'}`}>
      <div className="max-w-7xl mx-auto px-6 flex justify-between items-center">
        <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-borr-red rounded flex items-center justify-center text-white font-serif font-bold text-lg">B</div>
            <div className="text-white font-sans font-medium text-lg tracking-tight">
            Borr<span className="text-neutral-500">AI</span>
            </div>
        </div>
        
        <div className="flex gap-6">
            <a href="https://github.com/phamtrung0633/retail-ai" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 bg-white text-black px-4 py-2 rounded text-sm font-medium hover:bg-neutral-200 transition-colors">
              <Github size={16} />
              Star on GitHub
            </a>
        </div>
      </div>
    </nav>
  );
};

export default StickyNav;