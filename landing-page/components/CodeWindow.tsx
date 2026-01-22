import React from 'react';

interface CodeWindowProps {
  filename: string;
  code: string;
}

const CodeWindow: React.FC<CodeWindowProps> = ({ filename, code }) => {
  return (
    <div className="w-full my-8 border border-neutral-800 bg-[#0a0a0a] rounded-lg overflow-hidden shadow-2xl">
      <div className="flex items-center justify-between px-4 py-3 bg-[#111] border-b border-neutral-800">
        <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
            <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
        </div>
        <span className="text-xs font-mono text-neutral-500">{filename}</span>
      </div>
      <div className="p-6 overflow-x-auto">
        <pre className="text-neutral-300 font-mono text-xs leading-relaxed">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeWindow;