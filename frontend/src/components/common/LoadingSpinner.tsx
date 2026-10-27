import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading...',
  size = 'md',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  }[size];

  return (
    <div className="flex flex-col items-center justify-center p-8 gap-3 text-slate-400">
      <Loader2 className={`${sizeClasses} animate-spin text-sky-400`} />
      {message && <p className="text-sm font-medium">{message}</p>}
    </div>
  );
};
