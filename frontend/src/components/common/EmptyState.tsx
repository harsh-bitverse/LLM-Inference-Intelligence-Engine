import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  action,
  icon,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center border border-dashed border-slate-800 rounded-xl bg-slate-900/30">
      <div className="text-slate-500 mb-3">
        {icon || <Inbox className="w-10 h-10 stroke-[1.5]" />}
      </div>
      <h3 className="text-base font-medium text-slate-300">{title}</h3>
      {description && <p className="mt-1 text-sm text-slate-400 max-w-sm">{description}</p>}
      {action && (
        <button
          onClick={action.onClick}
          className="mt-4 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium rounded-lg transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
