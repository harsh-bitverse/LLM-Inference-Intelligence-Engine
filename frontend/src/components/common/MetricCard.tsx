import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number | null | undefined;
  unit?: string;
  subtitle?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  alert?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  badge,
  icon,
  alert = false,
}) => {
  const isAvailable = value !== null && value !== undefined;

  return (
    <div
      className={`p-4 rounded-xl border bg-slate-900/60 backdrop-blur-sm transition-all ${
        alert
          ? 'border-amber-600/50 bg-amber-950/20'
          : 'border-slate-800 hover:border-slate-700'
      }`}
    >
      <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
        <span className="truncate">{title}</span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>

      <div className="mt-2 flex items-baseline gap-1.5 flex-wrap">
        <span
          className={`text-2xl font-bold tracking-tight font-mono ${
            isAvailable ? (alert ? 'text-amber-300' : 'text-slate-100') : 'text-slate-500 text-lg'
          }`}
        >
          {isAvailable ? value : 'N/A'}
        </span>
        {isAvailable && unit && (
          <span className="text-xs text-slate-400 font-normal">{unit}</span>
        )}
      </div>

      {(subtitle || badge) && (
        <div className="mt-2.5 flex items-center justify-between gap-2 text-xs">
          {subtitle && <span className="text-slate-400 truncate">{subtitle}</span>}
          {badge}
        </div>
      )}
    </div>
  );
};
