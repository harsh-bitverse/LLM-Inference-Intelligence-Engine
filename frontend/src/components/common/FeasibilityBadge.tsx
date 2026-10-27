import React from 'react';
import { FeasibilityStatus } from '../../api/types';

interface FeasibilityBadgeProps {
  status: FeasibilityStatus | string;
}

export const FeasibilityBadge: React.FC<FeasibilityBadgeProps> = ({ status }) => {
  let colorStyles = '';
  let label = status;

  switch (status) {
    case 'FEASIBLE':
      colorStyles = 'bg-emerald-950/80 text-emerald-300 border border-emerald-600/60';
      label = 'FEASIBLE';
      break;
    case 'PROVEN_INFEASIBLE':
      colorStyles = 'bg-red-950/80 text-red-300 border border-red-600/60';
      label = 'PROVEN INFEASIBLE';
      break;
    case 'ESTIMATED_INFEASIBLE':
      colorStyles = 'bg-amber-950/80 text-amber-300 border border-amber-600/60';
      label = 'ESTIMATED INFEASIBLE';
      break;
    case 'UNKNOWN_FEASIBILITY':
    default:
      colorStyles = 'bg-zinc-800 text-zinc-300 border border-zinc-600/60';
      label = 'UNKNOWN FEASIBILITY';
      break;
  }

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${colorStyles}`}
      title={`Feasibility Status: ${status}`}
    >
      {label}
    </span>
  );
};
