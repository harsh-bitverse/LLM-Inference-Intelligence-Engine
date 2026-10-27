import React from 'react';
import { ExecutionMode, ProvenanceType } from '../../api/types';

interface EvidenceBadgeProps {
  provenance?: ProvenanceType | string;
  executionMode?: ExecutionMode | string;
  isSynthetic?: boolean;
}

export const EvidenceBadge: React.FC<EvidenceBadgeProps> = ({
  provenance,
  executionMode,
  isSynthetic,
}) => {
  return (
    <div className="inline-flex items-center gap-1.5 flex-wrap">
      {provenance && (
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium tracking-wide uppercase ${
            provenance === 'MEASURED'
              ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-700/50'
              : provenance === 'PREDICTED'
              ? 'bg-blue-950/80 text-blue-300 border border-blue-700/50'
              : provenance === 'THEORETICAL'
              ? 'bg-purple-950/80 text-purple-300 border border-purple-700/50'
              : 'bg-amber-950/80 text-amber-300 border border-amber-700/50'
          }`}
          title={`Data Provenance: ${provenance}`}
        >
          {provenance}
        </span>
      )}

      {executionMode && (
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium tracking-wide uppercase ${
            executionMode === 'REAL'
              ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-700/50'
              : 'bg-orange-950/80 text-orange-300 border border-orange-700/50'
          }`}
          title={`Execution Mode: ${executionMode}`}
        >
          {executionMode}
        </span>
      )}

      {isSynthetic && (
        <span
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium tracking-wide uppercase bg-rose-950/80 text-rose-300 border border-rose-700/50"
          title="Synthetic Dataset"
        >
          SYNTHETIC
        </span>
      )}
    </div>
  );
};
