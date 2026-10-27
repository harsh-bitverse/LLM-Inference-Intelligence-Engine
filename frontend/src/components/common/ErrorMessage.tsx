import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { ErrorDetail } from '../../api/types';

interface ErrorMessageProps {
  title?: string;
  message: string;
  errorDetail?: ErrorDetail;
  onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  title = 'An error occurred',
  message,
  errorDetail,
  onRetry,
}) => {
  return (
    <div className="bg-red-950/40 border border-red-800/60 rounded-lg p-5 my-4 text-red-200">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="font-semibold text-red-200 text-sm">{title}</h4>
            {errorDetail?.error && (
              <span className="bg-red-900/60 text-red-300 text-xs px-2 py-0.5 rounded font-mono border border-red-700/50">
                {errorDetail.error}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-red-300/90 break-words">{message}</p>

          {errorDetail?.detail != null && (
            <div className="mt-3 bg-red-950/80 rounded p-2.5 text-xs font-mono text-red-300 border border-red-900 overflow-x-auto max-h-40">
              <pre>{typeof errorDetail.detail === 'string' ? errorDetail.detail : JSON.stringify(errorDetail.detail, null, 2)}</pre>
            </div>
          )}

          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 bg-red-900/40 hover:bg-red-800/50 text-red-200 text-xs font-medium rounded border border-red-700/50 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry Request
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
