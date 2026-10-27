import React, { useEffect, useState } from 'react';
import {
  Activity,
  Cpu,
  Gauge,
  Stethoscope,
  TrendingUp,
  Sparkles,
  Compass,
  Server,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { HealthResponse } from '../../api/types';

export type NavTab =
  | 'overview'
  | 'hardware'
  | 'benchmarks'
  | 'diagnostics'
  | 'predictions'
  | 'optimizer'
  | 'recommendations';

interface NavbarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking');

  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        const res = await apiClient.checkHealth();
        if (isMounted) {
          setHealth(res);
          setHealthStatus(res.status === 'ok' ? 'healthy' : 'unhealthy');
        }
      } catch {
        if (isMounted) {
          setHealth(null);
          setHealthStatus('unhealthy');
        }
      }
    };

    check();
    const interval = setInterval(check, 15000); // 15s poll
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Activity className="w-4 h-4" /> },
    { id: 'hardware', label: 'Hardware', icon: <Cpu className="w-4 h-4" /> },
    { id: 'benchmarks', label: 'Benchmarks', icon: <Gauge className="w-4 h-4" /> },
    { id: 'diagnostics', label: 'Diagnostics', icon: <Stethoscope className="w-4 h-4" /> },
    { id: 'predictions', label: 'Predictions', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'optimizer', label: 'Optimization', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'recommendations', label: 'Recommendations', icon: <Compass className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Product Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-sky-600/20 border border-sky-500/30 flex items-center justify-center text-sky-400">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-slate-100 tracking-tight text-sm sm:text-base">
                  LLM Inference Intelligence Engine
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  v0.9.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Measurement-driven benchmarking, diagnostics & Pareto configuration sizer
              </p>
            </div>
          </div>

          {/* Backend Readiness Indicator */}
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                healthStatus === 'healthy'
                  ? 'bg-emerald-950/60 border-emerald-700/50 text-emerald-300'
                  : healthStatus === 'unhealthy'
                  ? 'bg-red-950/60 border-red-700/50 text-red-300'
                  : 'bg-slate-900 border-slate-700 text-slate-400'
              }`}
              title={
                healthStatus === 'healthy'
                  ? `Backend healthy (${health?.database || 'database connected'})`
                  : 'Backend unreachable or unhealthy'
              }
            >
              {healthStatus === 'healthy' ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              ) : healthStatus === 'unhealthy' ? (
                <AlertCircle className="w-3.5 h-3.5 text-red-400" />
              ) : (
                <HelpCircle className="w-3.5 h-3.5 text-slate-400 animate-pulse" />
              )}
              <span className="hidden md:inline font-mono">
                {healthStatus === 'healthy' ? 'API CONNECTED' : healthStatus === 'unhealthy' ? 'API DISCONNECTED' : 'CHECKING'}
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 sm:space-x-4 border-t border-slate-900 overflow-x-auto py-2 no-scrollbar">
          {navItems.map((item) => {
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-sky-500/10 text-sky-400 border border-sky-500/30 shadow-sm shadow-sky-950'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
