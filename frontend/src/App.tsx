import React, { useState } from 'react';
import { Navbar, NavTab } from './components/layout/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { HardwarePage } from './pages/HardwarePage';
import { BenchmarksPage } from './pages/BenchmarksPage';
import { DiagnosticsPage } from './pages/DiagnosticsPage';
import { PredictionsPage } from './pages/PredictionsPage';
import { OptimizerPage } from './pages/OptimizerPage';
import { RecommendationsPage } from './pages/RecommendationsPage';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar currentTab={currentTab} onSelectTab={setCurrentTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {currentTab === 'overview' && <OverviewPage onNavigate={setCurrentTab} />}
        {currentTab === 'hardware' && <HardwarePage />}
        {currentTab === 'benchmarks' && <BenchmarksPage />}
        {currentTab === 'diagnostics' && <DiagnosticsPage />}
        {currentTab === 'predictions' && <PredictionsPage />}
        {currentTab === 'optimizer' && <OptimizerPage />}
        {currentTab === 'recommendations' && <RecommendationsPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <span>LLM Inference Intelligence Engine (IIE)</span> •{' '}
            <span>Measurement-driven analytical & empirical intelligence</span>
          </div>
          <div className="flex items-center gap-4 text-slate-600">
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>SQLite WAL Catalog</span>
            <span>•</span>
            <span>React + TypeScript</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
