import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { Navbar } from '../components/layout/Navbar';
import { apiClient } from '../api/client';

describe('Navbar Component', () => {
  beforeEach(() => {
    vi.spyOn(apiClient, 'checkHealth').mockResolvedValue({
      status: 'ok',
      database: 'connected',
      version: '1.0.0',
    });
  });

  it('renders brand and all 7 navigation tabs', async () => {
    const handleSelect = vi.fn();
    await act(async () => {
      render(<Navbar currentTab="overview" onSelectTab={handleSelect} />);
    });

    expect(screen.getByText('LLM Inference Intelligence Engine')).toBeInTheDocument();
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('Hardware')).toBeInTheDocument();
    expect(screen.getByText('Benchmarks')).toBeInTheDocument();
    expect(screen.getByText('Diagnostics')).toBeInTheDocument();
    expect(screen.getByText('Predictions')).toBeInTheDocument();
    expect(screen.getByText('Optimization')).toBeInTheDocument();
    expect(screen.getByText('Recommendations')).toBeInTheDocument();
  });

  it('triggers onSelectTab callback when clicking navigation items', async () => {
    const handleSelect = vi.fn();
    await act(async () => {
      render(<Navbar currentTab="overview" onSelectTab={handleSelect} />);
    });

    const benchmarksTab = screen.getByRole('button', { name: /Benchmarks/i });
    fireEvent.click(benchmarksTab);
    expect(handleSelect).toHaveBeenCalledWith('benchmarks');

    const optimizerTab = screen.getByRole('button', { name: /Optimization/i });
    fireEvent.click(optimizerTab);
    expect(handleSelect).toHaveBeenCalledWith('optimizer');
  });

  it('displays API CONNECTED badge when health endpoint returns ok', async () => {
    await act(async () => {
      render(<Navbar currentTab="overview" onSelectTab={vi.fn()} />);
    });

    const badge = await screen.findByText('API CONNECTED');
    expect(badge).toBeInTheDocument();
  });
});
