import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { MetricCard } from '../components/common/MetricCard';
import { EmptyState } from '../components/common/EmptyState';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EvidenceBadge } from '../components/common/EvidenceBadge';

describe('Common UI Components', () => {
  it('renders ErrorMessage with retry button and triggers callback', () => {
    const handleRetry = vi.fn();
    render(
      <ErrorMessage
        title="Benchmark Failure"
        message="Process exited with non-zero code"
        onRetry={handleRetry}
      />
    );

    expect(screen.getByText('Benchmark Failure')).toBeInTheDocument();
    expect(screen.getByText('Process exited with non-zero code')).toBeInTheDocument();

    const retryBtn = screen.getByRole('button', { name: /Retry Request/i });
    expect(retryBtn).toBeInTheDocument();
    fireEvent.click(retryBtn);
    expect(handleRetry).toHaveBeenCalledTimes(1);
  });

  it('renders MetricCard with title, value, unit, subtitle, and badge', () => {
    render(
      <MetricCard
        title="Time to First Token"
        value={18.42}
        unit="ms"
        subtitle="Monotonic P95 TTFT"
        badge={<EvidenceBadge provenance="MEASURED" />}
      />
    );

    expect(screen.getByText('Time to First Token')).toBeInTheDocument();
    expect(screen.getByText('18.42')).toBeInTheDocument();
    expect(screen.getByText('ms')).toBeInTheDocument();
    expect(screen.getByText('Monotonic P95 TTFT')).toBeInTheDocument();
    expect(screen.getByText('MEASURED')).toBeInTheDocument();
  });

  it('renders EmptyState with icon and prompt message', () => {
    render(
      <EmptyState
        title="No Benchmarks Executed"
        description="Run a new inference benchmark suite to inspect live token latencies."
      />
    );

    expect(screen.getByText('No Benchmarks Executed')).toBeInTheDocument();
    expect(
      screen.getByText('Run a new inference benchmark suite to inspect live token latencies.')
    ).toBeInTheDocument();
  });

  it('renders LoadingSpinner with custom message', () => {
    render(<LoadingSpinner message="Evaluating Roofline intensities..." />);
    expect(screen.getByText('Evaluating Roofline intensities...')).toBeInTheDocument();
  });
});
