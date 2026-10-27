import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EvidenceBadge } from '../components/common/EvidenceBadge';
import { FeasibilityBadge } from '../components/common/FeasibilityBadge';

describe('Badges Component Suite', () => {
  it('renders EvidenceBadge with MEASURED provenance correctly', () => {
    render(<EvidenceBadge provenance="MEASURED" />);
    const badge = screen.getByText('MEASURED');
    expect(badge).toBeInTheDocument();
  });

  it('renders EvidenceBadge with PREDICTED provenance correctly', () => {
    render(<EvidenceBadge provenance="PREDICTED" />);
    const badge = screen.getByText('PREDICTED');
    expect(badge).toBeInTheDocument();
  });

  it('renders EvidenceBadge with THEORETICAL provenance correctly', () => {
    render(<EvidenceBadge provenance="THEORETICAL" />);
    const badge = screen.getByText('THEORETICAL');
    expect(badge).toBeInTheDocument();
  });

  it('renders FeasibilityBadge with FEASIBLE status', () => {
    render(<FeasibilityBadge status="FEASIBLE" />);
    const badge = screen.getByText('FEASIBLE');
    expect(badge).toBeInTheDocument();
  });

  it('renders FeasibilityBadge with PROVEN_INFEASIBLE status', () => {
    render(<FeasibilityBadge status="PROVEN_INFEASIBLE" />);
    const badge = screen.getByText('PROVEN INFEASIBLE');
    expect(badge).toBeInTheDocument();
  });

  it('renders FeasibilityBadge with UNKNOWN_FEASIBILITY status', () => {
    render(<FeasibilityBadge status="UNKNOWN_FEASIBILITY" />);
    const badge = screen.getByText('UNKNOWN FEASIBILITY');
    expect(badge).toBeInTheDocument();
  });
});
