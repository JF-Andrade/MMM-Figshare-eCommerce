import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface BudgetData {
  channel: string;
  current_spend: number;
  optimal_spend: number;
  change_pct: number;
}

interface Props {
  data: BudgetData[];
}

const BudgetShiftChart: React.FC<Props> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Max value for scaling
    const maxVal = d3.max(data, d => Math.max(d.current_spend, d.optimal_spend)) || 0;

    const y = d3.scaleLinear()
      .domain([0, maxVal * 1.1])
      .range([height, 0]);

    // Draw Axes
    g.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 0)
      .attr('y2', height)
      .attr('stroke', 'var(--border-color)')
      .attr('stroke-width', 2);

    g.append('line')
      .attr('x1', width)
      .attr('y1', 0)
      .attr('x2', width)
      .attr('y2', height)
      .attr('stroke', 'var(--border-color)')
      .attr('stroke-width', 2);

    // Labels for Axes
    g.append('text')
      .attr('x', 0)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-dim)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('CURRENT SPEND');

    g.append('text')
      .attr('x', width)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-dim)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('OPTIMIZED SPEND');

    // Draw Lines
    const lines = g.selectAll('.shift-line')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'shift-line');

    lines.append('line')
      .attr('x1', 0)
      .attr('y1', d => y(d.current_spend))
      .attr('x2', width)
      .attr('y2', d => y(d.optimal_spend))
      .attr('stroke', d => d.change_pct > 0 ? 'var(--success)' : (d.change_pct < 0 ? 'var(--danger)' : 'var(--text-muted)'))
      .attr('stroke-width', 2)
      .attr('opacity', 0.6)
      .style('cursor', 'pointer')
      .on('mouseover', function() {
        d3.select(this).attr('opacity', 1).attr('stroke-width', 4);
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.6).attr('stroke-width', 2);
      });

    // Channel Labels and Dots
    lines.append('circle')
      .attr('cx', 0)
      .attr('cy', d => y(d.current_spend))
      .attr('r', 4)
      .attr('fill', 'var(--text-main)');

    lines.append('circle')
      .attr('cx', width)
      .attr('cy', d => y(d.optimal_spend))
      .attr('r', 4)
      .attr('fill', 'var(--text-main)');

    lines.append('text')
      .attr('x', -10)
      .attr('y', d => y(d.current_spend))
      .attr('text-anchor', 'end')
      .attr('alignment-baseline', 'middle')
      .attr('fill', 'var(--text-main)')
      .style('font-size', '11px')
      .text(d => d.channel);

    lines.append('text')
      .attr('x', width + 10)
      .attr('y', d => y(d.optimal_spend))
      .attr('text-anchor', 'start')
      .attr('alignment-baseline', 'middle')
      .attr('fill', 'var(--text-main)')
      .style('font-size', '11px')
      .text(d => `${d.channel} (${d.change_pct > 0 ? '+' : ''}${d.change_pct.toFixed(1)}%)`);

  }, [data]);

  return (
    <div className="analytics-card" style={{ overflowX: 'auto' }}>
      <div className="card-title">Budget Reallocation (Slope)</div>
      <svg ref={svgRef} width="800" height="400" style={{ display: 'block', margin: '0 auto' }}></svg>
    </div>
  );
};

export default BudgetShiftChart;
