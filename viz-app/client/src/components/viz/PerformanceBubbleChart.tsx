import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useMMM } from '../../context/MMMContext';

const PerformanceBubbleChart: React.FC = () => {
  const { deliverables, selectedTerritory, loading } = useMMM();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (loading || !deliverables || !svgRef.current) return;

    // Data preparation
    let data = [];
    if (selectedTerritory === 'All') {
      data = deliverables.roi || [];
    } else {
      data = (deliverables.regional || []).filter((d: any) => d.territory === selectedTerritory);
    }

    if (data.length === 0) return;

    // Dimensions
    const width = 800;
    const height = 500;
    const margin = { top: 40, right: 100, bottom: 60, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, (d: any) => d.total_spend || d.spend) * 1.1])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, (d: any) => d.roi) * 1.2])
      .range([innerHeight, 0]);

    const rScale = d3.scaleSqrt()
      .domain([0, d3.max(data, (d: any) => d.contribution)])
      .range([5, 40]);

    const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d3.format("$.2s" as any));
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1f" as any));

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 45)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text("Total Spend ($)");

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -50)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text("ROI (x)");

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickSize(-innerHeight).tickFormat(() => ""))
      .style("stroke-opacity", 0.1)
      .style("stroke-dasharray", "3,3");

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(() => ""))
      .style("stroke-opacity", 0.1)
      .style("stroke-dasharray", "3,3");

    // Bubbles
    const bubbles = g.selectAll(".bubble")
      .data(data)
      .enter()
      .append("g")
      .attr("class", "bubble");

    bubbles.append("circle")
      .attr("cx", (d: any) => xScale(d.total_spend || d.spend))
      .attr("cy", (d: any) => yScale(d.roi))
      .attr("r", 0) // Start at 0 for animation
      .attr("fill", (d: any) => colorScale(d.channel))
      .attr("fill-opacity", 0.6)
      .attr("stroke", (d: any) => colorScale(d.channel))
      .attr("stroke-width", 2)
      .transition()
      .duration(800)
      .delay((_d, i) => i * 100)
      .attr("r", (d: any) => rScale(d.contribution));

    // Labels
    bubbles.append("text")
      .attr("x", (d: any) => xScale(d.total_spend || d.spend))
      .attr("y", (d: any) => yScale(d.roi))
      .attr("dy", ".3em")
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .style("font-size", "10px")
      .style("pointer-events", "none")
      .text((d: any) => {
        const val = rScale(d.contribution);
        return val > 20 ? d.channel.split('_')[0] : "";
      });

    // Tooltips (simple title for now)
    bubbles.append("title")
      .text((d: any) => `${d.channel}\nROI: ${d.roi.toFixed(2)}x\nSpend: $${(d.total_spend || d.spend).toLocaleString()}\nContrib: $${d.contribution.toLocaleString()}`);

  }, [deliverables, selectedTerritory, loading]);

  return (
    <div className="glass-card animate-in" style={{ padding: '1rem', position: 'relative' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1.5rem' }}>Channel Performance (ROI Matrix)</h3>
      <div style={{ width: '100%', overflow: 'hidden' }}>
        <svg 
          ref={svgRef} 
          viewBox="0 0 800 500" 
          width="100%" 
          height="auto" 
          style={{ background: 'transparent' }}
        />
      </div>
      <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--primary)', opacity: 0.6 }}></div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Bubble size = Conversion Contribution</span>
        </div>
      </div>
    </div>
  );
};

export default PerformanceBubbleChart;
