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

    const colorScale = d3.scaleOrdinal()
      .domain(data.map((d: any) => d.channel))
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d3.format("$.2s" as any));
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1f" as any));

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis-grid")
      .call(xAxis)
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 45)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "600")
      .text("TOTAL SPEND ($)");

    g.append("g")
      .attr("class", "axis-grid")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -50)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "600")
      .text("ROI (X)");

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickSize(-innerHeight).tickFormat(() => ""))
      .style("stroke", "var(--border-color)")
      .style("stroke-opacity", 0.1)
      .style("stroke-dasharray", "3,3");

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(() => ""))
      .style("stroke", "var(--border-color)")
      .style("stroke-opacity", 0.1)
      .style("stroke-dasharray", "3,3");

    // Bubbles
    const bubbles = g.selectAll(".bubble-group")
      .data(data, (d: any) => d.channel);

    const bubblesEnter = bubbles.enter()
      .append("g")
      .attr("class", "bubble-group");

    bubblesEnter.append("circle")
      .attr("class", "bubble")
      .attr("cx", (d: any) => xScale(d.total_spend || d.spend))
      .attr("cy", (d: any) => yScale(d.roi))
      .attr("r", 0)
      .attr("fill", (d: any) => colorScale(d.channel) as string)
      .attr("fill-opacity", 0.3)
      .attr("stroke", (d: any) => colorScale(d.channel) as string)
      .attr("stroke-width", 2)
      .transition()
      .duration(1000)
      .ease(d3.easeElasticOut)
      .attr("r", (d: any) => rScale(d.contribution));

    // Update existing bubbles
    bubbles.select(".bubble")
      .transition()
      .duration(750)
      .attr("cx", (d: any) => xScale(d.total_spend || d.spend))
      .attr("cy", (d: any) => yScale(d.roi))
      .attr("r", (d: any) => rScale(d.contribution));

    // Exit bubbles
    bubbles.exit()
      .transition()
      .duration(300)
      .attr("r", 0)
      .remove();

    // Labels
    bubblesEnter.append("text")
      .attr("class", "bubble-label")
      .attr("x", (d: any) => xScale(d.total_spend || d.spend))
      .attr("y", (d: any) => yScale(d.roi))
      .attr("dy", ".3em")
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .style("font-size", "10px")
      .style("font-weight", "600")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .text((d: any) => {
        const val = rScale(d.contribution);
        return val > 20 ? d.channel.split('_')[0] : "";
      })
      .transition()
      .duration(1000)
      .style("opacity", 1);

    bubbles.select(".bubble-label")
      .transition()
      .duration(750)
      .attr("x", (d: any) => xScale(d.total_spend || d.spend))
      .attr("y", (d: any) => yScale(d.roi));

    // Tooltips (simple title for now)
    bubblesEnter.append("title")
      .text((d: any) => `${d.channel}\nROI: ${d.roi.toFixed(2)}x\nSpend: $${(d.total_spend || d.spend).toLocaleString()}\nContrib: $${d.contribution.toLocaleString()}`);

  }, [deliverables, selectedTerritory, loading]);

  return (
    <div className="analytics-card animate-in">
      <div className="card-title">Channel Performance Matrix</div>
      <div style={{ width: '100%', overflow: 'hidden', padding: '1rem' }}>
        <svg 
          ref={svgRef} 
          viewBox="0 0 800 500" 
          width="100%" 
          height="auto" 
          style={{ background: 'transparent' }}
        />
      </div>
      <div style={{ padding: '0 1rem 1rem 1rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--text-dim)', opacity: 0.6 }}></div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Bubble Volume = Revenue Contribution</span>
        </div>
      </div>
    </div>
  );
};

export default PerformanceBubbleChart;
