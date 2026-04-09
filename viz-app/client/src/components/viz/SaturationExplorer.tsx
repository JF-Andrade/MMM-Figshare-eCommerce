import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useMMM } from '../../context/MMMContext';

const SaturationExplorer: React.FC = () => {
  const { deliverables, selectedTerritory, loading } = useMMM();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (loading || !deliverables || !svgRef.current) return;

    // Extract parameters
    let params = [];
    if (selectedTerritory === 'All') {
      params = deliverables.saturation || [];
    } else {
      params = (deliverables.saturation_territory || []).filter((p: any) => p.territory === selectedTerritory);
    }

    if (params.length === 0) return;

    // Dimensions
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 150, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Helper: Hill function
    const hill = (x: number, L: number, k: number) => Math.pow(x, k) / (Math.pow(x, k) + Math.pow(L, k));

    // Prepare curve data (100 points per channel)
    const curveData = params.map((p: any) => {
      const points = [];
      const maxSpend = p.max_spend || 1.0;
      for (let i = 0; i <= 100; i++) {
        const x = (i / 100) * maxSpend;
        const x_norm = i / 100; // Model is trained on [0, 1] normalized spend
        points.push({ x, y: hill(x_norm, p.L_mean, p.k_mean) });
      }
      return { channel: p.channel, points };
    });

    // Scales
    const globalMaxX = d3.max(params, (p: any) => p.max_spend) || 1.0;
    const xScale = d3.scaleLinear()
      .domain([0, globalMaxX])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);

    const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format("$.2s" as any)))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "var(--text-dim)")
      .text("Weekly Spend ($)");

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".0%")))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "var(--text-dim)")
      .text("Relative Effectiveness (%)");

    // Line generator
    const lineGenerator = d3.line<any>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX);

    // Draw lines
    const curves = g.selectAll(".curve-group")
      .data(curveData)
      .enter()
      .append("g")
      .attr("class", "curve-group");

    curves.append("path")
      .attr("d", d => lineGenerator(d.points))
      .attr("fill", "none")
      .attr("stroke", d => colorScale(d.channel))
      .attr("stroke-width", 2)
      .attr("opacity", 0.6)
      .on("mouseover", function() {
        d3.select(this).attr("stroke-width", 4).attr("opacity", 1);
      })
      .on("mouseout", function() {
        d3.select(this).attr("stroke-width", 2).attr("opacity", 0.6);
      });

    // Legend
    const legend = g.append("g")
      .attr("transform", `translate(${innerWidth + 20}, 0)`);

    curveData.forEach((d, i) => {
      const lg = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`)
        .style("cursor", "pointer");

      lg.append("rect")
        .attr("width", 12)
        .attr("height", 12)
        .attr("fill", colorScale(d.channel));

      lg.append("text")
        .attr("x", 20)
        .attr("y", 10)
        .attr("fill", "var(--text-dim)")
        .style("font-size", "12px")
        .text(d.channel.replace(/_/g, ' ').toLowerCase());
    });

  }, [deliverables, selectedTerritory, loading]);

  return (
    <div className="glass-card animate-in" style={{ padding: '1rem', marginTop: '2rem' }}>
      <h3 style={{ marginTop: 0, marginBottom: '1.5rem' }}>Saturation Curves (Diminishing Returns)</h3>
      <div style={{ width: '100%', overflow: 'hidden' }}>
        <svg 
          ref={svgRef} 
          viewBox="0 0 800 400" 
          width="100%" 
          height="auto" 
          style={{ background: 'transparent' }}
        />
      </div>
    </div>
  );
};

export default SaturationExplorer;
