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

    const colorScale = d3.scaleOrdinal()
      .domain(params.map((p: any) => p.channel))
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("class", "axis-grid")
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format("$.2s" as any)))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "600")
      .text("WEEKLY SPEND ($)");

    g.append("g")
      .attr("class", "axis-grid")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".0%")))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -45)
      .attr("fill", "var(--text-dim)")
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "600")
      .text("EFFECTIVENESS (%)");

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
      .attr("class", "line")
      .attr("d", d => lineGenerator(d.points))
      .attr("fill", "none")
      .attr("stroke", d => colorScale(d.channel) as string)
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
      .attr("transform", `translate(${innerWidth + 30}, 0)`);

    curveData.forEach((d, i) => {
      const lg = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`)
        .style("cursor", "pointer")
        .on("mouseover", () => {
          g.selectAll(".line").attr("opacity", 0.1);
          g.selectAll(".line").filter((cd: any) => cd.channel === d.channel).attr("opacity", 1).attr("stroke-width", 4);
        })
        .on("mouseout", () => {
          g.selectAll(".line").attr("opacity", 0.6).attr("stroke-width", 2);
        });

      lg.append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("rx", 1)
        .attr("fill", colorScale(d.channel) as string);

      lg.append("text")
        .attr("x", 16)
        .attr("y", 9)
        .attr("fill", "var(--text-dim)")
        .style("font-size", "11px")
        .style("font-weight", "500")
        .style("text-transform", "capitalize")
        .text(d.channel.replace(/_/g, ' '));
    });

  }, [deliverables, selectedTerritory, loading]);

  return (
    <div className="analytics-card animate-in">
      <div className="card-title">Saturation Curves</div>
      <div style={{ width: '100%', overflow: 'hidden', padding: '1rem' }}>
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
