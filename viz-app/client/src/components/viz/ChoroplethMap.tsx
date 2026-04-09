import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useMMM } from '../../context/MMMContext';

const ChoroplethMap: React.FC = () => {
  const { deliverables, setSelectedTerritory, loading } = useMMM();
  const svgRef = useRef<SVGSVGElement>(null);
  const [geoData, setGeoData] = useState<any>(null);

  useEffect(() => {
    // Load World GeoJSON
    fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
      .then(res => res.json())
      .then(data => setGeoData(data))
      .catch(err => console.error('Failed to load GeoJSON:', err));
  }, []);

  useEffect(() => {
    if (loading || !deliverables || !svgRef.current || !geoData) return;

    // Aggregate ROI per territory
    const regional = deliverables.regional || [];
    const territoryROI = d3.rollups(
      regional,
      (v: any) => d3.mean(v, (d: any) => d.roi),
      (d: any) => d.territory
    );

    const roiMap = new Map(territoryROI as any);

    // Filter GeoJSON to relevant countries (subset for performance and clarity)
    const countries = regional.map((d: any) => d.territory);
    
    // Dimensions
    const width = 800;
    const height = 400;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const projection = d3.geoMercator()
      .scale(120)
      .translate([width / 2, height / 1.5]);

    const path = d3.geoPath().projection(projection);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, d3.max(territoryROI, (d: any) => d[1]) || 10]);

    const g = svg.append("g");

    // Draw countries
    g.selectAll("path")
      .data(geoData.features)
      .enter()
      .append("path")
      .attr("d", path as any)
      .attr("fill", (d: any) => {
        const iso = d.id; // Map uses ISO3 or Name
        const name = d.properties.name;
        // Search by name or mapping (simple match for demo)
        const roi = roiMap.get(name) || roiMap.get(iso);
        return roi ? colorScale(roi as any) : "#1e293b";
      })
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 0.5)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d: any) {
        const name = d.properties.name;
        const roi = roiMap.get(name);
        if (roi) {
          d3.select(this).attr("stroke", "white").attr("stroke-width", 1.5);
        }
      })
      .on("mouseout", function() {
        d3.select(this).attr("stroke", "#0f172a").attr("stroke-width", 0.5);
      })
      .on("click", (_event, d: any) => {
        const name = d.properties.name;
        if (roiMap.has(name)) {
          setSelectedTerritory(name);
        }
      });

    // Legend
    const legendWidth = 200;
    const legendHeight = 10;
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 250}, ${height - 30})`);

    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale).ticks(3).tickSize(13);

    const defs = svg.append("defs");
    const linearGradient = defs.append("linearGradient")
      .attr("id", "map-gradient");

    linearGradient.selectAll("stop")
      .data(colorScale.ticks().map((t, i, nodes) => ({ offset: `${100*i/nodes.length}%`, color: colorScale(t) })))
      .enter().append("stop")
      .attr("offset", d => d.offset)
      .attr("stop-color", d => d.color);

    legend.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#map-gradient)");

    legend.append("g")
      .call(legendAxis)
      .select(".domain").remove();

    legend.append("text")
      .attr("y", -10)
      .attr("fill", "var(--text-dim)")
      .style("font-size", "10px")
      .text("Avg Regional ROI (x)");

  }, [deliverables, loading, geoData]);

  return (
    <div className="glass-card animate-in" style={{ padding: '1rem', marginTop: '2rem' }}>
      <h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>Territory Performance (Average ROI)</h3>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '1rem' }}>Click a highlighted country to filter the dashboard.</p>
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

export default ChoroplethMap;
