import React from 'react';
import { useMMM } from '../context/MMMContext';

const Sidebar: React.FC = () => {
  const { runs, selectedRun, setSelectedRun, territories, selectedTerritory, setSelectedTerritory } = useMMM();

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getHealthColor = (score: number) => {
    if (score >= 90) return 'var(--success)';
    if (score >= 70) return 'var(--warning)';
    return 'var(--danger)';
  };

  const healthScore = parseFloat(selectedRun?.tags?.model_health_score || '0');

  return (
    <aside className="flex flex-col gap-4" style={{ width: '320px', minHeight: 'calc(100vh - 4rem)', marginRight: '2rem', paddingRight: '1rem', borderRight: '1px solid var(--border-color)' }}>
      <header>
        <h2 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>MMM Advanced</h2>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Intelligence Engine</div>
      </header>

      <section className="flex flex-col gap-2" style={{ marginTop: '2rem' }}>
        <label className="card-title">Analysis Run</label>
        <select 
          className="analytics-card"
          style={{ width: '100%', padding: '0.75rem', background: 'var(--bg-color)', color: 'var(--text-main)', cursor: 'pointer', fontSize: '0.85rem' }}
          value={selectedRun?.run_id || ''} 
          onChange={(e) => {
            const run = runs.find(r => r.run_id === e.target.value);
            if (run) setSelectedRun(run);
          }}
        >
          {runs.map(run => {
            const r2 = run.metrics?.r2_test || 0;
            const modelName = run.model_type === 'hierarchical' ? 'HB-MMM' : 'Baseline';
            return (
              <option key={run.run_id} value={run.run_id}>
                {modelName} | {formatDate(run.start_time)} | R²: {r2.toFixed(3)}
              </option>
            );
          })}
        </select>

        {selectedRun && (
          <div className="analytics-card flex justify-between items-center" style={{ marginTop: '0.5rem', padding: '0.75rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Health Score</span>
            <span style={{ 
              fontWeight: 700, 
              color: getHealthColor(healthScore),
              padding: '0.2rem 0.6rem',
              borderRadius: '2px',
              background: `${getHealthColor(healthScore)}22`,
              fontSize: '0.85rem'
            }}>
              {healthScore.toFixed(0)}%
            </span>
          </div>
        )}
      </section>

      <section className="flex flex-col gap-2" style={{ marginTop: '1rem' }}>
        <label className="card-title">Territory Filter</label>
        <select 
          className="analytics-card"
          style={{ width: '100%', padding: '0.75rem', background: 'var(--bg-color)', color: 'var(--text-main)', cursor: 'pointer', fontSize: '0.85rem' }}
          value={selectedTerritory} 
          onChange={(e) => setSelectedTerritory(e.target.value)}
        >
          <option value="All">All Territories (Aggregated)</option>
          {territories.map(terr => (
            <option key={terr} value={terr}>{terr}</option>
          ))}
        </select>
      </section>

      <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
        <div className="card-title">Run Summary</div>
        <div className="flex flex-col gap-2">
          <div className="flex justify-between" style={{ fontSize: '0.85rem' }}>
            <span style={{ color: var('--text-dim') }}>R-Squared</span>
            <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{(selectedRun?.metrics?.r2_test || 0).toFixed(4)}</span>
          </div>
          <div className="flex justify-between" style={{ fontSize: '0.85rem' }}>
            <span style={{ color: var('--text-dim') }}>MAPE (Holdout)</span>
            <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{(selectedRun?.metrics?.mape_test || 0).toFixed(2)}%</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="flex animate-in" style={{ padding: '2rem', height: '100vh', boxSizing: 'border-box' }}>
      <Sidebar />
      <main style={{ flex: 1, overflowY: 'auto' }}>
        {children}
      </main>
    </div>
  );
};

export default Layout;
