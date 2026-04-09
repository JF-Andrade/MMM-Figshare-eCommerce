import React from 'react';
import { useMMM } from '../context/MMMContext';

const Sidebar: React.FC = () => {
  const { runs, selectedRun, setSelectedRun, territories, selectedTerritory, setSelectedTerritory } = useMMM();

  return (
    <aside className="glass-card" style={{ width: '300px', height: 'calc(100vh - 4rem)', marginRight: '2rem', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginTop: 0 }}>Model Run</h3>
      <select 
        value={selectedRun?.run_id || ''} 
        onChange={(e) => {
          const run = runs.find(r => r.run_id === e.target.value);
          if (run) setSelectedRun(run);
        }}
        style={{ width: '100%', padding: '0.5rem', background: 'var(--glass-bg)', border: 'var(--glass-border)', borderRadius: '4px', color: 'white', marginBottom: '2rem' }}
      >
        {runs.map(run => (
          <option key={run.run_id} value={run.run_id}>
            {run.run_name} ({(run.metrics?.r2_test || 0).toFixed(2)})
          </option>
        ))}
      </select>

      <h3>Territory</h3>
      <select 
        value={selectedTerritory} 
        onChange={(e) => setSelectedTerritory(e.target.value)}
        style={{ width: '100%', padding: '0.5rem', background: 'var(--glass-bg)', border: 'var(--glass-border)', borderRadius: '4px', color: 'white', marginBottom: '2rem' }}
      >
        <option value="All">All Territories</option>
        {territories.map(terr => (
          <option key={terr} value={terr}>{terr}</option>
        ))}
      </select>

      <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Run Details</div>
        <div style={{ fontSize: '0.9rem' }}>R²: {(selectedRun?.metrics?.r2_test || 0).toFixed(4)}</div>
        <div style={{ fontSize: '0.9rem' }}>MAPE: {(selectedRun?.metrics?.mape_test || 0).toFixed(1)}%</div>
      </div>
    </aside>
  );
};

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div style={{ display: 'flex', padding: '2rem' }}>
      <Sidebar />
      <main style={{ flex: 1 }}>
        {children}
      </main>
    </div>
  );
};

export default Layout;
