import React from 'react'
import { MMMProvider } from './context/MMMContext'
import Layout from './context/Layout'
import PerformanceBubbleChart from './components/viz/PerformanceBubbleChart'
import SaturationExplorer from './components/viz/SaturationExplorer'
import ChoroplethMap from './components/viz/ChoroplethMap'

function Dashboard() {
  return (
    <Layout>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem' }}>
        <ChoroplethMap />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <PerformanceBubbleChart />
          <SaturationExplorer />
        </div>
      </div>
    </Layout>
  )
}

function App() {
  return (
    <MMMProvider>
      <Dashboard />
    </MMMProvider>
  )
}

export default App
