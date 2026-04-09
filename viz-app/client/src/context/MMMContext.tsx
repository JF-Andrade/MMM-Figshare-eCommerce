import React, { createContext, useContext, useState, useEffect } from 'react';

interface MMMContextType {
  runs: any[];
  selectedRun: any | null;
  setSelectedRun: (run: any) => void;
  selectedTerritory: string;
  setSelectedTerritory: (territory: string) => void;
  deliverables: any | null;
  loading: boolean;
  territories: string[];
}

const MMMContext = createContext<MMMContextType | undefined>(undefined);

export const MMMProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<any | null>(null);
  const [selectedTerritory, setSelectedTerritory] = useState<string>('All');
  const [deliverables, setDeliverables] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [territories, setTerritories] = useState<string[]>([]);

  // Fetch runs on mount
  useEffect(() => {
    fetch('/api/runs')
      .then(res => res.json())
      .then(data => {
        setRuns(data);
        if (data.length > 0) setSelectedRun(data[0]);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch runs:', err);
        setLoading(false);
      });
  }, []);

  // Fetch deliverables when selectedRun changes
  useEffect(() => {
    if (!selectedRun) return;

    setLoading(true);
    fetch(`/api/runs/${selectedRun.run_id}/data`)
      .then(res => res.json())
      .then(data => {
        setDeliverables(data);
        
        // Extract territories
        if (data.contributions_territory) {
          const terrs = Array.from(new Set(data.contributions_territory.map((d: any) => d.territory))) as string[];
          setTerritories(terrs.sort());
        } else {
          setTerritories([]);
        }
        
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch deliverables:', err);
        setLoading(false);
      });
  }, [selectedRun]);

  return (
    <MMMContext.Provider value={{
      runs,
      selectedRun,
      setSelectedRun,
      selectedTerritory,
      setSelectedTerritory,
      deliverables,
      loading,
      territories
    }}>
      {children}
    </MMMContext.Provider>
  );
};

export const useMMM = () => {
  const context = useContext(MMMContext);
  if (context === undefined) {
    throw new Error('useMMM must be used within a MMMProvider');
  }
  return context;
};
