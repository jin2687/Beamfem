import { useState, useCallback } from 'react';
import { Play, RotateCcw, Layers } from 'lucide-react';
import SimulationCanvas from './components/SimulationCanvas';
import {
  solveFEM,
  buildCubeTruss,
  buildBridgeTruss,
  buildGridTower,
  type FEMResult,
  type TrussModel,
} from './logic/femSolver';

type PresetName = 'cube' | 'bridge' | 'tower';

const PRESETS: Record<PresetName, () => TrussModel> = {
  cube: () => buildCubeTruss(1.0),
  bridge: () => buildBridgeTruss(),
  tower: () => buildGridTower(),
};

const PRESET_LABELS: Record<PresetName, string> = {
  cube: 'Cube Truss',
  bridge: 'Bridge',
  tower: 'Grid Tower',
};

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
  display,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
  display?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-slate-300">
        <span>{label}</span>
        <span className="font-mono text-blue-300">{display ?? value.toFixed(2)} {unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-blue-500 h-1.5 rounded cursor-pointer"
      />
    </div>
  );
}

function StatBadge({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col items-center bg-slate-800 rounded-lg px-3 py-2 min-w-[90px]">
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-sm font-mono font-semibold ${color}`}>{value}</span>
    </div>
  );
}

export default function App() {
  // Parameters
  const [E, setE] = useState(200e9);   // Pa (steel)
  const [A, setA] = useState(1e-4);    // m^2
  const [rho, setRho] = useState(7850); // kg/m^3
  const [g, setG] = useState(9.81);    // m/s^2
  const [dispScale, setDispScale] = useState(500);

  // Model
  const [preset, setPreset] = useState<PresetName>('cube');
  const [model, setModel] = useState<TrussModel>(() => buildCubeTruss(1.0));
  const [result, setResult] = useState<FEMResult | null>(null);
  const [solved, setSolved] = useState(false);
  const [solveTime, setSolveTime] = useState<number>(0);

  const loadPreset = useCallback((name: PresetName) => {
    setPreset(name);
    setModel(PRESETS[name]());
    setResult(null);
    setSolved(false);
  }, []);

  const handleSolve = useCallback(() => {
    const t0 = performance.now();
    const res = solveFEM({
      nodes: model.nodes,
      elements: model.elements,
      fixedNodes: model.fixedNodes,
      E,
      A,
      rho,
      g,
    });
    const t1 = performance.now();
    setSolveTime(t1 - t0);
    setResult(res);
    setSolved(true);
  }, [model, E, A, rho, g]);

  const handleReset = useCallback(() => {
    setResult(null);
    setSolved(false);
  }, []);

  const fmtSci = (v: number) => {
    if (v === 0) return '0';
    const exp = Math.floor(Math.log10(Math.abs(v)));
    const mant = v / Math.pow(10, exp);
    return `${mant.toFixed(2)}e${exp}`;
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Left sidebar */}
      <aside className="w-72 flex-shrink-0 bg-slate-900 border-r border-slate-700 flex flex-col overflow-y-auto">
        {/* Header */}
        <div className="px-4 py-3 border-b border-slate-700 bg-slate-800">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" />
            <h1 className="text-sm font-semibold tracking-wide text-white">3D Truss FEM</h1>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">Static analysis · Gravity loading</p>
        </div>

        <div className="flex-1 px-4 py-3 flex flex-col gap-5">
          {/* Preset selector */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Preset Model</h2>
            <div className="flex flex-col gap-1.5">
              {(Object.keys(PRESETS) as PresetName[]).map(name => (
                <button
                  key={name}
                  onClick={() => loadPreset(name)}
                  className={`text-sm px-3 py-1.5 rounded text-left transition-colors ${
                    preset === name
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {PRESET_LABELS[name]}
                  <span className="ml-2 text-xs text-slate-400">
                    {model.nodes.length === 0 || preset !== name ? '' :
                      `${model.nodes.length}n / ${model.elements.length}e`}
                  </span>
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">
              {model.nodes.length} nodes · {model.elements.length} elements ·{' '}
              {model.fixedNodes.length} fixed
            </p>
          </section>

          {/* Material / section */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Material & Section</h2>
            <div className="flex flex-col gap-3">
              <SliderRow
                label="Young's modulus E"
                value={E / 1e9}
                min={1}
                max={400}
                step={1}
                unit="GPa"
                display={(E / 1e9).toFixed(0)}
                onChange={v => setE(v * 1e9)}
              />
              <SliderRow
                label="Cross-section area A"
                value={A * 1e4}
                min={0.1}
                max={50}
                step={0.1}
                unit="cm²"
                display={(A * 1e4).toFixed(1)}
                onChange={v => setA(v / 1e4)}
              />
              <SliderRow
                label="Density ρ"
                value={rho}
                min={100}
                max={20000}
                step={100}
                unit="kg/m³"
                display={rho.toFixed(0)}
                onChange={setRho}
              />
            </div>
          </section>

          {/* Loading */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Loading</h2>
            <div className="flex flex-col gap-3">
              <SliderRow
                label="Gravity g"
                value={g}
                min={0}
                max={20}
                step={0.1}
                unit="m/s²"
                display={g.toFixed(2)}
                onChange={setG}
              />
            </div>
          </section>

          {/* Visualization */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Visualization</h2>
            <SliderRow
              label="Displacement scale"
              value={dispScale}
              min={1}
              max={5000}
              step={10}
              unit="×"
              display={dispScale.toFixed(0)}
              onChange={setDispScale}
            />
            <div className="mt-2 flex gap-2 text-xs">
              <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-red-400" /> Tension</span>
              <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-blue-400" /> Compression</span>
              <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-orange-400" /> Fixed node</span>
            </div>
          </section>
        </div>

        {/* Action buttons */}
        <div className="px-4 py-3 border-t border-slate-700 flex flex-col gap-2">
          <button
            onClick={handleSolve}
            className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors"
          >
            <Play className="w-4 h-4" />
            Solve
          </button>
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm py-1.5 rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </aside>

      {/* Main canvas area */}
      <div className="flex-1 flex flex-col relative">
        {/* Canvas */}
        <div className="flex-1">
          <SimulationCanvas
            nodes={model.nodes}
            elements={model.elements}
            fixedNodes={model.fixedNodes}
            result={result}
            dispScale={dispScale}
          />
        </div>

        {/* Results overlay */}
        {solved && result && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3 bg-slate-900/90 backdrop-blur px-4 py-2.5 rounded-xl border border-slate-700 shadow-xl">
            <StatBadge
              label="Max |disp|"
              value={`${fmtSci(result.maxDisplacement * 1000)} mm`}
              color="text-green-300"
            />
            <StatBadge
              label="Max stress"
              value={`${fmtSci(result.maxStress / 1e6)} MPa`}
              color="text-red-300"
            />
            <StatBadge
              label="Min stress"
              value={`${fmtSci(result.minStress / 1e6)} MPa`}
              color="text-blue-300"
            />
            <StatBadge
              label="Solve time"
              value={`${solveTime.toFixed(1)} ms`}
              color="text-slate-300"
            />
          </div>
        )}

        {/* Hint overlay when not yet solved */}
        {!solved && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center bg-slate-900/70 backdrop-blur px-6 py-4 rounded-xl border border-slate-700">
              <p className="text-slate-300 text-sm">Select a preset and press <span className="text-blue-400 font-semibold">Solve</span> to run FEM analysis</p>
              <p className="text-slate-500 text-xs mt-1">Orbit: drag · Zoom: scroll · Pan: right-drag</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
