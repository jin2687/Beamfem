import { useState, useCallback } from 'react';
import { Play, RotateCcw, Layers, AlertTriangle } from 'lucide-react';
import SimulationCanvas from './components/SimulationCanvas';
import {
  solveFEM,
  buildCubeTruss,
  buildBridgeTruss,
  buildGridTower,
  buildStraightChain,
  buildCatenaryChain,
  type FEMResult,
  type TrussModel,
} from './logic/femSolver';
import {
  solveBeam,
  buildBeamChain,
  buildPortalFrame,
  buildContinuousBeam,
  type BeamResult,
  type BeamModel,
  type BeamMaterial,
} from './logic/beamSolver';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SolverMode = 'truss' | 'beam';
type TrussPreset = 'cube' | 'bridge' | 'tower' | 'chain-straight' | 'chain-catenary';
type BeamPreset = 'chain' | 'portal' | 'continuous';

const TRUSS_PRESETS: Record<TrussPreset, () => TrussModel> = {
  cube: () => buildCubeTruss(1.0),
  bridge: () => buildBridgeTruss(),
  tower: () => buildGridTower(),
  'chain-straight': () => buildStraightChain(10, 4.0),
  'chain-catenary': () => buildCatenaryChain(14, 4.0, 0.7),
};
const TRUSS_LABELS: Record<TrussPreset, string> = {
  cube: 'Cube Truss', bridge: 'Bridge', tower: 'Grid Tower',
  'chain-straight': 'Straight Chain ⚠', 'chain-catenary': 'Catenary Chain',
};
const TRUSS_GROUPS: { label: string; names: TrussPreset[] }[] = [
  { label: 'Standard', names: ['cube', 'bridge', 'tower'] },
  { label: 'Chain / Degenerate', names: ['chain-straight', 'chain-catenary'] },
];

const BEAM_PRESETS: Record<BeamPreset, () => BeamModel> = {
  chain: () => buildBeamChain(12, 4.0),
  portal: () => buildPortalFrame(),
  continuous: () => buildContinuousBeam(5, 1.2),
};
const BEAM_LABELS: Record<BeamPreset, string> = {
  chain: 'Straight Beam Chain',
  portal: 'Portal Frame',
  continuous: 'Continuous Beam',
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SliderRow({
  label, value, min, max, step, unit, display, onChange,
}: {
  label: string; value: number; min: number; max: number;
  step: number; unit: string; display?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-slate-300">
        <span>{label}</span>
        <span className="font-mono text-blue-300">{display ?? value.toFixed(2)} {unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-blue-500 h-1.5 rounded cursor-pointer" />
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

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [mode, setMode] = useState<SolverMode>('truss');

  // Shared params
  const [E, setE] = useState(200e9);
  const [A, setA] = useState(1e-4);
  const [rho, setRho] = useState(7850);
  const [g, setG] = useState(9.81);
  const [dispScale, setDispScale] = useState(500);

  // Truss state
  const [trussPreset, setTrussPreset] = useState<TrussPreset>('cube');
  const [trussModel, setTrussModel] = useState<TrussModel>(() => buildCubeTruss(1.0));
  const [trussResult, setTrussResult] = useState<FEMResult | null>(null);

  // Beam state
  const [beamPreset, setBeamPreset] = useState<BeamPreset>('chain');
  const [beamModel, setBeamModel] = useState<BeamModel>(() => buildBeamChain(12, 4.0));
  const [beamResult, setBeamResult] = useState<BeamResult | null>(null);
  // Beam-specific: second moment of area (derived from A for solid circle by default)
  // I = A² / (4π) for solid circle
  const [useCustomI, setUseCustomI] = useState(false);
  const [customI, setCustomI] = useState(() => (1e-4) ** 2 / (4 * Math.PI)); // ≈ 7.96e-10

  const [solved, setSolved] = useState(false);
  const [solveTime, setSolveTime] = useState(0);
  const [isMechanism, setIsMechanism] = useState(false);
  const [singularCount, setSingularCount] = useState(0);

  const switchMode = (m: SolverMode) => {
    setMode(m);
    setSolved(false);
    setTrussResult(null);
    setBeamResult(null);
  };

  const loadTrussPreset = useCallback((name: TrussPreset) => {
    setTrussPreset(name);
    setTrussModel(TRUSS_PRESETS[name]());
    setTrussResult(null);
    setSolved(false);
  }, []);

  const loadBeamPreset = useCallback((name: BeamPreset) => {
    setBeamPreset(name);
    setBeamModel(BEAM_PRESETS[name]());
    setBeamResult(null);
    setSolved(false);
  }, []);

  const handleSolve = useCallback(() => {
    const t0 = performance.now();

    if (mode === 'truss') {
      const res = solveFEM({
        nodes: trussModel.nodes, elements: trussModel.elements,
        fixedNodes: trussModel.fixedNodes, E, A, rho, g,
      });
      setTrussResult(res);
      setIsMechanism(res.isMechanism);
      setSingularCount(res.singularDOFs.length);
    } else {
      // For beam: I = A²/(4π) for solid circular section
      const I = useCustomI ? customI : A * A / (4 * Math.PI);
      const nu = 0.3;
      const G = E / (2 * (1 + nu));
      const mat: BeamMaterial = { E, G, A, Iy: I, Iz: I, J: 2 * I, rho };
      const res = solveBeam({
        nodes: beamModel.nodes, elements: beamModel.elements,
        fixedNodes: beamModel.fixedNodes, material: mat, g,
      });
      setBeamResult(res);
      setIsMechanism(res.isMechanism);
      setSingularCount(res.singularDOFs.length);
    }

    setSolveTime(performance.now() - t0);
    setSolved(true);
  }, [mode, trussModel, beamModel, E, A, rho, g, useCustomI, customI]);

  const handleReset = () => {
    setTrussResult(null);
    setBeamResult(null);
    setSolved(false);
    setIsMechanism(false);
  };

  const fmtSci = (v: number) => {
    if (!isFinite(v) || v === 0) return v === 0 ? '0' : '∞';
    const exp = Math.floor(Math.log10(Math.abs(v)));
    return `${(v / 10 ** exp).toFixed(2)}e${exp}`;
  };

  const curResult = mode === 'truss' ? trussResult : beamResult;
  const maxDisp   = curResult?.maxDisplacement ?? 0;
  const I_display = useCustomI ? customI : A * A / (4 * Math.PI);

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-72 flex-shrink-0 bg-slate-900 border-r border-slate-700 flex flex-col overflow-y-auto">
        {/* Header */}
        <div className="px-4 py-3 border-b border-slate-700 bg-slate-800">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" />
            <h1 className="text-sm font-semibold tracking-wide text-white">3D Structural FEM</h1>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">Static analysis · Gravity loading</p>
        </div>

        <div className="flex-1 px-4 py-3 flex flex-col gap-5">
          {/* Solver mode toggle */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Solver Type</h2>
            <div className="flex gap-1">
              {(['truss', 'beam'] as SolverMode[]).map(m => (
                <button key={m} onClick={() => switchMode(m)}
                  className={`flex-1 py-1.5 rounded text-sm font-medium transition-colors ${
                    mode === m
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}>
                  {m === 'truss' ? 'Truss (3 DOF)' : 'Beam (6 DOF)'}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">
              {mode === 'truss'
                ? 'Axial force only — collinear chains are mechanisms'
                : 'Axial + bending + shear — straight chains solvable'}
            </p>
          </section>

          {/* Presets */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Preset Model</h2>

            {mode === 'truss' && (
              <div className="flex flex-col gap-3">
                {TRUSS_GROUPS.map(g => (
                  <div key={g.label}>
                    <p className="text-xs text-slate-600 mb-1">{g.label}</p>
                    <div className="flex flex-col gap-1">
                      {g.names.map(name => (
                        <button key={name} onClick={() => loadTrussPreset(name)}
                          className={`text-sm px-3 py-1.5 rounded text-left transition-colors ${
                            trussPreset === name
                              ? 'bg-blue-600 text-white'
                              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                          }`}>
                          {TRUSS_LABELS[name]}
                          {trussPreset === name && (
                            <span className="ml-1.5 text-xs opacity-60">
                              {trussModel.nodes.length}n/{trussModel.elements.length}e
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {mode === 'beam' && (
              <div className="flex flex-col gap-1">
                {(Object.keys(BEAM_PRESETS) as BeamPreset[]).map(name => (
                  <button key={name} onClick={() => loadBeamPreset(name)}
                    className={`text-sm px-3 py-1.5 rounded text-left transition-colors ${
                      beamPreset === name
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}>
                    {BEAM_LABELS[name]}
                    {beamPreset === name && (
                      <span className="ml-1.5 text-xs opacity-60">
                        {beamModel.nodes.length}n/{beamModel.elements.length}e
                      </span>
                    )}
                  </button>
                ))}
                <p className="text-xs text-slate-500 mt-0.5">
                  Straight Beam Chain = the truss mechanism, now solvable
                </p>
              </div>
            )}
          </section>

          {/* Material */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Material &amp; Section
            </h2>
            <div className="flex flex-col gap-3">
              <SliderRow label="Young's modulus E" value={E / 1e9}
                min={1} max={400} step={1} unit="GPa" display={(E / 1e9).toFixed(0)}
                onChange={v => setE(v * 1e9)} />
              <SliderRow label="Cross-section area A" value={A * 1e4}
                min={0.1} max={50} step={0.1} unit="cm²" display={(A * 1e4).toFixed(1)}
                onChange={v => setA(v / 1e4)} />
              <SliderRow label="Density ρ" value={rho}
                min={100} max={20000} step={100} unit="kg/m³" display={rho.toFixed(0)}
                onChange={setRho} />

              {mode === 'beam' && (
                <div className="flex flex-col gap-2 border-t border-slate-700 pt-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Second moment of area I</span>
                    <button
                      onClick={() => setUseCustomI(v => !v)}
                      className={`px-2 py-0.5 rounded text-xs ${useCustomI ? 'bg-blue-700' : 'bg-slate-700'}`}>
                      {useCustomI ? 'custom' : 'auto'}
                    </button>
                  </div>
                  {useCustomI ? (
                    <SliderRow label="" value={I_display * 1e8}
                      min={0.01} max={100} step={0.01} unit="×10⁻⁸ m⁴"
                      display={(I_display * 1e8).toFixed(2)}
                      onChange={v => setCustomI(v / 1e8)} />
                  ) : (
                    <p className="text-xs text-slate-500">
                      Auto (solid circle): I = A²/4π = {(I_display * 1e10).toFixed(2)} ×10⁻¹⁰ m⁴
                    </p>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* Load */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Loading</h2>
            <SliderRow label="Gravity g" value={g}
              min={0} max={20} step={0.1} unit="m/s²" display={g.toFixed(2)}
              onChange={setG} />
          </section>

          {/* Visualization */}
          <section>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Visualization</h2>
            <SliderRow label="Displacement scale" value={dispScale}
              min={1} max={5000} step={10} unit="×" display={dispScale.toFixed(0)}
              onChange={setDispScale} />
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-red-400 inline-block" />
                {mode === 'truss' ? 'Tension' : 'Sagging'}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-blue-400 inline-block" />
                {mode === 'truss' ? 'Compression' : 'Hogging'}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-orange-400 inline-block" />Fixed node
              </span>
            </div>
          </section>
        </div>

        {/* Actions */}
        <div className="px-4 py-3 border-t border-slate-700 flex flex-col gap-2">
          <button onClick={handleSolve}
            className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors">
            <Play className="w-4 h-4" /> Solve
          </button>
          <button onClick={handleReset}
            className="flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm py-1.5 rounded-lg transition-colors">
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
        </div>
      </aside>

      {/* Canvas */}
      <div className="flex-1 flex flex-col relative">
        <div className="flex-1">
          {mode === 'truss' ? (
            <SimulationCanvas mode="truss"
              nodes={trussModel.nodes} elements={trussModel.elements}
              fixedNodes={trussModel.fixedNodes}
              result={trussResult} dispScale={dispScale} />
          ) : (
            <SimulationCanvas mode="beam"
              nodes={beamModel.nodes} elements={beamModel.elements}
              fixedNodes={beamModel.fixedNodes}
              result={beamResult} dispScale={dispScale} />
          )}
        </div>

        {/* Mechanism warning */}
        {solved && isMechanism && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-start gap-2
            bg-amber-950/90 backdrop-blur px-4 py-2.5 rounded-xl border border-amber-600
            shadow-xl max-w-md">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-amber-300 text-xs font-semibold">機構（Mechanism）— 剛性行列が特異</p>
              <p className="text-amber-500 text-xs mt-0.5">
                この構造は横方向の剛性ゼロです。<br />
                <strong className="text-amber-400">Beam (6 DOF)</strong> モードに切り替えると、
                曲げ剛性が生じて同じ直線チェーンが解けます。
              </p>
              <p className="text-amber-600 text-xs mt-0.5">特異 DOF 数: {singularCount}</p>
            </div>
          </div>
        )}

        {/* Result stats */}
        {solved && curResult && !isMechanism && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3
            bg-slate-900/90 backdrop-blur px-4 py-2.5 rounded-xl border border-slate-700 shadow-xl">
            <StatBadge label="Max |disp|"
              value={`${fmtSci(maxDisp * 1000)} mm`}
              color="text-green-300" />
            {mode === 'truss' && trussResult && (
              <>
                <StatBadge label="Max σ"
                  value={`${fmtSci(trussResult.maxStress / 1e6)} MPa`}
                  color="text-red-300" />
                <StatBadge label="Min σ"
                  value={`${fmtSci(trussResult.minStress / 1e6)} MPa`}
                  color="text-blue-300" />
              </>
            )}
            {mode === 'beam' && beamResult && (
              <>
                <StatBadge label="Max bend. σ"
                  value={`${fmtSci(beamResult.maxBendingStress / 1e6)} MPa`}
                  color="text-red-300" />
                <StatBadge label="Min bend. σ"
                  value={`${fmtSci(beamResult.minBendingStress / 1e6)} MPa`}
                  color="text-blue-300" />
              </>
            )}
            <StatBadge label="Solve" value={`${solveTime.toFixed(1)} ms`} color="text-slate-300" />
          </div>
        )}

        {/* Hint */}
        {!solved && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center bg-slate-900/70 backdrop-blur px-6 py-4 rounded-xl border border-slate-700">
              <p className="text-slate-300 text-sm">
                Select a preset and press <span className="text-blue-400 font-semibold">Solve</span>
              </p>
              <p className="text-slate-500 text-xs mt-1">Orbit: drag · Zoom: scroll · Pan: right-drag</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
