import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import * as THREE from 'three';
import type { Node, Element, FEMResult } from '../logic/femSolver';
import type { BeamNode, BeamElement, BeamResult } from '../logic/beamSolver';

// ---------------------------------------------------------------------------
// Shared color helpers
// ---------------------------------------------------------------------------

function stressColor(stress: number, minS: number, maxS: number): THREE.Color {
  const absMax = Math.max(Math.abs(minS), Math.abs(maxS), 1e-10);
  const norm = Math.max(-1, Math.min(1, stress / absMax));
  if (norm > 0) return new THREE.Color(1, 1 - norm, 1 - norm);   // tension → red
  const t = -norm;
  return new THREE.Color(1 - t, 1 - t, 1);                        // compression → blue
}

// ---------------------------------------------------------------------------
// Truss rendering
// ---------------------------------------------------------------------------

function TrussCylinder({
  pA, pB, color,
}: { pA: THREE.Vector3; pB: THREE.Vector3; color: THREE.Color }) {
  const mid = useMemo(() => pA.clone().add(pB).multiplyScalar(0.5), [pA, pB]);
  const dir = useMemo(() => pB.clone().sub(pA), [pA, pB]);
  const length = dir.length();
  const quat = useMemo(() => {
    const q = new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    return q;
  }, [dir]);
  return (
    <mesh position={mid} quaternion={quat}>
      <cylinderGeometry args={[0.018, 0.018, length, 8]} />
      <meshStandardMaterial color={color} roughness={0.6} metalness={0.3} />
    </mesh>
  );
}

interface TrussSceneProps {
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[];
  result: FEMResult | null;
  dispScale: number;
}

function TrussScene({ nodes, elements, fixedNodes, result, dispScale }: TrussSceneProps) {
  const fixedSet = useMemo(() => new Set(fixedNodes), [fixedNodes]);

  const positions = useMemo<THREE.Vector3[]>(() => nodes.map((n, i) => {
    const dx = result ? result.displacements[i * 3] * dispScale : 0;
    const dy = result ? result.displacements[i * 3 + 1] * dispScale : 0;
    const dz = result ? result.displacements[i * 3 + 2] * dispScale : 0;
    return new THREE.Vector3(n.x + dx, n.y + dy, n.z + dz);
  }), [nodes, result, dispScale]);

  const center = useMemo(() => {
    if (!nodes.length) return new THREE.Vector3();
    let sx = 0, sy = 0, sz = 0;
    for (const n of nodes) { sx += n.x; sy += n.y; sz += n.z; }
    return new THREE.Vector3(sx / nodes.length, sy / nodes.length, sz / nodes.length);
  }, [nodes]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-3, -4, -3]} intensity={0.4} />

      <group position={center.clone().negate()}>
        {/* Ghost (undeformed) */}
        {result && elements.map(el => {
          const pA = new THREE.Vector3(nodes[el.nodeA].x, nodes[el.nodeA].y, nodes[el.nodeA].z);
          const pB = new THREE.Vector3(nodes[el.nodeB].x, nodes[el.nodeB].y, nodes[el.nodeB].z);
          const mid = pA.clone().add(pB).multiplyScalar(0.5);
          const dir = pB.clone().sub(pA);
          const L = dir.length();
          const q = new THREE.Quaternion();
          q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
          return (
            <mesh key={`g${el.id}`} position={mid} quaternion={q}>
              <cylinderGeometry args={[0.010, 0.010, L, 6]} />
              <meshStandardMaterial color="#334155" transparent opacity={0.3} />
            </mesh>
          );
        })}

        {/* Deformed colored elements */}
        {elements.map(el => {
          const pA = positions[el.nodeA];
          const pB = positions[el.nodeB];
          const color = result
            ? stressColor(result.axialStresses[el.id], result.minStress, result.maxStress)
            : new THREE.Color('#64748b');
          return <TrussCylinder key={el.id} pA={pA} pB={pB} color={color} />;
        })}

        {/* Nodes */}
        {positions.map((pos, i) => (
          <mesh key={i} position={pos}>
            <sphereGeometry args={[0.04, 12, 12]} />
            <meshStandardMaterial color={fixedSet.has(i) ? '#f97316' : '#94a3b8'} />
          </mesh>
        ))}
      </group>

      <OrbitControls makeDefault enablePan enableZoom enableRotate dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
      </GizmoHelper>
    </>
  );
}

// ---------------------------------------------------------------------------
// Beam rendering
// ---------------------------------------------------------------------------

/** Tube geometry through an array of world-space points */
function BeamCurve({
  points,
  color,
  radius,
}: { points: THREE.Vector3[]; color: THREE.Color; radius: number }) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    try {
      const curve = new THREE.CatmullRomCurve3(points);
      const g = new THREE.TubeGeometry(curve, points.length * 2, radius, 8, false);
      return g;
    } catch {
      return null;
    }
  }, [points, radius]);

  if (!geometry) return null;
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} roughness={0.5} metalness={0.3} />
    </mesh>
  );
}

interface BeamSceneProps {
  nodes: BeamNode[];
  elements: BeamElement[];
  fixedNodes: number[];
  result: BeamResult | null;
  dispScale: number;
}

function BeamScene({ nodes, elements, fixedNodes, result, dispScale }: BeamSceneProps) {
  const fixedSet = useMemo(() => new Set(fixedNodes), [fixedNodes]);

  /** Scale displaced curve: p = base + scale*(deformed - base) */
  const scaledCurves = useMemo<THREE.Vector3[][]>(() => {
    if (!result) {
      // straight lines only
      return elements.map(el => {
        const a = nodes[el.nodeA], b = nodes[el.nodeB];
        return [
          new THREE.Vector3(a.x, a.y, a.z),
          new THREE.Vector3(b.x, b.y, b.z),
        ];
      });
    }
    return result.curves.map((curve, ei) => {
      const base = result.curvesBase[ei];
      if (!curve.length || !base.length) return [];
      return curve.map(([x, y, z], pi) => {
        const [bx, by, bz] = base[pi];
        return new THREE.Vector3(
          bx + (x - bx) * dispScale,
          by + (y - by) * dispScale,
          bz + (z - bz) * dispScale,
        );
      });
    });
  }, [nodes, elements, result, dispScale]);

  const center = useMemo(() => {
    if (!nodes.length) return new THREE.Vector3();
    let sx = 0, sy = 0, sz = 0;
    for (const n of nodes) { sx += n.x; sy += n.y; sz += n.z; }
    return new THREE.Vector3(sx / nodes.length, sy / nodes.length, sz / nodes.length);
  }, [nodes]);

  // Beam display radius: rough guess from cross-section area
  const r = 0.03;

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-3, -4, -3]} intensity={0.4} />

      <group position={center.clone().negate()}>
        {/* Ghost (undeformed straight lines) */}
        {result && elements.map((_el, ei) => {
          const base = result.curvesBase[ei];
          if (!base.length) return null;
          const pts = base.map(([x, y, z]) => new THREE.Vector3(x, y, z));
          return <BeamCurve key={`g${ei}`} points={pts} color={new THREE.Color('#1e3a5f')} radius={r * 0.55} />;
        })}

        {/* Deformed colored beam curves */}
        {elements.map((_el, ei) => {
          const pts = scaledCurves[ei];
          if (!pts.length) return null;
          let color = new THREE.Color('#64748b');
          if (result) {
            color = stressColor(
              result.bendingStresses[ei],
              result.minBendingStress,
              result.maxBendingStress,
            );
          }
          return <BeamCurve key={ei} points={pts} color={color} radius={r} />;
        })}

        {/* Nodes */}
        {nodes.map((n, i) => {
          const u = result ? result.displacements : null;
          const dx = u ? u[i * 6] * dispScale : 0;
          const dy = u ? u[i * 6 + 1] * dispScale : 0;
          const dz = u ? u[i * 6 + 2] * dispScale : 0;
          return (
            <mesh key={i} position={[n.x + dx, n.y + dy, n.z + dz]}>
              <sphereGeometry args={[0.05, 12, 12]} />
              <meshStandardMaterial color={fixedSet.has(i) ? '#f97316' : '#94a3b8'} />
            </mesh>
          );
        })}
      </group>

      <OrbitControls makeDefault enablePan enableZoom enableRotate dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
      </GizmoHelper>
    </>
  );
}

// ---------------------------------------------------------------------------
// Public component — dispatches to Truss or Beam scene
// ---------------------------------------------------------------------------

interface TrussProps {
  mode: 'truss';
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[];
  result: FEMResult | null;
  dispScale: number;
}

interface BeamProps {
  mode: 'beam';
  nodes: BeamNode[];
  elements: BeamElement[];
  fixedNodes: number[];
  result: BeamResult | null;
  dispScale: number;
}

type Props = TrussProps | BeamProps;

export default function SimulationCanvas(props: Props) {
  return (
    <Canvas
      camera={{ position: [3, 2.5, 3.5], fov: 45 }}
      style={{ background: '#0f172a' }}
      shadows
    >
      {props.mode === 'truss' ? (
        <TrussScene
          nodes={props.nodes}
          elements={props.elements}
          fixedNodes={props.fixedNodes}
          result={props.result}
          dispScale={props.dispScale}
        />
      ) : (
        <BeamScene
          nodes={props.nodes}
          elements={props.elements}
          fixedNodes={props.fixedNodes}
          result={props.result}
          dispScale={props.dispScale}
        />
      )}
    </Canvas>
  );
}
