import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import * as THREE from 'three';
import type { Node, Element, FEMResult } from '../logic/femSolver';

interface TrussSceneProps {
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[];
  result: FEMResult | null;
  dispScale: number;
}

/** Map a stress value to a color: blue=compression, grey=zero, red=tension */
function stressColor(stress: number, minS: number, maxS: number): THREE.Color {
  const absMax = Math.max(Math.abs(minS), Math.abs(maxS), 1e-10);
  const norm = stress / absMax; // -1 to 1
  if (norm > 0) {
    // tension: interpolate white -> red
    return new THREE.Color(1, 1 - norm, 1 - norm);
  } else {
    // compression: interpolate white -> blue
    const t = -norm;
    return new THREE.Color(1 - t, 1 - t, 1);
  }
}

/** A single truss element rendered as a cylinder */
function TrussElement({
  pA, pB, color,
}: {
  pA: THREE.Vector3;
  pB: THREE.Vector3;
  color: THREE.Color;
}) {
  const mid = useMemo(() => pA.clone().add(pB).multiplyScalar(0.5), [pA, pB]);
  const dir = useMemo(() => pB.clone().sub(pA), [pA, pB]);
  const length = dir.length();
  const quaternion = useMemo(() => {
    const q = new THREE.Quaternion();
    const axis = new THREE.Vector3(0, 1, 0);
    q.setFromUnitVectors(axis, dir.clone().normalize());
    return q;
  }, [dir]);

  const mat = useMemo(
    () => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.3 }),
    [color]
  );

  return (
    <mesh position={mid} quaternion={quaternion}>
      <cylinderGeometry args={[0.018, 0.018, length, 8]} />
      <primitive object={mat} attach="material" />
    </mesh>
  );
}

/** Node sphere */
function NodeSphere({
  pos,
  fixed,
}: {
  pos: THREE.Vector3;
  fixed: boolean;
}) {
  return (
    <mesh position={pos}>
      <sphereGeometry args={[0.04, 12, 12]} />
      <meshStandardMaterial color={fixed ? '#f97316' : '#94a3b8'} />
    </mesh>
  );
}

function TrussScene({ nodes, elements, fixedNodes, result, dispScale }: TrussSceneProps) {
  const fixedSet = useMemo(() => new Set(fixedNodes), [fixedNodes]);

  // Compute displaced node positions
  const positions = useMemo<THREE.Vector3[]>(() => {
    return nodes.map((n, i) => {
      const dx = result ? result.displacements[i * 3] * dispScale : 0;
      const dy = result ? result.displacements[i * 3 + 1] * dispScale : 0;
      const dz = result ? result.displacements[i * 3 + 2] * dispScale : 0;
      return new THREE.Vector3(n.x + dx, n.y + dy, n.z + dz);
    });
  }, [nodes, result, dispScale]);

  // Bounding box center for camera target
  const center = useMemo(() => {
    if (nodes.length === 0) return new THREE.Vector3(0, 0, 0);
    let sumX = 0, sumY = 0, sumZ = 0;
    for (const n of nodes) { sumX += n.x; sumY += n.y; sumZ += n.z; }
    return new THREE.Vector3(sumX / nodes.length, sumY / nodes.length, sumZ / nodes.length);
  }, [nodes]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-3, -4, -3]} intensity={0.4} />

      <group position={center.clone().negate()}>
        {/* Undeformed ghost (grey, low opacity) when result exists */}

        {result && (
          <group>
            {elements.map(el => {
              const pA = new THREE.Vector3(nodes[el.nodeA].x, nodes[el.nodeA].y, nodes[el.nodeA].z);
              const pB = new THREE.Vector3(nodes[el.nodeB].x, nodes[el.nodeB].y, nodes[el.nodeB].z);
              return (
                <mesh key={`ghost-${el.id}`}
                  position={pA.clone().add(pB).multiplyScalar(0.5)}
                  quaternion={(() => {
                    const q = new THREE.Quaternion();
                    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pB.clone().sub(pA).normalize());
                    return q;
                  })()}>
                  <cylinderGeometry args={[0.010, 0.010, pA.distanceTo(pB), 6]} />
                  <meshStandardMaterial color="#334155" transparent opacity={0.3} />
                </mesh>
              );
            })}
          </group>
        )}

        {/* Deformed structure */}
        {elements.map(el => {
          const pA = positions[el.nodeA];
          const pB = positions[el.nodeB];
          let color = new THREE.Color('#64748b');
          if (result) {
            color = stressColor(
              result.axialStresses[el.id],
              result.minStress,
              result.maxStress
            );
          }
          return <TrussElement key={el.id} pA={pA} pB={pB} color={color} />;
        })}

        {/* Nodes */}
        {positions.map((pos, i) => (
          <NodeSphere key={i} pos={pos} fixed={fixedSet.has(i)} />
        ))}
      </group>

      <OrbitControls makeDefault enablePan enableZoom enableRotate dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
      </GizmoHelper>
    </>
  );
}

interface Props {
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[];
  result: FEMResult | null;
  dispScale: number;
}

export default function SimulationCanvas({ nodes, elements, fixedNodes, result, dispScale }: Props) {
  return (
    <Canvas
      camera={{ position: [3, 2.5, 3.5], fov: 45 }}
      style={{ background: '#0f172a' }}
      shadows
    >
      <TrussScene
        nodes={nodes}
        elements={elements}
        fixedNodes={fixedNodes}
        result={result}
        dispScale={dispScale}
      />
    </Canvas>
  );
}
