/**
 * 3D Truss FEM Solver
 * Solves K*u = f for linear static analysis under gravity loading.
 *
 * Each node has 3 DOFs: (x, y, z).
 * Truss elements carry axial force only.
 */

export interface Node {
  id: number;
  x: number; // m
  y: number; // m
  z: number; // m
}

export interface Element {
  id: number;
  nodeA: number; // node index
  nodeB: number; // node index
}

export interface BoundaryCondition {
  nodeId: number;
  dof: 0 | 1 | 2; // 0=x, 1=y, 2=z
}

export interface FEMInput {
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[]; // node indices that are fully fixed (all 3 DOFs)
  E: number; // Young's modulus (Pa)
  A: number; // Cross-sectional area (m^2)
  rho: number; // Density (kg/m^3) — for computing self-weight
  g: number; // Gravitational acceleration (m/s^2)
}

export interface FEMResult {
  displacements: Float64Array; // length = nNodes * 3
  axialForces: Float64Array; // length = nElements (positive = tension)
  axialStresses: Float64Array; // axialForces / A
  maxDisplacement: number;
  minStress: number;
  maxStress: number;
  /** true when the stiffness matrix was (near-)singular → mechanism */
  isMechanism: boolean;
  /** free DOFs whose pivot was ~0, indicating zero-stiffness directions */
  singularDOFs: number[];
}

/** Assemble global stiffness matrix and solve Ku=f */
export function solveFEM(input: FEMInput): FEMResult {
  const { nodes, elements, fixedNodes, E, A, rho, g } = input;
  const nNodes = nodes.length;
  const nDOF = nNodes * 3;

  // --- Assemble K (dense, stored as flat row-major Float64Array) ---
  const K = new Float64Array(nDOF * nDOF);

  for (const el of elements) {
    const nA = el.nodeA;
    const nB = el.nodeB;
    const dx = nodes[nB].x - nodes[nA].x;
    const dy = nodes[nB].y - nodes[nA].y;
    const dz = nodes[nB].z - nodes[nA].z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-12) continue;

    const cx = dx / L;
    const cy = dy / L;
    const cz = dz / L;

    // Local-to-global direction cosines: t = [cx, cy, cz, -cx, -cy, -cz]
    const t = [cx, cy, cz, -cx, -cy, -cz];
    const k0 = (E * A) / L;

    // 6x6 element stiffness in global coords: k_e = k0 * t^T * t
    const dofs = [nA * 3, nA * 3 + 1, nA * 3 + 2, nB * 3, nB * 3 + 1, nB * 3 + 2];
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        K[dofs[i] * nDOF + dofs[j]] += k0 * t[i] * t[j];
      }
    }
  }

  // --- Assemble force vector (gravity = -g in y direction) ---
  // Each element contributes half its weight to each end node
  const f = new Float64Array(nDOF);
  for (const el of elements) {
    const nA = el.nodeA;
    const nB = el.nodeB;
    const dx = nodes[nB].x - nodes[nA].x;
    const dy = nodes[nB].y - nodes[nA].y;
    const dz = nodes[nB].z - nodes[nA].z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const weight = rho * A * L * g; // total weight of element (N)
    // distribute half to each node, acting in -y direction
    f[nA * 3 + 1] -= weight / 2;
    f[nB * 3 + 1] -= weight / 2;
  }

  // --- Apply boundary conditions (penalty / elimination) ---
  // Use large-number penalty method for simplicity
  const fixedDOFs = new Set<number>();
  for (const nodeId of fixedNodes) {
    fixedDOFs.add(nodeId * 3);
    fixedDOFs.add(nodeId * 3 + 1);
    fixedDOFs.add(nodeId * 3 + 2);
  }

  // Zero out rows/cols for fixed DOFs and set diagonal to 1 (enforce u=0)
  for (const dof of fixedDOFs) {
    for (let j = 0; j < nDOF; j++) {
      K[dof * nDOF + j] = 0;
      K[j * nDOF + dof] = 0;
    }
    K[dof * nDOF + dof] = 1;
    f[dof] = 0;
  }

  // --- Solve K*u = f using Gaussian elimination ---
  const singularDOFs: number[] = [];
  const u = gaussElimination(K, f, nDOF, fixedDOFs, singularDOFs);

  // --- Compute axial forces / stresses ---
  const axialForces = new Float64Array(elements.length);
  const axialStresses = new Float64Array(elements.length);

  for (let ei = 0; ei < elements.length; ei++) {
    const el = elements[ei];
    const nA = el.nodeA;
    const nB = el.nodeB;
    const dx = nodes[nB].x - nodes[nA].x;
    const dy = nodes[nB].y - nodes[nA].y;
    const dz = nodes[nB].z - nodes[nA].z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-12) continue;
    const cx = dx / L;
    const cy = dy / L;
    const cz = dz / L;

    const uAx = u[nA * 3];
    const uAy = u[nA * 3 + 1];
    const uAz = u[nA * 3 + 2];
    const uBx = u[nB * 3];
    const uBy = u[nB * 3 + 1];
    const uBz = u[nB * 3 + 2];

    // Axial elongation projected on element axis
    const delta = cx * (uBx - uAx) + cy * (uBy - uAy) + cz * (uBz - uAz);
    const force = (E * A / L) * delta;
    axialForces[ei] = force;
    axialStresses[ei] = force / A;
  }

  const dispArr = Array.from(u);
  const hasNaN = dispArr.some(v => !isFinite(v));
  const maxDisp = hasNaN ? Infinity : Math.max(...dispArr.map(Math.abs));
  const stressArr = Array.from(axialStresses);
  const minStress = Math.min(...stressArr.filter(isFinite));
  const maxStress = Math.max(...stressArr.filter(isFinite));
  const isMechanism = singularDOFs.length > 0 || hasNaN;

  return {
    displacements: u,
    axialForces,
    axialStresses,
    maxDisplacement: maxDisp,
    minStress: isFinite(minStress) ? minStress : 0,
    maxStress: isFinite(maxStress) ? maxStress : 0,
    isMechanism,
    singularDOFs,
  };
}

/** Simple Gaussian elimination with partial pivoting. Returns solution vector u.
 *  singularDOFs is populated with column indices whose pivot was ~0 (free DOF = mechanism). */
function gaussElimination(
  K: Float64Array,
  f: Float64Array,
  n: number,
  fixedDOFs: Set<number>,
  singularDOFs: number[],
): Float64Array {
  // Work on copies
  const A = new Float64Array(K);
  const b = new Float64Array(f);
  // Track original column indices through row swaps (to map back to DOF numbers)
  const colIdx = Array.from({ length: n }, (_, i) => i);

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxVal = Math.abs(A[col * n + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(A[row * n + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = A[col * n + j]; A[col * n + j] = A[maxRow * n + j]; A[maxRow * n + j] = tmp;
      }
      const tmp = b[col]; b[col] = b[maxRow]; b[maxRow] = tmp;
      const ti = colIdx[col]; colIdx[col] = colIdx[maxRow]; colIdx[maxRow] = ti;
    }

    const pivot = A[col * n + col];
    // Threshold relative to max diagonal for detecting singularity
    if (Math.abs(pivot) < 1e-10) {
      // Only flag as mechanism if this is NOT a fixed DOF (those have diag=1 exactly)
      if (!fixedDOFs.has(colIdx[col])) {
        singularDOFs.push(colIdx[col]);
      }
      continue;
    }

    for (let row = col + 1; row < n; row++) {
      const factor = A[row * n + col] / pivot;
      for (let j = col; j < n; j++) A[row * n + j] -= factor * A[col * n + j];
      b[row] -= factor * b[col];
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) sum -= A[i * n + j] * x[j];
    const diag = A[i * n + i];
    x[i] = Math.abs(diag) > 1e-10 ? sum / diag : 0;
  }
  return x;
}

// ---------------------------------------------------------------------------
// Preset geometry builders
// ---------------------------------------------------------------------------

export interface TrussModel {
  nodes: Node[];
  elements: Element[];
  fixedNodes: number[];
}

/** 2×2×2 cube truss (8 corner nodes + face/body diagonals for stability) */
export function buildCubeTruss(size = 1.0): TrussModel {
  const nodes: Node[] = [];
  let id = 0;
  for (let iz = 0; iz <= 1; iz++) {
    for (let iy = 0; iy <= 1; iy++) {
      for (let ix = 0; ix <= 1; ix++) {
        nodes.push({ id: id++, x: ix * size, y: iy * size, z: iz * size });
      }
    }
  }
  // Bottom nodes (y=0): indices 0,1,2,3  (ix+iz*2 mapping: 0=(0,0,0),1=(1,0,0),2=(0,0,1),3=(1,0,1))
  // Top nodes (y=1):    indices 4,5,6,7

  const edges: [number, number][] = [
    // Bottom face edges
    [0, 1], [2, 3], [0, 2], [1, 3],
    // Top face edges
    [4, 5], [6, 7], [4, 6], [5, 7],
    // Vertical pillars
    [0, 4], [1, 5], [2, 6], [3, 7],
    // Bottom face diagonals
    [0, 3], [1, 2],
    // Top face diagonals
    [4, 7], [5, 6],
    // Side face diagonals
    [0, 5], [1, 4],  // front face
    [2, 7], [3, 6],  // back face
    [0, 6], [2, 4],  // left face
    [1, 7], [3, 5],  // right face
    // Body diagonals
    [0, 7], [1, 6],
  ];

  const elements: Element[] = edges.map(([a, b], i) => ({ id: i, nodeA: a, nodeB: b }));
  // Fix bottom 4 nodes
  const fixedNodes = [0, 1, 2, 3];
  return { nodes, elements, fixedNodes };
}

/** Simple 3-span bridge along X axis */
export function buildBridgeTruss(): TrussModel {
  const L = 2.0; // span length per bay
  const H = 1.0; // height
  const W = 0.8; // half-width

  const nodes: Node[] = [];
  let nid = 0;
  const nSpans = 4;
  // Bottom chord: z = -W and z = +W
  for (let i = 0; i <= nSpans; i++) {
    nodes.push({ id: nid++, x: i * L, y: 0, z: -W });
    nodes.push({ id: nid++, x: i * L, y: 0, z:  W });
  }
  // Top chord
  for (let i = 0; i <= nSpans; i++) {
    nodes.push({ id: nid++, x: i * L, y: H, z: -W });
    nodes.push({ id: nid++, x: i * L, y: H, z:  W });
  }

  const nPerRow = (nSpans + 1) * 2; // nodes per chord level
  // helper: bottom[i*2+side], top[i*2+side]
  const B = (i: number, s: number) => i * 2 + s;
  const T = (i: number, s: number) => nPerRow + i * 2 + s;

  const edges: [number, number][] = [];

  for (let i = 0; i < nSpans; i++) {
    for (const s of [0, 1]) {
      // Bottom chord along X
      edges.push([B(i, s), B(i + 1, s)]);
      // Top chord along X
      edges.push([T(i, s), T(i + 1, s)]);
      // Vertical pillars
      edges.push([B(i, s), T(i, s)]);
      // Diagonal in XY plane
      edges.push([B(i, s), T(i + 1, s)]);
    }
    // Cross bracing in ZX plane (bottom and top)
    edges.push([B(i, 0), B(i, 1)]);
    edges.push([T(i, 0), T(i, 1)]);
    // End panels
    edges.push([B(i, 0), T(i, 1)]);
    edges.push([B(i, 1), T(i, 0)]);
  }
  // Last cross members
  edges.push([B(nSpans, 0), B(nSpans, 1)]);
  edges.push([T(nSpans, 0), T(nSpans, 1)]);
  for (const s of [0, 1]) edges.push([B(nSpans, s), T(nSpans, s)]);

  const elements: Element[] = edges.map(([a, b], i) => ({ id: i, nodeA: a, nodeB: b }));

  // Fix first two and last two bottom nodes
  const fixedNodes = [B(0, 0), B(0, 1), B(nSpans, 0), B(nSpans, 1)];
  return { nodes, elements, fixedNodes };
}

/** 3D grid tower 2×2 bays, 3 stories */
export function buildGridTower(): TrussModel {
  const BW = 1.0; // bay width
  const SH = 1.2; // story height
  const nBayX = 2;
  const nBayZ = 2;
  const nStories = 3;

  const nodes: Node[] = [];
  let nid = 0;
  const idx: number[][][] = []; // idx[ix][iy][iz]

  for (let iy = 0; iy <= nStories; iy++) {
    idx[iy] = [];
    for (let ix = 0; ix <= nBayX; ix++) {
      idx[iy][ix] = [];
      for (let iz = 0; iz <= nBayZ; iz++) {
        idx[iy][ix][iz] = nid;
        nodes.push({ id: nid++, x: ix * BW, y: iy * SH, z: iz * BW });
      }
    }
  }

  const edges: [number, number][] = [];

  for (let iy = 0; iy <= nStories; iy++) {
    for (let ix = 0; ix <= nBayX; ix++) {
      for (let iz = 0; iz <= nBayZ; iz++) {
        // Horizontal X
        if (ix < nBayX) edges.push([idx[iy][ix][iz], idx[iy][ix + 1][iz]]);
        // Horizontal Z
        if (iz < nBayZ) edges.push([idx[iy][ix][iz], idx[iy][ix][iz + 1]]);
        // Vertical Y
        if (iy < nStories) edges.push([idx[iy][ix][iz], idx[iy + 1][ix][iz]]);
      }
    }
    // Face diagonals per story
    if (iy < nStories) {
      for (let ix = 0; ix < nBayX; ix++) {
        for (let iz = 0; iz < nBayZ; iz++) {
          // XY plane
          edges.push([idx[iy][ix][iz], idx[iy + 1][ix + 1][iz]]);
          // ZY plane
          edges.push([idx[iy][ix][iz], idx[iy + 1][ix][iz + 1]]);
          // XZ diagonal in horizontal plane
          edges.push([idx[iy][ix][iz], idx[iy][ix + 1][iz + 1]]);
        }
      }
    }
  }

  const elements: Element[] = edges.map(([a, b], i) => ({ id: i, nodeA: a, nodeB: b }));
  // Fix all bottom nodes
  const fixedNodes: number[] = [];
  for (let ix = 0; ix <= nBayX; ix++)
    for (let iz = 0; iz <= nBayZ; iz++)
      fixedNodes.push(idx[0][ix][iz]);

  return { nodes, elements, fixedNodes };
}

// ---------------------------------------------------------------------------
// Unstable / degenerate topology examples
// ---------------------------------------------------------------------------

/**
 * Straight horizontal chain: n nodes equally spaced along X, connected in a
 * single line, both ends fixed.
 *
 * This is a MECHANISM under transverse (gravity) load — the stiffness matrix
 * is singular for all y/z DOFs, so the solver will flag isMechanism = true
 * and displacements will be zero (no lateral stiffness at all).
 */
export function buildStraightChain(nNodes = 8, length = 4.0): TrussModel {
  const nodes: Node[] = [];
  for (let i = 0; i < nNodes; i++) {
    nodes.push({ id: i, x: (i / (nNodes - 1)) * length, y: 0, z: 0 });
  }
  const elements: Element[] = [];
  for (let i = 0; i < nNodes - 1; i++) {
    elements.push({ id: i, nodeA: i, nodeB: i + 1 });
  }
  // Fix both ends (all 3 DOFs)
  const fixedNodes = [0, nNodes - 1];
  return { nodes, elements, fixedNodes };
}

/**
 * Pre-sagged catenary chain: same topology as the straight chain, but nodes
 * are placed along a parabolic curve (y = -sag * 4x/L*(1-x/L)).
 *
 * Because elements are no longer collinear, they CAN develop lateral
 * components and the structure IS stable under gravity — this is the
 * principle behind suspension cables and catenary bridges.
 *
 * Under self-weight the chain further sags and each element is in tension.
 */
export function buildCatenaryChain(
  nNodes = 12,
  length = 4.0,
  sag = 0.6,
): TrussModel {
  const nodes: Node[] = [];
  for (let i = 0; i < nNodes; i++) {
    const t = i / (nNodes - 1); // 0 → 1
    const x = t * length;
    const y = -sag * 4 * t * (1 - t); // parabolic sag, max at mid-span
    nodes.push({ id: i, x, y, z: 0 });
  }
  const elements: Element[] = [];
  for (let i = 0; i < nNodes - 1; i++) {
    elements.push({ id: i, nodeA: i, nodeB: i + 1 });
  }
  const fixedNodes = [0, nNodes - 1];
  return { nodes, elements, fixedNodes };
}
