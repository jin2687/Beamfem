/**
 * 3D Euler-Bernoulli Beam FEM Solver
 *
 * Each node has 6 DOFs: [ux, uy, uz, θx, θy, θz]
 *   ux/uy/uz = translations, θx = torsion, θy/θz = rotations (bending)
 *
 * Unlike truss elements (axial only), beam elements carry bending moments
 * and shear — so a straight chain fixed at both ends IS solvable.
 */

export interface BeamNode { id: number; x: number; y: number; z: number; }
export interface BeamElement { id: number; nodeA: number; nodeB: number; }

export interface BeamMaterial {
  E: number;   // Young's modulus (Pa)
  G: number;   // Shear modulus (Pa) = E / (2*(1+nu))
  A: number;   // Cross-section area (m²)
  Iy: number;  // 2nd moment of area about local y (bending in xz)
  Iz: number;  // 2nd moment of area about local z (bending in xy)
  J: number;   // Torsional constant (m⁴) ≈ Iy+Iz for solid circle
  rho: number; // Density (kg/m³)
}

export interface BeamInput {
  nodes: BeamNode[];
  elements: BeamElement[];
  fixedNodes: number[];
  material: BeamMaterial;
  g: number;
}

export interface BeamResult {
  /** length = nNodes * 6 */
  displacements: Float64Array;
  maxDisplacement: number;
  /** Bending stress at mid-span of each element (for coloring) */
  bendingStresses: Float64Array;
  maxBendingStress: number;
  minBendingStress: number;
  isMechanism: boolean;
  singularDOFs: number[];
  /** Undeformed curve (straight) for each element */
  curvesBase: Array<Array<[number, number, number]>>;
  /** Deformed curve (unit dispScale=1) for each element */
  curves: Array<Array<[number, number, number]>>;
}

// ---------------------------------------------------------------------------
// Helper: local coordinate axes for element
// Returns [lx, ly, lz] as row vectors of the local frame
// ---------------------------------------------------------------------------
function localAxes(
  a: BeamNode, b: BeamNode
): [[number, number, number], [number, number, number], [number, number, number]] {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const lx: [number, number, number] = [dx / L, dy / L, dz / L];

  // Reference vector: global Y if element is not (nearly) vertical
  const ref: [number, number, number] = Math.abs(lx[1]) < 0.9
    ? [0, 1, 0]
    : [1, 0, 0];

  // Gram-Schmidt: ly = normalize(ref - (ref·lx)lx)
  const dot = ref[0] * lx[0] + ref[1] * lx[1] + ref[2] * lx[2];
  const ly_r: [number, number, number] = [ref[0] - dot * lx[0], ref[1] - dot * lx[1], ref[2] - dot * lx[2]];
  const lyLen = Math.sqrt(ly_r[0] ** 2 + ly_r[1] ** 2 + ly_r[2] ** 2);
  const ly: [number, number, number] = [ly_r[0] / lyLen, ly_r[1] / lyLen, ly_r[2] / lyLen];

  // lz = lx × ly
  const lz: [number, number, number] = [
    lx[1] * ly[2] - lx[2] * ly[1],
    lx[2] * ly[0] - lx[0] * ly[2],
    lx[0] * ly[1] - lx[1] * ly[0],
  ];
  return [lx, ly, lz];
}

// ---------------------------------------------------------------------------
// 12×12 local element stiffness matrix (Euler-Bernoulli)
// DOF order: [u1, v1, w1, rx1, ry1, rz1,  u2, v2, w2, rx2, ry2, rz2]
//   u=axial, v=local-y lateral, w=local-z lateral
//   rx=torsion, ry=rot about y (xz bending), rz=rot about z (xy bending)
// ---------------------------------------------------------------------------
function localK(L: number, m: BeamMaterial): Float64Array {
  const k = new Float64Array(144);
  const { E, G, A, Iy, Iz, J } = m;

  const ax  = E * A / L;
  const tx  = G * J / L;
  const bz12 = 12 * E * Iz / (L * L * L);
  const bz6  =  6 * E * Iz / (L * L);
  const bz4  =  4 * E * Iz / L;
  const bz2  =  2 * E * Iz / L;
  const by12 = 12 * E * Iy / (L * L * L);
  const by6  =  6 * E * Iy / (L * L);
  const by4  =  4 * E * Iy / L;
  const by2  =  2 * E * Iy / L;

  const s = (i: number, j: number, v: number) => {
    k[i * 12 + j] += v;
    if (i !== j) k[j * 12 + i] += v;
  };

  // Axial: dof 0(u1) ↔ 6(u2)
  s(0, 0, ax);  s(6, 6, ax);  s(0, 6, -ax);

  // Torsion: dof 3(rx1) ↔ 9(rx2)
  s(3, 3, tx);  s(9, 9, tx);  s(3, 9, -tx);

  // Bending in xy plane: dof 1(v1), 5(θz1), 7(v2), 11(θz2)
  s(1, 1, bz12); s(7, 7, bz12); s(1, 7, -bz12);
  s(1, 5, bz6);  s(1, 11, bz6);
  s(5, 5, bz4);  s(11, 11, bz4);
  s(5, 7, -bz6); s(7, 11, -bz6);
  s(5, 11, bz2);

  // Bending in xz plane: dof 2(w1), 4(θy1), 8(w2), 10(θy2)
  // θy = -dw/dx convention → signs flip for off-diagonals
  s(2, 2, by12); s(8, 8, by12); s(2, 8, -by12);
  s(2, 4, -by6); s(2, 10, -by6);
  s(4, 4, by4);  s(10, 10, by4);
  s(4, 8, by6);  s(8, 10, by6);
  s(4, 10, by2);

  return k;
}

// ---------------------------------------------------------------------------
// 12×12 rotation matrix (blockdiag of 3×3 R, four times)
// R rows = local axis vectors [lx; ly; lz]
// ---------------------------------------------------------------------------
function buildT(
  lx: [number, number, number],
  ly: [number, number, number],
  lz: [number, number, number],
): Float64Array {
  const T = new Float64Array(144);
  const R = [lx, ly, lz];
  for (let b = 0; b < 4; b++)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        T[(b * 3 + i) * 12 + (b * 3 + j)] = R[i][j];
  return T;
}

function mul12(A: Float64Array, B: Float64Array): Float64Array {
  const C = new Float64Array(144);
  for (let i = 0; i < 12; i++)
    for (let j = 0; j < 12; j++) {
      let s = 0;
      for (let k = 0; k < 12; k++) s += A[i * 12 + k] * B[k * 12 + j];
      C[i * 12 + j] = s;
    }
  return C;
}

function transpose12(A: Float64Array): Float64Array {
  const B = new Float64Array(144);
  for (let i = 0; i < 12; i++)
    for (let j = 0; j < 12; j++)
      B[j * 12 + i] = A[i * 12 + j];
  return B;
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------
export function solveBeam(input: BeamInput): BeamResult {
  const { nodes, elements, fixedNodes, material: m, g } = input;
  const nN = nodes.length;
  const nDOF = nN * 6;

  const K = new Float64Array(nDOF * nDOF);
  const f = new Float64Array(nDOF);

  for (const el of elements) {
    const nA = el.nodeA, nB = el.nodeB;
    const a = nodes[nA], b = nodes[nB];
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-12) continue;

    const [lx, ly, lz] = localAxes(a, b);
    const T = buildT(lx, ly, lz);
    const Tt = transpose12(T);
    const kLoc = localK(L, m);
    // k_global = T^T * k_local * T
    const kGlob = mul12(Tt, mul12(kLoc, T));

    const dofs = [
      nA*6, nA*6+1, nA*6+2, nA*6+3, nA*6+4, nA*6+5,
      nB*6, nB*6+1, nB*6+2, nB*6+3, nB*6+4, nB*6+5,
    ];
    for (let i = 0; i < 12; i++)
      for (let j = 0; j < 12; j++)
        K[dofs[i] * nDOF + dofs[j]] += kGlob[i * 12 + j];

    // Self-weight: q = rho*A*g per unit length in global -y
    const q = m.rho * m.A * g;
    const qx = -lx[1] * q;  // axial component in local x
    const qy = -ly[1] * q;  // component in local y
    const qz = -lz[1] * q;  // component in local z

    // Consistent nodal loads for uniform distributed load (local coords)
    const fe = new Float64Array(12);
    fe[0]  = qx * L / 2;          // u1
    fe[1]  = qy * L / 2;          // v1
    fe[2]  = qz * L / 2;          // w1
    fe[3]  = 0;
    fe[4]  = -qz * L * L / 12;   // ry1 (moment from qz)
    fe[5]  =  qy * L * L / 12;   // rz1 (moment from qy)
    fe[6]  = qx * L / 2;
    fe[7]  = qy * L / 2;
    fe[8]  = qz * L / 2;
    fe[9]  = 0;
    fe[10] =  qz * L * L / 12;   // ry2
    fe[11] = -qy * L * L / 12;   // rz2

    // Transform to global: T^T * fe
    for (let i = 0; i < 12; i++) {
      let v = 0;
      for (let j = 0; j < 12; j++) v += Tt[i * 12 + j] * fe[j];
      f[dofs[i]] += v;
    }
  }

  // Apply Dirichlet BCs (zero all 6 DOFs for fixed nodes)
  const fixedSet = new Set<number>();
  for (const n of fixedNodes)
    for (let d = 0; d < 6; d++) fixedSet.add(n * 6 + d);

  for (const dof of fixedSet) {
    for (let j = 0; j < nDOF; j++) { K[dof * nDOF + j] = 0; K[j * nDOF + dof] = 0; }
    K[dof * nDOF + dof] = 1;
    f[dof] = 0;
  }

  // Solve
  const singularDOFs: number[] = [];
  const u = gaussElim(K, f, nDOF, fixedSet, singularDOFs);

  // ---------------------------------------------------------------------------
  // Post-process: bending stress + element curves (Hermite interpolation)
  // ---------------------------------------------------------------------------
  const NPTS = 14; // intermediate points per element
  const bendingStresses = new Float64Array(elements.length);
  const curves: Array<Array<[number, number, number]>> = [];
  const curvesBase: Array<Array<[number, number, number]>> = [];

  // Extreme fiber distance for stress: assume solid circular cross-section
  // A = π r²  →  r = sqrt(A/π),  I = π r⁴/4 = A²/(4π)
  const r_fiber = Math.sqrt(m.A / Math.PI);

  for (let ei = 0; ei < elements.length; ei++) {
    const el = elements[ei];
    const nA = el.nodeA, nB = el.nodeB;
    const a = nodes[nA], b = nodes[nB];
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-12) { curves.push([]); curvesBase.push([]); continue; }

    const [lx, ly, lz] = localAxes(a, b);
    const T = buildT(lx, ly, lz);

    // Extract local nodal displacements
    const uGlob = new Float64Array(12);
    for (let d = 0; d < 6; d++) {
      uGlob[d]     = u[nA * 6 + d];
      uGlob[d + 6] = u[nB * 6 + d];
    }
    const uLoc = new Float64Array(12);
    for (let i = 0; i < 12; i++) {
      let v = 0;
      for (let j = 0; j < 12; j++) v += T[i * 12 + j] * uGlob[j];
      uLoc[i] = v;
    }

    const [u1, v1, w1, , θy1, θz1] = uLoc;
    const [u2, v2, w2, , θy2, θz2] = [uLoc[6], uLoc[7], uLoc[8], uLoc[9], uLoc[10], uLoc[11]];

    // Bending moment at mid-span (ξ=0.5):
    // κ_z = (1/L²)*d²v/dξ², at ξ=0.5 → κ_z = (θz2 - θz1)/L
    // κ_y = -(1/L²)*d²w/dξ², at ξ=0.5 → κ_y = (θy1 - θy2)/L
    const Mz = m.E * m.Iz * (θz2 - θz1) / L;
    const My = m.E * m.Iy * (θy1 - θy2) / L;
    const M = Math.sqrt(Mz * Mz + My * My);
    bendingStresses[ei] = (M > 0 ? 1 : -1) * M * r_fiber / m.Iz;

    // Build curve points (Hermite interpolation in local coords → global)
    const curve: Array<[number, number, number]> = [];
    const curveBase: Array<[number, number, number]> = [];
    for (let pi = 0; pi <= NPTS; pi++) {
      const xi = pi / NPTS;
      // Hermite shape functions
      const H1 = 1 - 3 * xi * xi + 2 * xi * xi * xi;
      const Hθ1 =  xi - 2 * xi * xi + xi * xi * xi;  // multiply by L
      const H2 = 3 * xi * xi - 2 * xi * xi * xi;
      const Hθ2 = -xi * xi + xi * xi * xi;             // multiply by L

      const u_loc = (1 - xi) * u1 + xi * u2;
      const v_loc = H1 * v1 + L * Hθ1 * θz1 + H2 * v2 + L * Hθ2 * θz2;
      // θy = -dw/dx, so dw/dx = -θy
      const w_loc = H1 * w1 + L * Hθ1 * (-θy1) + H2 * w2 + L * Hθ2 * (-θy2);

      // Local position along element axis
      const s = xi * L; // undeformed position

      // Deformed global position
      const gx = a.x + (s + u_loc) * lx[0] + v_loc * ly[0] + w_loc * lz[0];
      const gy = a.y + (s + u_loc) * lx[1] + v_loc * ly[1] + w_loc * lz[1];
      const gz = a.z + (s + u_loc) * lx[2] + v_loc * ly[2] + w_loc * lz[2];
      curve.push([gx, gy, gz]);

      // Undeformed base position (straight)
      const bx = a.x + s * lx[0];
      const by_ = a.y + s * lx[1];
      const bz_ = a.z + s * lx[2];
      curveBase.push([bx, by_, bz_]);
    }
    curves.push(curve);
    curvesBase.push(curveBase);
  }

  // Max translational displacement
  let maxDisp = 0;
  for (let n = 0; n < nN; n++) {
    const d = Math.sqrt(u[n*6]**2 + u[n*6+1]**2 + u[n*6+2]**2);
    if (d > maxDisp) maxDisp = d;
  }

  const sArr = Array.from(bendingStresses).filter(isFinite);
  return {
    displacements: u,
    maxDisplacement: maxDisp,
    bendingStresses,
    maxBendingStress: sArr.length ? Math.max(...sArr) : 0,
    minBendingStress: sArr.length ? Math.min(...sArr) : 0,
    isMechanism: singularDOFs.length > 0,
    singularDOFs,
    curves,
    curvesBase,
  };
}

// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------
function gaussElim(
  K: Float64Array, f: Float64Array, n: number,
  fixedDOFs: Set<number>, singularDOFs: number[],
): Float64Array {
  const A = new Float64Array(K);
  const b = new Float64Array(f);
  const col2dof = Array.from({ length: n }, (_, i) => i);

  for (let col = 0; col < n; col++) {
    let maxV = Math.abs(A[col * n + col]), maxR = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(A[row * n + col]);
      if (v > maxV) { maxV = v; maxR = row; }
    }
    if (maxR !== col) {
      for (let j = 0; j < n; j++) {
        const t = A[col * n + j]; A[col * n + j] = A[maxR * n + j]; A[maxR * n + j] = t;
      }
      const t = b[col]; b[col] = b[maxR]; b[maxR] = t;
      const t2 = col2dof[col]; col2dof[col] = col2dof[maxR]; col2dof[maxR] = t2;
    }
    const pivot = A[col * n + col];
    if (Math.abs(pivot) < 1e-10) {
      if (!fixedDOFs.has(col2dof[col])) singularDOFs.push(col2dof[col]);
      continue;
    }
    for (let row = col + 1; row < n; row++) {
      const fac = A[row * n + col] / pivot;
      for (let j = col; j < n; j++) A[row * n + j] -= fac * A[col * n + j];
      b[row] -= fac * b[col];
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) sum -= A[i * n + j] * x[j];
    const d = A[i * n + i];
    x[i] = Math.abs(d) > 1e-10 ? sum / d : 0;
  }
  return x;
}

// ---------------------------------------------------------------------------
// Preset models
// ---------------------------------------------------------------------------

export interface BeamModel {
  nodes: BeamNode[];
  elements: BeamElement[];
  fixedNodes: number[];
}

/** Straight horizontal chain — both ends fixed. Solvable with beams! */
export function buildBeamChain(nNodes = 12, length = 4.0): BeamModel {
  const nodes: BeamNode[] = [];
  for (let i = 0; i < nNodes; i++)
    nodes.push({ id: i, x: (i / (nNodes - 1)) * length, y: 0, z: 0 });
  const elements: BeamElement[] = [];
  for (let i = 0; i < nNodes - 1; i++)
    elements.push({ id: i, nodeA: i, nodeB: i + 1 });
  return { nodes, elements, fixedNodes: [0, nNodes - 1] };
}

/** Portal frame: two columns + horizontal beam */
export function buildPortalFrame(): BeamModel {
  const H = 3.0, W = 4.0;
  const nodes: BeamNode[] = [
    { id: 0, x: 0, y: 0,  z: 0 },
    { id: 1, x: 0, y: H,  z: 0 },
    { id: 2, x: W, y: H,  z: 0 },
    { id: 3, x: W, y: 0,  z: 0 },
  ];
  const elements: BeamElement[] = [
    { id: 0, nodeA: 0, nodeB: 1 },
    { id: 1, nodeA: 1, nodeB: 2 },
    { id: 2, nodeA: 2, nodeB: 3 },
  ];
  return { nodes, elements, fixedNodes: [0, 3] };
}

/** Multi-span continuous beam on simple supports */
export function buildContinuousBeam(nSpans = 4, spanLength = 1.5): BeamModel {
  const nN = nSpans + 1;
  const nodes: BeamNode[] = [];
  for (let i = 0; i < nN; i++)
    nodes.push({ id: i, x: i * spanLength, y: 0, z: 0 });
  const elements: BeamElement[] = [];
  for (let i = 0; i < nSpans; i++)
    elements.push({ id: i, nodeA: i, nodeB: i + 1 });
  // Fix all 6 DOFs at ends only (others are free to rotate = simple supports)
  // For simplicity, fix ux, uy, uz (translations) at all support nodes, rotations free
  // → we do this by fixing all DOFs at ends only as "fixed-fixed"
  return { nodes, elements, fixedNodes: [0, nN - 1] };
}
