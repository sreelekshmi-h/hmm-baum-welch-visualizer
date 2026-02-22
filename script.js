// Pure JS Discrete HMM + Baum–Welch (EM) with scaling, plus Viterbi decoding.

// ---------- Utilities ----------
function mulberry32(seed) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function clampMin(x, eps) { return x < eps ? eps : x; }

function normalizeVec(v, eps=1e-12) {
  let sum = 0;
  for (let i=0;i<v.length;i++) { v[i] = clampMin(v[i], eps); sum += v[i]; }
  for (let i=0;i<v.length;i++) v[i] /= sum;
  return v;
}

function normalizeRows(mat, eps=1e-12) {
  for (let i=0;i<mat.length;i++) {
    let sum = 0;
    for (let j=0;j<mat[i].length;j++) { mat[i][j] = clampMin(mat[i][j], eps); sum += mat[i][j]; }
    for (let j=0;j<mat[i].length;j++) mat[i][j] /= sum;
  }
  return mat;
}

function zeros2(r,c) {
  const a = new Array(r);
  for (let i=0;i<r;i++) a[i] = new Array(c).fill(0);
  return a;
}
function zeros3(t,r,c) {
  const a = new Array(t);
  for (let k=0;k<t;k++) a[k] = zeros2(r,c);
  return a;
}

function argmax(arr) {
  let bestI = 0, bestV = arr[0];
  for (let i=1;i<arr.length;i++) { if (arr[i] > bestV) { bestV = arr[i]; bestI = i; } }
  return bestI;
}

// ---------- HMM Model ----------
class DiscreteHMM {
  constructor(N, M, seed=0, initMode="random") {
    this.N = N;
    this.M = M;
    this.rng = mulberry32(seed);

    if (initMode === "uniform") {
      this.pi = new Array(N).fill(1/N);
      this.A = zeros2(N,N).map(row => row.map(_ => 1/N));
      this.B = zeros2(N,M).map(row => row.map(_ => 1/M));
    } else {
      this.pi = normalizeVec(new Array(N).fill(0).map(_ => this.rng()));
      this.A = normalizeRows(zeros2(N,N).map(row => row.map(_ => this.rng())));
      this.B = normalizeRows(zeros2(N,M).map(row => row.map(_ => this.rng())));
    }
  }

  forwardScaled(O) {
    const T = O.length;
    const alpha = zeros2(T, this.N);
    const c = new Array(T).fill(0);
    for (let i=0;i<this.N;i++) alpha[0][i] = this.pi[i] * this.B[i][O[0]];
    let s0 = alpha[0].reduce((a,b)=>a+b,0);
    if (s0 === 0) s0 = 1e-300;
    c[0] = s0;
    for (let i=0;i<this.N;i++) alpha[0][i] /= c[0];
    for (let t=1;t<T;t++) {
      for (let j=0;j<this.N;j++) {
        let sum = 0;
        for (let i=0;i<this.N;i++) sum += alpha[t-1][i] * this.A[i][j];
        alpha[t][j] = sum * this.B[j][O[t]];
      }
      let st = alpha[t].reduce((a,b)=>a+b,0);
      if (st === 0) st = 1e-300;
      c[t] = st;
      for (let j=0;j<this.N;j++) alpha[t][j] /= c[t];
    }
    let loglik = 0;
    for (let t=0;t<T;t++) loglik += Math.log(c[t]);
    return {alpha, c, loglik};
  }

  backwardScaled(O, c) {
    const T = O.length;
    const beta = zeros2(T, this.N);
    for (let i=0;i<this.N;i++) beta[T-1][i] = 1;
    for (let t=T-2;t>=0;t--) {
      for (let i=0;i<this.N;i++) {
        let sum = 0;
        for (let j=0;j<this.N;j++) sum += this.A[i][j] * this.B[j][O[t+1]] * beta[t+1][j];
        beta[t][i] = sum / c[t+1];
      }
    }
    return beta;
  }

  baumWelch(O, maxIter=30, tol=1e-4, eps=1e-12) {
    const T = O.length;
    const loglikHist = [];
    let prev = null;

    for (let it=0; it<maxIter; it++) {
      const {alpha, c, loglik} = this.forwardScaled(O);
      const beta = this.backwardScaled(O, c);
      loglikHist.push(loglik);

      const gamma = zeros2(T, this.N);
      for (let t=0;t<T;t++) {
        let sum = 0;
        for (let i=0;i<this.N;i++) { gamma[t][i] = alpha[t][i] * beta[t][i]; sum += gamma[t][i]; }
        sum = clampMin(sum, eps);
        for (let i=0;i<this.N;i++) gamma[t][i] /= sum;
      }

      const xi = zeros3(T-1, this.N, this.N);
      for (let t=0;t<T-1;t++) {
        const obsNext = O[t+1];
        let denom = 0;
        for (let i=0;i<this.N;i++) {
          for (let j=0;j<this.N;j++) {
            const val = alpha[t][i] * this.A[i][j] * this.B[j][obsNext] * beta[t+1][j];
            xi[t][i][j] = val;
            denom += val;
          }
        }
        denom = clampMin(denom, eps);
        for (let i=0;i<this.N;i++) for (let j=0;j<this.N;j++) xi[t][i][j] /= denom;
      }

      const piNew = gamma[0].slice();

      const ANew = zeros2(this.N, this.N);
      const denomA = new Array(this.N).fill(0);
      for (let i=0;i<this.N;i++) {
        let s = 0;
        for (let t=0;t<T-1;t++) s += gamma[t][i];
        denomA[i] = clampMin(s, eps);
      }
      for (let i=0;i<this.N;i++)
        for (let j=0;j<this.N;j++) {
          let num = 0;
          for (let t=0;t<T-1;t++) num += xi[t][i][j];
          ANew[i][j] = num / denomA[i];
        }

      const BNew = zeros2(this.N, this.M);
      const denomB = new Array(this.N).fill(0);
      for (let i=0;i<this.N;i++) {
        let s = 0;
        for (let t=0;t<T;t++) s += gamma[t][i];
        denomB[i] = clampMin(s, eps);
      }
      for (let i=0;i<this.N;i++)
        for (let k=0;k<this.M;k++) {
          let num = 0;
          for (let t=0;t<T;t++) { if (O[t] === k) num += gamma[t][i]; }
          BNew[i][k] = num / denomB[i];
        }

      this.pi = normalizeVec(piNew, eps);
      this.A = normalizeRows(ANew, eps);
      this.B = normalizeRows(BNew, eps);

      if (prev !== null && Math.abs(loglik - prev) < tol) break;
      prev = loglik;
    }

    return {loglikHist};
  }

  viterbi(O) {
    const T = O.length;
    const logA = zeros2(this.N, this.N);
    const logB = zeros2(this.N, this.M);
    const logpi = new Array(this.N).fill(0);
    for (let i=0;i<this.N;i++) logpi[i] = Math.log(clampMin(this.pi[i], 1e-300));
    for (let i=0;i<this.N;i++) {
      for (let j=0;j<this.N;j++) logA[i][j] = Math.log(clampMin(this.A[i][j], 1e-300));
      for (let k=0;k<this.M;k++) logB[i][k] = Math.log(clampMin(this.B[i][k], 1e-300));
    }
    const dp = zeros2(T, this.N);
    const back = zeros2(T, this.N);
    for (let i=0;i<this.N;i++) { dp[0][i] = logpi[i] + logB[i][O[0]]; back[0][i] = 0; }
    for (let t=1;t<T;t++) {
      for (let j=0;j<this.N;j++) {
        let bestScore = -Infinity, bestPrev = 0;
        for (let i=0;i<this.N;i++) {
          const score = dp[t-1][i] + logA[i][j];
          if (score > bestScore) { bestScore = score; bestPrev = i; }
        }
        back[t][j] = bestPrev;
        dp[t][j] = bestScore + logB[j][O[t]];
      }
    }
    const path = new Array(T).fill(0);
    path[T-1] = argmax(dp[T-1]);
    for (let t=T-2;t>=0;t--) path[t] = back[t+1][path[t+1]];
    return path;
  }
}

// ---------- UI Logic ----------
const obsEl = document.getElementById("obs");
const nStatesEl = document.getElementById("nStates");
const nIterEl = document.getElementById("nIter");
const tolEl = document.getElementById("tol");
const seedEl = document.getElementById("seed");
const initModeEl = document.getElementById("initMode");
const trainBtn = document.getElementById("trainBtn");
const statusEl = document.getElementById("status");
const piTableDiv = document.getElementById("piTable");
const aTableDiv = document.getElementById("aTable");
const bTableDiv = document.getElementById("bTable");
const obsOut = document.getElementById("obsOut");
const vitOut = document.getElementById("vitOut");
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");

function buildVocab(symbols) {
  const uniq = Array.from(new Set(symbols)).sort();
  const vocab = new Map();
  uniq.forEach((s,i)=>vocab.set(s,i));
  return {vocab, inv: uniq};
}

function toIntSeq(symbols, vocab) { return symbols.map(s => vocab.get(s)); }

// ---------- Chart ----------
function drawChart(values) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!values || values.length < 2) return;

  const pad = { top: 20, right: 20, bottom: 30, left: 55 };
  const W = canvas.width - pad.left - pad.right;
  const H = canvas.height - pad.top - pad.bottom;

  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = (maxV - minV) || 1;

  // Grid lines
  ctx.strokeStyle = "#dde3ef";
  ctx.lineWidth = 1;
  const gridLines = 4;
  for (let g = 0; g <= gridLines; g++) {
    const y = pad.top + H - (g / gridLines) * H;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + W, y);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "#c4cfe0";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + H);
  ctx.lineTo(pad.left + W, pad.top + H);
  ctx.stroke();

  // Fill area
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad.left + (i/(values.length-1))*W;
    const y = pad.top + H - ((v - minV)/range)*H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  const lastX = pad.left + W;
  ctx.lineTo(lastX, pad.top + H);
  ctx.lineTo(pad.left, pad.top + H);
  ctx.closePath();
  ctx.fillStyle = "rgba(37,99,235,0.08)";
  ctx.fill();

  // Line
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2.5;
  ctx.lineJoin = "round";
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad.left + (i/(values.length-1))*W;
    const y = pad.top + H - ((v - minV)/range)*H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots at first and last
  [0, values.length-1].forEach(i => {
    const x = pad.left + (i/(values.length-1))*W;
    const y = pad.top + H - ((values[i] - minV)/range)*H;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI*2);
    ctx.fillStyle = "#2563eb";
    ctx.fill();
  });

  // Labels
  ctx.fillStyle = "#6b7a99";
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxV.toFixed(1), pad.left - 6, pad.top + 4);
  ctx.fillText(minV.toFixed(1), pad.left - 6, pad.top + H + 4);
  ctx.textAlign = "center";
  ctx.fillText("1", pad.left, pad.top + H + 16);
  ctx.fillText(values.length, pad.left + W, pad.top + H + 16);
}

// ---------- Tables ----------
function toTable(mat, colLabels=null, rowLabels=null, digits=4) {
  const rows = mat.length, cols = mat[0].length;
  let html = `<table><thead><tr><th></th>`;
  for (let c=0;c<cols;c++) html += `<th>${colLabels ? colLabels[c] : 'S'+c}</th>`;
  html += `</tr></thead><tbody>`;
  for (let r=0;r<rows;r++) {
    html += `<tr><th>${rowLabels ? rowLabels[r] : 'S'+r}</th>`;
    for (let c=0;c<cols;c++) {
      const v = mat[r][c];
      const intensity = Math.round(v * 80);
      html += `<td style="background:rgba(37,99,235,${v*0.18})">${v.toFixed(digits)}</td>`;
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  return html;
}

function vecToTable(vec, digits=4) {
  let html = `<table><thead><tr>`;
  for (let i=0;i<vec.length;i++) html += `<th>S${i}</th>`;
  html += `</tr></thead><tbody><tr>`;
  for (let i=0;i<vec.length;i++) html += `<td style="background:rgba(37,99,235,${vec[i]*0.18})">${vec[i].toFixed(digits)}</td>`;
  html += `</tr></tbody></table>`;
  return html;
}

// ---------- HMM Diagram ----------
const STATE_COLORS = [
  "#e97c2f", "#3b82f6", "#10b981", "#8b5cf6", "#ef4444", "#f59e0b",
];

function renderHMMDiagram(hmm, symbolNames) {
  const container = document.getElementById("diagram");
  container.innerHTML = "";

  const N = hmm.N;
  const M = hmm.M;
  const A = hmm.A;
  const B = hmm.B;
  const pi = hmm.pi;

  // Adaptive layout — node size and spacing scale with N so diagram never crowds
  const NODE_R     = Math.max(14, Math.min(24, Math.floor(180 / (N + 1))));
  const LOOP_H     = NODE_R * 2 + 8;
  const LABEL_PAD  = 3;

  const MARGIN_LEFT  = 60;
  const MARGIN_RIGHT = 40;

  // Ensure nodes are never closer than 3.6 radii apart (room for labels)
  const minSpacing   = NODE_R * 6.0;
  const BASE_USABLE  = 1100;
  const neededUsable = N > 1 ? (N - 1) * minSpacing : minSpacing;
  const effectiveUsable = Math.max(BASE_USABLE, neededUsable);
  const W = MARGIN_LEFT + effectiveUsable + MARGIN_RIGHT;

  const stateSpacing = N > 1 ? effectiveUsable / (N - 1) : 0;
  const statePosX = Array.from({length: N}, (_, i) =>
    N === 1 ? W / 2 : MARGIN_LEFT + i * stateSpacing);

  // Obs nodes span same horizontal range as states
  const obsUsable  = N > 1 ? (N - 1) * stateSpacing : effectiveUsable;
  const obsSpacing = M > 1 ? obsUsable / (M - 1) : 0;
  const obsPosX    = Array.from({length: M}, (_, k) =>
    M === 1 ? W / 2 : MARGIN_LEFT + k * obsSpacing);

  // Row Y centres — all derived from NODE_R
  const ROW_START  = 50;
  const ROW_LOOP   = ROW_START + 70 + LOOP_H;
  const ROW_STATE  = ROW_LOOP + NODE_R + 40;
  const ROW_EMIT_L = ROW_STATE + NODE_R + 50;
  const ROW_OBS    = ROW_EMIT_L + 80;
  const H          = ROW_OBS + 50;

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("style", "display:block;overflow:visible");

  // ---- Arrow markers ----
  const defs = document.createElementNS(svgNS, "defs");
  const markerDefs = [
    { id: "arr-pi",    color: "#7c3aed" },
    ...STATE_COLORS.map((c, i) => ({ id: `arr-t${i}`,  color: c })),
    ...STATE_COLORS.map((c, i) => ({ id: `arr-e${i}`,  color: c + "bb" })),
  ];
  defs.innerHTML = markerDefs.map(m => `
    <marker id="${m.id}" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,1 L10,5 L0,9 Z" fill="${m.color}"/>
    </marker>`).join('');
  svg.appendChild(defs);

  // ---- Helpers ----
  function el(tag, attrs) {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  // Draw text with a white pill behind it so it's always readable
  function labelPill(x, y, str, color, size=11) {
    const g = document.createElementNS(svgNS, "g");
    // measure approx width: ~6.5px per char at size 11
    const charW = size * 0.62;
    const tw = str.length * charW;
    const pw = tw + LABEL_PAD * 2 + 4;
    const ph = size + LABEL_PAD * 2;
    const pill = el("rect", {
      x: x - pw/2, y: y - ph + 2,
      width: pw, height: ph,
      rx: 4, ry: 4,
      fill: "white",
      stroke: color + "44",
      "stroke-width": 1
    });
    const txt = el("text", {
      x, y,
      fill: color,
      "font-size": size,
      "font-family": "'IBM Plex Mono', monospace",
      "text-anchor": "middle",
      "font-weight": "600"
    });
    txt.textContent = str;
    g.appendChild(pill);
    g.appendChild(txt);
    svg.appendChild(g);
  }

  function rowLabel(y, str) {
    const t = el("text", {
      x: 8, y: y + 4,
      fill: "#b0bcd4",
      "font-size": 10,
      "font-family": "'IBM Plex Mono', monospace",
      "text-anchor": "start",
      "letter-spacing": "0.1em"
    });
    t.textContent = str;
    svg.appendChild(t);
  }

  // ---- Background row bands ----
  [ [ROW_START - 22, 44, "#f3f0ff22", "#7c3aed22"],
    [ROW_STATE - NODE_R - 10, NODE_R*2+20, "#f8fafd", "#dde3ef"],
    [ROW_OBS - 24, 48, "#f8fafd", "#dde3ef"],
  ].forEach(([y, h, fill, stroke]) => {
    svg.appendChild(el("rect", {
      x: MARGIN_LEFT - 10, y,
      width: effectiveUsable + 20, height: h,
      rx: 10, fill, stroke, "stroke-width": 1
    }));
  });

  rowLabel(ROW_START, "START");
  rowLabel(ROW_STATE, "HIDDEN");
  rowLabel(ROW_OBS,   "OBSERVE");

  // ---- START node ----
  const startX = W / 2;
  const startY = ROW_START;
  svg.appendChild(el("rect", {
    x: startX - 40, y: startY - 16,
    width: 80, height: 32,
    rx: 16,
    fill: "#ede9fe",
    stroke: "#7c3aed",
    "stroke-width": 2
  }));
  const stTxt = el("text", {
    x: startX, y: startY + 6,
    fill: "#7c3aed",
    "font-size": 13,
    "font-family": "'IBM Plex Mono', monospace",
    "text-anchor": "middle",
    "font-weight": "700"
  });
  stTxt.textContent = "START";
  svg.appendChild(stTxt);

  // ---- π arrows: START → states ----
  for (let i = 0; i < N; i++) {
    if (pi[i] < 0.005) continue;
    const x2 = statePosX[i];
    const y1 = startY + 16;
    const y2 = ROW_STATE - NODE_R;
    // Control point: bias horizontally toward target
    const cx = (startX * 0.3 + x2 * 0.7);
    const cy = (y1 + y2) / 2;
    const d = `M ${startX} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
    svg.appendChild(el("path", {
      d, fill: "none",
      stroke: "#7c3aed",
      "stroke-width": 1.6,
      "stroke-dasharray": "5 3",
      "marker-end": "url(#arr-pi)"
    }));
    // Label: place at 40% along the curve, offset left of the line
    const t = 0.4;
    const lx = (1-t)*(1-t)*startX + 2*(1-t)*t*cx + t*t*x2;
    const ly = (1-t)*(1-t)*y1     + 2*(1-t)*t*cy + t*t*y2;
    // perpendicular nudge to the left of travel direction
    const dx = x2 - startX, dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const perpX = -dy/len * 18;  // left perpendicular
    const perpY =  dx/len * 18;
    labelPill(lx + perpX, ly + perpY + 6, `π=${pi[i].toFixed(2)}`, "#7c3aed", 10);
  }

  // ---- Self-loops ----
  for (let i = 0; i < N; i++) {
    const p = A[i][i];
    if (p < 0.005) continue;
    const color = STATE_COLORS[i % STATE_COLORS.length];
    const cx = statePosX[i];
    const cy = ROW_STATE;
    // Arc: departs top-left of circle, peaks at ROW_LOOP, returns top-right
    const d = `M ${cx - NODE_R*0.7} ${cy - NODE_R*0.72}
               C ${cx - NODE_R*2.2} ${cy - NODE_R - LOOP_H},
                 ${cx + NODE_R*2.2} ${cy - NODE_R - LOOP_H},
                 ${cx + NODE_R*0.7} ${cy - NODE_R*0.72}`;
    svg.appendChild(el("path", {
      d, fill: "none",
      stroke: color + "cc",
      "stroke-width": 1.8,
      "marker-end": `url(#arr-t${i})`
    }));
    // Label at apex — always above the arc
    labelPill(cx, cy - NODE_R - LOOP_H - 10, p.toFixed(2), color, 11);
  }

  // ---- Transition arrows (i ≠ j) ----
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const p = A[i][j];
      if (p < 0.005) continue;

      const color = STATE_COLORS[i % STATE_COLORS.length];
      const x1 = statePosX[i];
      const x2 = statePosX[j];
      const y  = ROW_STATE;

      // Offset the arrow edge to the circle boundary
      const angle = Math.atan2(0, x2 - x1);
      const sx = x1 + Math.cos(angle) * NODE_R;
      const ex = x2 - Math.cos(angle) * NODE_R;

      // Curve above (i<j) or below (i>j) to separate bidirectional arrows
      const bulge = i < j ? -40 : 40;
      const mx = (sx + ex) / 2;
      const my = y + bulge;

      const d = `M ${sx} ${y} Q ${mx} ${my} ${ex} ${y}`;
      svg.appendChild(el("path", {
        d, fill: "none",
        stroke: color + "aa",
        "stroke-width": 1.7,
        "marker-end": `url(#arr-t${i})`
      }));

      // Label at bezier midpoint, pushed away from the curve
      const t = 0.5;
      const qx = (1-t)*(1-t)*sx + 2*(1-t)*t*mx + t*t*ex;
      const qy = (1-t)*(1-t)*y  + 2*(1-t)*t*my + t*t*y;
      // Offset label further from curve apex
      const labelOffY = bulge < 0 ? -14 : 14;
      labelPill(qx, qy + labelOffY, p.toFixed(2), color, 11);
    }
  }

  // ---- Emission arrows (state → obs box), dashed ----
  for (let i = 0; i < N; i++) {
    const color = STATE_COLORS[i % STATE_COLORS.length];
    for (let k = 0; k < M; k++) {
      const p = B[i][k];
      if (p < 0.02) continue;

      const x1 = statePosX[i];
      const y1 = ROW_STATE + NODE_R;
      const x2 = obsPosX[k];
      const y2 = ROW_OBS - 20;

      // Straight-ish path with slight S-curve
      const mx = (x1 + x2) / 2;
      const my = ROW_EMIT_L;
      const d = `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
      svg.appendChild(el("path", {
        d, fill: "none",
        stroke: color + "66",
        "stroke-width": 1.4,
        "stroke-dasharray": "5 3",
        "marker-end": `url(#arr-e${i})`
      }));

      // Label at ~40% along curve, nudged sideways
      const t2 = 0.42;
      const lx2 = (1-t2)*(1-t2)*x1 + 2*(1-t2)*t2*mx + t2*t2*x2;
      const ly2 = (1-t2)*(1-t2)*y1 + 2*(1-t2)*t2*my + t2*t2*y2;
      // push label to the side of the arrow
      const sideX = x2 >= x1 ? 16 : -16;
      labelPill(lx2 + sideX, ly2, p.toFixed(2), color, 10);
    }
  }

  // ---- Draw state nodes (on top so they cover arrow ends) ----
  for (let i = 0; i < N; i++) {
    const x = statePosX[i];
    const y = ROW_STATE;
    const color = STATE_COLORS[i % STATE_COLORS.length];

    // Drop shadow
    svg.appendChild(el("circle", {
      cx: x+2, cy: y+3, r: NODE_R,
      fill: "rgba(0,0,0,0.08)"
    }));
    // Circle
    svg.appendChild(el("circle", {
      cx: x, cy: y, r: NODE_R,
      fill: color,
      stroke: "#fff",
      "stroke-width": 3
    }));
    // State label
    const lbl = el("text", {
      x, y: y + 6,
      fill: "white",
      "font-size": Math.max(10, NODE_R * 0.48),
      "font-family": "'DM Sans', sans-serif",
      "text-anchor": "middle",
      "font-weight": "700"
    });
    lbl.textContent = `S${i}`;
    svg.appendChild(lbl);

    // Best-emission hint below node
    if (symbolNames && symbolNames.length > 0) {
      let bestK = 0;
      for (let k = 1; k < M; k++) if (B[i][k] > B[i][bestK]) bestK = k;
      const hint = el("text", {
        x, y: y + NODE_R + 16,
        fill: color,
        "font-size": 10,
        "font-family": "'IBM Plex Mono', monospace",
        "text-anchor": "middle"
      });
      hint.textContent = `↑${symbolNames[bestK]}(${B[i][bestK].toFixed(2)})`;
      svg.appendChild(hint);
    }
  }

  // ---- Observation boxes ----
  const OBS_W = 64, OBS_H = 34;
  for (let k = 0; k < M; k++) {
    const x = obsPosX[k];
    const y = ROW_OBS;
    const label = symbolNames ? symbolNames[k] : `O${k}`;

    svg.appendChild(el("rect", {
      x: x - OBS_W/2 + 2, y: y - OBS_H/2 + 3,
      width: OBS_W, height: OBS_H, rx: 8,
      fill: "rgba(0,0,0,0.05)"
    }));
    svg.appendChild(el("rect", {
      x: x - OBS_W/2, y: y - OBS_H/2,
      width: OBS_W, height: OBS_H, rx: 8,
      fill: "#fff",
      stroke: "#c4cfe0",
      "stroke-width": 1.8
    }));
    const lbl = el("text", {
      x, y: y + 5,
      fill: "#334155",
      "font-size": 13,
      "font-family": "'IBM Plex Mono', monospace",
      "text-anchor": "middle",
      "font-weight": "600"
    });
    lbl.textContent = label;
    svg.appendChild(lbl);
  }

  container.appendChild(svg);
}

// ---------- Train ----------
trainBtn.addEventListener("click", () => {
  statusEl.textContent = "Training…";
  trainBtn.disabled = true;

  setTimeout(() => {
    const raw = obsEl.value.trim();
    const symbols = raw.split(/\s+/).filter(Boolean);
    if (symbols.length === 0) {
      statusEl.textContent = "⚠ Please enter an observation sequence.";
      trainBtn.disabled = false;
      return;
    }

    const N = Number(nStatesEl.value);
    const maxIter = Number(nIterEl.value);
    const tol = Number(tolEl.value);
    const seed = Number(seedEl.value);
    const initMode = initModeEl.value;

    const {vocab, inv} = buildVocab(symbols);
    const O = toIntSeq(symbols, vocab);
    const M = inv.length;

    const hmm = new DiscreteHMM(N, M, seed, initMode);
    const {loglikHist} = hmm.baumWelch(O, maxIter, tol);
    const path = hmm.viterbi(O);

    piTableDiv.innerHTML = vecToTable(hmm.pi);
    aTableDiv.innerHTML = toTable(hmm.A);
    bTableDiv.innerHTML = toTable(hmm.B, inv, Array.from({length:N},(_,i)=>`S${i}`));

    obsOut.textContent = JSON.stringify({
      symbols,
      vocab: Object.fromEntries(vocab)
    }, null, 2);
    vitOut.textContent = JSON.stringify({
      path,
      sequence: path.map(s => `S${s}`)
    }, null, 2);

    drawChart(loglikHist);
    renderHMMDiagram(hmm, inv);
    statusEl.textContent = `✅ Done — symbols: ${M}, length: ${O.length}, iters: ${loglikHist.length}`;
    trainBtn.disabled = false;
  }, 10);
});