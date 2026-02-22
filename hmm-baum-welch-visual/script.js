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

// ---------- HMM Diagram (reference-image style) ----------
const STATE_COLORS = [
  "#e97c2f",  // orange
  "#3b82f6",  // blue
  "#10b981",  // green
  "#8b5cf6",  // purple
  "#ef4444",  // red
  "#f59e0b",  // amber
];

function renderHMMDiagram(hmm, symbolNames) {
  const container = document.getElementById("diagram");
  container.innerHTML = "";

  const N = hmm.N;
  const M = hmm.M;
  const A = hmm.A;
  const B = hmm.B;
  const pi = hmm.pi;

  const W = 920;
  const ROW_START = 80;    // START row y
  const ROW_STATE = 240;   // hidden states row y
  const ROW_OBS = 420;     // observations row y
  const H = 530;

  const NODE_R = 36;
  const OBS_W = 70, OBS_H = 38;

  // X positions for states
  const stateSpacing = Math.min(180, (W - 80) / N);
  const stateStartX = W/2 - ((N-1) * stateSpacing) / 2;
  const statePosX = Array.from({length: N}, (_, i) => stateStartX + i * stateSpacing);

  // X positions for obs symbols
  const obsSpacing = Math.min(160, (W - 80) / M);
  const obsStartX = W/2 - ((M-1) * obsSpacing) / 2;
  const obsPosX = Array.from({length: M}, (_, k) => obsStartX + k * obsSpacing);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Defs: markers for solid and dashed arrows
  const defs = document.createElementNS(svgNS, "defs");
  defs.innerHTML = `
    <marker id="arr-solid" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/>
    </marker>
    <marker id="arr-start" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/>
    </marker>
    ${STATE_COLORS.map((col, i) => `
    <marker id="arr-state-${i}" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="${col}88"/>
    </marker>
    <marker id="arr-emit-${i}" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="${col}99"/>
    </marker>
    `).join('')}
  `;
  svg.appendChild(defs);

  // Helper: create SVG element with attrs
  function el(tag, attrs={}) {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function text(x, y, str, opts={}) {
    const t = el("text", {
      x, y,
      fill: opts.fill || "#475569",
      "font-size": opts.size || 12,
      "font-family": opts.mono ? "'IBM Plex Mono', monospace" : "'DM Sans', sans-serif",
      "text-anchor": opts.anchor || "middle",
      "font-weight": opts.bold ? "600" : "400",
      ...opts.extra
    });
    t.textContent = str;
    svg.appendChild(t);
    return t;
  }

  // Row labels
  function rowLabel(y, label) {
    const t = el("text", {
      x: 12, y: y + 5,
      fill: "#94a3b8",
      "font-size": 11,
      "font-family": "'IBM Plex Mono', monospace",
      "text-anchor": "start",
      "letter-spacing": "0.08em"
    });
    t.textContent = label;
    svg.appendChild(t);
  }

  rowLabel(ROW_START, "START");
  rowLabel(ROW_STATE, "HIDDEN");
  rowLabel(ROW_OBS, "OBSERVE");

  // Row separator lines
  [ROW_START + 55, ROW_STATE + 65].forEach(y => {
    svg.appendChild(el("line", {
      x1: 70, y1: y, x2: W - 10, y2: y,
      stroke: "#e2e8f0", "stroke-width": 1, "stroke-dasharray": "4 4"
    }));
  });

  // ---- START node ----
  const startX = W / 2;
  const startY = ROW_START;

  const startRect = el("rect", {
    x: startX - 38, y: startY - 18,
    width: 76, height: 36,
    rx: 18, ry: 18,
    fill: "#f3f0ff",
    stroke: "#7c3aed",
    "stroke-width": 2
  });
  svg.appendChild(startRect);
  text(startX, startY + 5, "START", { fill: "#7c3aed", size: 13, bold: true, mono: true });

  // ---- START → each state (π arrows) ----
  for (let i = 0; i < N; i++) {
    if (pi[i] < 0.01) continue;
    const x2 = statePosX[i];
    const y1 = startY + 18;
    const y2 = ROW_STATE - NODE_R;

    // Curved path from START to state
    const midX = (startX + x2) / 2;
    const midY = (y1 + y2) / 2;
    const d = `M ${startX} ${y1} Q ${midX} ${midY} ${x2} ${y2}`;
    svg.appendChild(el("path", {
      d, fill: "none",
      stroke: "#7c3aed",
      "stroke-width": 1.8,
      "stroke-dasharray": "6 3",
      "marker-end": "url(#arr-start)",
      opacity: 0.8
    }));

    // Label π near midpoint
    const lx = midX + (x2 < startX ? -12 : 12);
    const ly = midY - 6;
    text(lx, ly, `π=${pi[i].toFixed(2)}`, { fill: "#7c3aed", size: 11, mono: true, anchor: "middle" });
  }

  // ---- Transition arrows (state → state) ----
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const p = A[i][j];
      if (p < 0.01) continue;

      const color = STATE_COLORS[i % STATE_COLORS.length];
      const x1 = statePosX[i];
      const x2 = statePosX[j];
      const y = ROW_STATE;

      if (i === j) {
        // Self-loop above state
        const loopR = NODE_R + 16;
        const d = `M ${x1 - 14} ${y - NODE_R + 4}
                   C ${x1 - loopR} ${y - NODE_R - 44},
                     ${x1 + loopR} ${y - NODE_R - 44},
                     ${x1 + 14} ${y - NODE_R + 4}`;
        svg.appendChild(el("path", {
          d, fill: "none",
          stroke: color + "bb",
          "stroke-width": 1.8,
          "marker-end": `url(#arr-state-${i})`,
        }));
        text(x1, y - NODE_R - 50, p.toFixed(2), { fill: color, size: 11, mono: true });
      } else {
        // Curved arc between states
        const dx = x2 - x1;
        const sign = i < j ? -1 : 1;
        const curve = 50 * sign;
        const mx = (x1 + x2) / 2;
        const my = y + curve;

        const ux = x2 - x1, uy = 0;
        const dist = Math.abs(ux);
        const normX = ux / dist;
        const sx = x1 + normX * NODE_R;
        const ex = x2 - normX * NODE_R;

        const d = `M ${sx} ${y} Q ${mx} ${my} ${ex} ${y}`;
        svg.appendChild(el("path", {
          d, fill: "none",
          stroke: color + "99",
          "stroke-width": 1.6,
          "marker-end": `url(#arr-state-${i})`,
        }));

        // Label near curve midpoint
        const t = 0.5;
        const qx = (1-t)*(1-t)*sx + 2*(1-t)*t*mx + t*t*ex;
        const qy = (1-t)*(1-t)*y + 2*(1-t)*t*my + t*t*y;
        text(qx, qy + (sign < 0 ? -7 : 15), p.toFixed(2), { fill: color, size: 11, mono: true });
      }
    }
  }

  // ---- Emission arrows (state → obs) dashed ----
  for (let i = 0; i < N; i++) {
    const color = STATE_COLORS[i % STATE_COLORS.length];
    for (let k = 0; k < M; k++) {
      const p = B[i][k];
      if (p < 0.02) continue;

      const x1 = statePosX[i];
      const y1 = ROW_STATE + NODE_R;
      const x2 = obsPosX[k];
      const y2 = ROW_OBS - OBS_H/2;

      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2 + 20;

      const d = `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
      svg.appendChild(el("path", {
        d, fill: "none",
        stroke: color + "77",
        "stroke-width": 1.4,
        "stroke-dasharray": "5 3",
        "marker-end": `url(#arr-emit-${i})`,
        opacity: 0.85
      }));

      // Label near midpoint
      const t = 0.5;
      const lx = (1-t)*(1-t)*x1 + 2*(1-t)*t*mx + t*t*x2;
      const ly = (1-t)*(1-t)*y1 + 2*(1-t)*t*my + t*t*y2;
      text(lx + 4, ly, p.toFixed(2), { fill: color + "cc", size: 10, mono: true });
    }
  }

  // ---- State nodes ----
  for (let i = 0; i < N; i++) {
    const x = statePosX[i];
    const y = ROW_STATE;
    const color = STATE_COLORS[i % STATE_COLORS.length];

    // Shadow
    svg.appendChild(el("circle", {
      cx: x + 2, cy: y + 3, r: NODE_R,
      fill: "rgba(0,0,0,0.06)"
    }));

    // Node
    svg.appendChild(el("circle", {
      cx: x, cy: y, r: NODE_R,
      fill: color,
      stroke: "#fff",
      "stroke-width": 2.5
    }));

    // Label
    const t = el("text", {
      x, y: y + 6,
      fill: "white",
      "font-size": 15,
      "font-family": "'DM Sans', sans-serif",
      "text-anchor": "middle",
      "font-weight": "600"
    });
    t.textContent = `S${i}`;
    svg.appendChild(t);

    // Top emission symbol below
    if (symbolNames && symbolNames.length > 0) {
      const probs = B[i];
      let bestK = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[bestK]) bestK = k;
      text(x, y + NODE_R + 18,
        `↑ ${symbolNames[bestK]} (${probs[bestK].toFixed(2)})`,
        { fill: color, size: 11, mono: true });
    }
  }

  // ---- Observation nodes ----
  for (let k = 0; k < M; k++) {
    const x = obsPosX[k];
    const y = ROW_OBS;
    const label = symbolNames ? symbolNames[k] : `O${k}`;

    // Shadow
    svg.appendChild(el("rect", {
      x: x - OBS_W/2 + 2, y: y - OBS_H/2 + 3,
      width: OBS_W, height: OBS_H,
      rx: 10,
      fill: "rgba(0,0,0,0.05)"
    }));

    // Box
    svg.appendChild(el("rect", {
      x: x - OBS_W/2, y: y - OBS_H/2,
      width: OBS_W, height: OBS_H,
      rx: 10,
      fill: "#fff",
      stroke: "#c4cfe0",
      "stroke-width": 1.8
    }));

    const t = el("text", {
      x, y: y + 5,
      fill: "#334155",
      "font-size": 14,
      "font-family": "'IBM Plex Mono', monospace",
      "text-anchor": "middle",
      "font-weight": "600"
    });
    t.textContent = label;
    svg.appendChild(t);
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