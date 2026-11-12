// pumpfun-top5-forecast-01 entry: Monte Carlo rank forecaster.
//
// Idea: pump.fun's newest coins churn violently. In the practice history the
// top 5 changes almost completely every 10 minutes (1-4 of the next top 5 are
// coins that did not exist in the previous mark), the #1 coin's cap roughly
// triples in 40 minutes, and at the last practice read 3 of the top 5 were
// coins the history never named. Pure persistence (the house baseline) is
// therefore far too confident at a 7-13 hour horizon. This entry instead:
//
//  - builds every coin's log10-cap trajectory from the history marks,
//  - estimates each rank band's per-10-minute drift and spread from the
//    history's own consecutive-mark transitions (shrunk toward priors),
//  - simulates each candidate coin's log-cap at every answer read as a
//    two-state mixture: with a per-step survival probability that grows
//    with the coin's size and age it holds near its peak (rising slowly),
//    otherwise it fades to the ~28 SOL floor every dumped coin ends at.
//    A symmetric random walk is wrong here: it gives ~250 coins a fat
//    upper tail, so their max draw crowds out the newborns that in reality
//    take most top places.
//  - adds a newborn process: coins born after the history cut, drawn at the
//    rate and sizes at which new coins actually entered the top 5,
//  - ranks the simulated caps, and averages the place probabilities and the
//    per-place size histograms over the seven reads (the one forecast is
//    scored at every read, so the mixture is the honest single forecast).
//
// Node 22, standard library only, deterministic (fixed-seed RNG, no clock,
// no network, no Math.random). Reads $MAKEFUN_INPUT_DIR else /input, writes
// $MAKEFUN_OUTPUT_DIR else /output.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// ---------- deterministic RNG (mulberry32 + Box-Muller) ----------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeRandn(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0; let v = 0; let s = 0;
    do { u = rng() * 2 - 1; v = rng() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m;
    return u * m;
  };
}
function poisson(rng: () => number, lambda: number): number {
  if (lambda <= 0) return 0;
  const l = Math.exp(-lambda);
  let k = 0; let p = 1;
  do { k++; p *= rng(); } while (p > l);
  return k - 1;
}

// ---------- input ----------
interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue?: string;
  mayhem?: boolean;
  created_at?: string;
}
interface Mark { at: string; ok: boolean; rows: Row[] | null; }

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  t0: string; history_until: string; places: number; cadence_minutes: number;
  reads: string[]; size: { edges: number[] }; output: string;
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok === true && Array.isArray(m.rows) && m.rows.length > 0);
if (marks.length === 0) throw new Error('the history holds no ok mark with rows');
const PLACES = task.places;
const EDGES = task.size.edges;
const CADENCE = task.cadence_minutes || 10;
const t0 = Date.parse(task.t0);
const lastMark = marks[marks.length - 1]!;
const lastTime = Date.parse(lastMark.at);
const lastRows = [...lastMark.rows!].sort((a, b) => a.rank - b.rank);
if (lastRows.length < PLACES) throw new Error(`the last mark ranks ${lastRows.length} coins, fewer than the ${PLACES} places`);
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

// ---------- trajectories ----------
interface Pt { t: number; logCap: number; rank: number }
const traj = new Map<string, Pt[]>();
const firstSeen = new Map<string, number>();
const everTop20 = new Set<string>();
for (const m of marks) {
  const t = (Date.parse(m.at) - t0) / 60000;
  for (const r of m.rows!) {
    const pt: Pt = { t, logCap: sizeOf(r.market_cap_lamports), rank: r.rank };
    const s = traj.get(r.mint);
    if (s) s.push(pt); else traj.set(r.mint, [pt]);
    if (!firstSeen.has(r.mint)) firstSeen.set(r.mint, t);
    if (r.rank <= 20) everTop20.add(r.mint);
  }
}

// ---------- band dynamics from consecutive-mark transitions ----------
type Band = 'top5' | '6-20' | '21-100' | '100+';
const bandOf = (rank: number): Band => (rank <= 5 ? 'top5' : rank <= 20 ? '6-20' : rank <= 100 ? '21-100' : '100+');
const DRIFT_PRIOR: Record<Band, number> = { 'top5': -0.08, '6-20': -0.04, '21-100': 0, '100+': 0 };
const SD_PRIOR: Record<Band, number> = { 'top5': 0.30, '6-20': 0.25, '21-100': 0.08, '100+': 0.05 };
const dlog: Record<Band, number[]> = { 'top5': [], '6-20': [], '21-100': [], '100+': [] };
for (const s of traj.values()) {
  const sorted = [...s].sort((a, b) => a.t - b.t);
  for (let i = 0; i + 1 < sorted.length; i++) {
    const a = sorted[i]!; const b = sorted[i + 1]!;
    if (Math.abs((b.t - a.t) - CADENCE) < 0.01) dlog[bandOf(a.rank)].push(b.logCap - a.logCap);
  }
}
const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
const sd = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
};
const bandDrift: Record<Band, number> = {} as Record<Band, number>;
const bandSd: Record<Band, number> = {} as Record<Band, number>;
for (const b of ['top5', '6-20', '21-100', '100+'] as Band[]) {
  const n = dlog[b].length;
  const w = n / (n + 10); // shrink empirical toward prior when few transitions
  bandDrift[b] = w * mean(dlog[b]) + (1 - w) * DRIFT_PRIOR[b];
  bandSd[b] = Math.max(0.03, w * sd(dlog[b]) + (1 - w) * SD_PRIOR[b]);
}

// ---------- newborn process: coins that entered the top 5 brand new ----------
// first-appearance log-caps of coins whose first row was in the top 5
const newbornCaps: number[] = [];
let newTop5PerStep = 0; let nSteps = 0;
{
  const markTop5: string[][] = marks.map((m) => [...m.rows!].sort((a, b) => a.rank - b.rank).slice(0, 5).map((r) => r.mint));
  const markTop20: Set<string>[] = marks.map((m) => new Set([...m.rows!].sort((a, b) => a.rank - b.rank).slice(0, 20).map((r) => r.mint)));
  for (let i = 1; i < marks.length; i++) {
    const fresh = markTop5[i]!.filter((mint) => !markTop20[i - 1]!.has(mint));
    newTop5PerStep += fresh.length; nSteps++;
  }
  for (const [mint, s] of traj) {
    const sorted = [...s].sort((a, b) => a.t - b.t);
    if (sorted[0]!.rank <= 5) newbornCaps.push(sorted[0]!.logCap);
  }
}
const newbornMean = newbornCaps.length >= 3 ? mean(newbornCaps) : 3.0;
const newbornSd = newbornCaps.length >= 3 ? Math.max(0.35, sd(newbornCaps)) : 0.35;
// per read, expect about as many brand-new top-5 coins as one step produces;
// the practice's last read held 3 newborns in its top 5, so floor at 3.
// Several newborns compete for the top places at a read, so draw a few
// extra contenders and let the ranking pick the best of them.
const lambdaNew = Math.min(6, Math.max(5.0, nSteps > 0 ? newTop5PerStep / nSteps + 2 : 5.0));

// ---------- candidates ----------
// A coin's future is bimodal: it either holds near its peak or dumps to the
// floor (~28 SOL, log10 1.45, where ranks 80-200 all sit). The per-step
// survival probability grows with the coin's size (big coins persist: Claude
// and DeepSeek held top 5 for 40+ minutes; 70-130 SOL coins crashed to rank
// 60+ in one step) and with its age in marks (a coin that just appeared is
// a flash pump until proven otherwise).
interface Cand { mint: string; c0: number; surviveStep: number; ageMarks: number }
const candSet = new Set<string>(lastRows.map((r) => r.mint));
for (const mint of everTop20) candSet.add(mint);
const candidates: Cand[] = [];
for (const mint of candSet) {
  const s = traj.get(mint);
  if (!s || s.length === 0) continue;
  const sorted = [...s].sort((a, b) => a.t - b.t);
  const last = sorted[sorted.length - 1]!;
  // only coins seen in the last two marks can still matter
  if (last.t < (Date.parse(lastMark.at) - t0) / 60000 - CADENCE * 1.5) continue;
  const ageMarks = sorted.length;
  const ageFactor = 0.7 + 0.3 * Math.min(ageMarks, 5) / 5;
  const sizeFactor = Math.min(0.92, Math.max(0.15, 0.55 + 0.35 * (last.logCap - 2.0)));
  candidates.push({ mint, c0: last.logCap, surviveStep: sizeFactor * ageFactor, ageMarks });
}
const FADE_LEVEL = 1.5; // log10 cap in SOL of a dumped coin (~28-30 SOL)
const HOLD_RISE = 0.03; // a survivor's log-cap rises ~0.03 per 10-min step
const HOLD_SD_STEP = 0.12;
const HOLD_SD_CAP = 0.35;
const FADE_SD = 0.25;

// ---------- Monte Carlo over the reads ----------
const NSIMS = 8000;
const rng = mulberry32(0x9e3779b9);
const randn = makeRandn(rng);
const reads = task.reads.length > 0 ? task.reads : [task.history_until];
const placeProb: Map<string, number>[] = Array.from({ length: PLACES }, () => new Map<string, number>());
const sizeHist: number[][] = Array.from({ length: PLACES }, () => new Array<number>(EDGES.length + 1).fill(0));
for (const readAt of reads) {
  const hMin = Math.max(0, (Date.parse(readAt) - lastTime) / 60000);
  const steps = Math.max(0, hMin / CADENCE);
  // precompute per-candidate hold probability / hold distribution at this read
  const holdP: number[] = [];
  const holdMu: number[] = [];
  const holdSd: number[] = [];
  for (const c of candidates) {
    holdP.push(Math.pow(c.surviveStep, steps));
    holdMu.push(c.c0 + HOLD_RISE * Math.min(steps, 10));
    holdSd.push(Math.min(HOLD_SD_STEP * Math.sqrt(Math.max(steps, 0.25)), HOLD_SD_CAP));
  }
  for (let sim = 0; sim < NSIMS; sim++) {
    // top-5 by simulated log-cap: linear scan keeping the best five
    const topMint: (string | null)[] = [null, null, null, null, null];
    const topCap: number[] = [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity];
    const consider = (mint: string, cap: number) => {
      for (let k = 0; k < PLACES; k++) {
        if (cap > topCap[k]!) {
          for (let j = PLACES - 1; j > k; j--) { topCap[j] = topCap[j - 1]!; topMint[j] = topMint[j - 1]; }
          topCap[k] = cap; topMint[k] = mint;
          return;
        }
      }
    };
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i]!;
      const cap = rng() < holdP[i]!
        ? holdMu[i]! + holdSd[i]! * randn()
        : FADE_LEVEL + FADE_SD * randn();
      consider(c.mint, cap);
    }
    const nNew = poisson(rng, lambdaNew);
    for (let j = 0; j < nNew; j++) {
      consider(`__newborn_${sim}_${j}`, newbornMean + newbornSd * randn());
    }
    for (let k = 0; k < PLACES; k++) {
      const mint = topMint[k];
      if (mint === null || mint.startsWith('__newborn_')) continue;
      const m = placeProb[k]!;
      m.set(mint, (m.get(mint) ?? 0) + 1);
      const bin = binOfCap(topCap[k]!);
      sizeHist[k]![bin] = (sizeHist[k]![bin] ?? 0) + 1;
    }
    // newborn-held places still count for the size histogram (their cap is real)
    for (let k = 0; k < PLACES; k++) {
      if (topMint[k] !== null && topMint[k]!.startsWith('__newborn_')) {
        const bin = binOfCap(topCap[k]!);
        sizeHist[k]![bin] = (sizeHist[k]![bin] ?? 0) + 1;
      }
    }
  }
}

function binOfCap(logCap: number): number {
  let lo = 0; let hi = EDGES.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (EDGES[mid]! <= logCap) lo = mid + 1; else hi = mid; }
  return lo;
}

// average over reads (mixture): divide by NSIMS * reads.length
const denom = NSIMS * reads.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < PLACES; k++) {
  const entries = [...placeProb[k]!.entries()].sort((a, b) => b[1] - a[1]);
  let sum = 0;
  for (const [mint, count] of entries.slice(0, 40)) {
    const p = Math.floor((count / denom) * MICRO) / MICRO;
    if (p < 0.002) break;
    probabilities.push({ mint, place: k + 1, p });
    sum += p;
  }
  void sum;
}
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < PLACES; k++) {
  for (let bin = 0; bin <= EDGES.length; bin++) {
    const p = Math.floor(((sizeHist[k]![bin] ?? 0) / denom) * MICRO) / MICRO;
    if (p >= 0.0005) sizes.push({ place: k + 1, bin, p });
  }
}
// top5: greedy argmax per place, distinct
const top5: string[] = [];
for (let k = 0; k < PLACES; k++) {
  const entries = [...placeProb[k]!.entries()].sort((a, b) => b[1] - a[1]);
  const pick = entries.find(([mint]) => !top5.includes(mint));
  if (pick) top5.push(pick[0]);
}
while (top5.length < PLACES) {
  const fallback = lastRows[top5.length]!.mint;
  if (!top5.includes(fallback)) top5.push(fallback); else break;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
