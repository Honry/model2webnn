// Token selection shared by the LLM runtimes: greedy, or temperature / top-k / top-p sampling with an optional presence penalty. The parameters come from the caller or, when the caller passes no
// temperature, from the package (plan.recommended.sampling: the model card's values; greedy decoding of chat models tends to fall into repetition loops).
import type { Half } from './protocol.js';

export interface SamplingOptions { temperature?: number; topK?: number; topP?: number; presencePenalty?: number }
export interface RecommendedSampling { temperature?: number; top_p?: number; top_k?: number; presence_penalty?: number }
export interface Sampling { temperature: number; topK: number; topP: number; presencePenalty: number }

/** What the caller passed; without a temperature the package's recommended sampling (when it has one), else greedy; temperature 0 = greedy. */
export function samplingOf(o: SamplingOptions, recommended?: RecommendedSampling): Sampling {
  const rec = o.temperature === undefined ? recommended : undefined;
  return { temperature: o.temperature ?? rec?.temperature ?? 0, topK: o.topK ?? rec?.top_k ?? 40, topP: o.topP ?? rec?.top_p ?? 1, presencePenalty: o.presencePenalty ?? rec?.presence_penalty ?? 0 };
}

/** The next token from the first `vocab` logits. `seen` = the tokens generated so far (presence penalty, applied to the logits in place: they are read anew every step). */
export function pickToken(logits: Half, vocab: number, s: Sampling, seen: Set<number> = new Set()): number {
  if (s.presencePenalty) for (const t of seen) logits[t] -= s.presencePenalty;
  if (!s.temperature) {
    let best = 0, bestValue = -Infinity;
    for (let i = 0; i < vocab; i++) if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
    return best;
  }
  const k = Math.min(s.topK, vocab), top: number[] = [];
  for (let i = 0; i < vocab; i++) {
    if (top.length < k || logits[i] > logits[top[top.length - 1]]) {
      top.push(i);
      top.sort((x, y) => logits[y] - logits[x]);
      if (top.length > k) top.pop();
    }
  }
  const max = logits[top[0]], weights = top.map((i) => Math.exp((logits[i] - max) / s.temperature)), total = weights.reduce((x, y) => x + y, 0);
  let keep = top.length;
  if (s.topP < 1) {   // nucleus: the shortest prefix of the (sorted) candidates with the probability mass >= topP
    let mass = 0;
    for (keep = 0; keep < top.length && mass < s.topP; keep++) mass += weights[keep] / total;
  }
  let r = Math.random() * weights.slice(0, keep).reduce((x, y) => x + y, 0);
  for (let j = 0; j < keep; j++) { r -= weights[j]; if (r <= 0) return top[j]; }
  return top[0];
}
