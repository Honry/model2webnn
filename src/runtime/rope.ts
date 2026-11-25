// RoPE tables on the host (the graphs take cos / sin as inputs): default, linear and YaRN scaling, mirroring python/m2w/frontends/hf_llm/rope.py,
// which mirrors transformers' modeling_rope_utils. Angles are rounded to float32 like the reference does (a position of 2000 times an inverse frequency near 1
// is where float32 vs float64 shows up in the 4th digit).

export interface RopeEntry {
  type: 'default' | 'linear' | 'yarn' | 'proportional';
  theta: number;
  factor?: number;
  original_max_position_embeddings?: number;
  beta_fast?: number;
  beta_slow?: number;
  attention_factor?: number;
  mscale?: number;
  mscale_all_dim?: number;
  truncate?: boolean;
  partial_rotary_factor?: number;   // proportional | default (partial rotary)
}

const f32 = Math.fround;

function posFreqs(theta: number, dim: number): number[] {
  return Array.from({ length: dim / 2 }, (_, i) => theta ** ((2 * i) / dim));
}

export function yarnInvFreq(e: RopeEntry, dim: number): { invFreq: number[]; attentionFactor: number } {
  const factor = e.factor as number, original = e.original_max_position_embeddings as number;
  const getMscale = (scale: number, m = 1) => (scale <= 1 ? 1 : 0.1 * m * Math.log(scale) + 1);
  const attentionFactor = e.attention_factor ?? (e.mscale && e.mscale_all_dim ? getMscale(factor, e.mscale) / getMscale(factor, e.mscale_all_dim) : getMscale(factor));
  const correctionDim = (rotations: number) => (dim * Math.log(original / (rotations * 2 * Math.PI))) / (2 * Math.log(e.theta));
  let low = correctionDim(e.beta_fast ?? 32), high = correctionDim(e.beta_slow ?? 1);
  if (e.truncate ?? true) { low = Math.floor(low); high = Math.ceil(high); }
  low = Math.max(low, 0); high = Math.min(high, dim - 1);
  if (low === high) high += 0.001;   // prevent a singularity
  const freqs = posFreqs(e.theta, dim);
  const invFreq = freqs.map((pf, i) => {
    const ramp = Math.min(Math.max((i - low) / (high - low), 0), 1);
    const extrapolation = 1 - ramp;
    return (1 / (factor * pf)) * (1 - extrapolation) + (1 / pf) * extrapolation;
  });
  return { invFreq, attentionFactor };
}

/** inverse frequencies [dim/2] and the factor that multiplies cos and sin */
export function invFreqFor(e: RopeEntry, headDim: number): { invFreq: number[]; attentionFactor: number } {
  if (e.type === 'default' && (e.partial_rotary_factor ?? 1) !== 1) {   // partial rotary (Qwen3.5): the first `rot` dims rotate (exponents over rot); the packed head dims are permuted so that the pairs sit at (i, i + headDim / 2), the others get frequency 0
    const rot = Math.floor(headDim * (e.partial_rotary_factor as number));
    return { invFreq: Array.from({ length: headDim / 2 }, (_, i) => (i < rot / 2 ? 1 / e.theta ** ((2 * i) / rot) : 0)), attentionFactor: 1 };
  }
  if (e.type === 'default') return { invFreq: posFreqs(e.theta, headDim).map((p) => 1 / p), attentionFactor: 1 };
  if (e.type === 'linear') return { invFreq: posFreqs(e.theta, headDim).map((p) => 1 / p / (e.factor as number)), attentionFactor: 1 };
  if (e.type === 'yarn') return yarnInvFreq(e, headDim);
  if (e.type === 'proportional') {   // Gemma 4 full attention: only the first partial_rotary_factor of the frequencies rotate (the exponent divides by the FULL head dim), the rest is 0
    const angles = Math.floor(((e.partial_rotary_factor ?? 1) * headDim) / 2);
    return { invFreq: Array.from({ length: headDim / 2 }, (_, i) => (i < angles ? 1 / e.theta ** ((2 * i) / headDim) : 0)), attentionFactor: 1 };
  }
  throw new Error(`rope type '${(e as RopeEntry).type}' is not supported`);
}

/** cos / sin [positions.length][headDim] as float32; both halves repeat the same angles (rotate-half layout) */
export function cosSin(positions: number[], invFreq: number[], attentionFactor = 1): { cos: Float32Array; sin: Float32Array } {
  const half = invFreq.length, dim = half * 2;
  const cos = new Float32Array(positions.length * dim), sin = new Float32Array(positions.length * dim);
  const inv32 = invFreq.map(f32);
  positions.forEach((position, row) => {
    for (let i = 0; i < half; i++) {
      const angle = f32(f32(position) * inv32[i]);
      const c = f32(Math.cos(angle) * attentionFactor), s = f32(Math.sin(angle) * attentionFactor);
      cos[row * dim + i] = cos[row * dim + half + i] = c;
      sin[row * dim + i] = sin[row * dim + half + i] = s;
    }
  });
  return { cos, sin };
}
