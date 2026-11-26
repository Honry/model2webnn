import { describe, expect, it } from 'vitest';
import { pickToken, samplingOf } from './sampling.js';

describe('sampling', () => {
  const pick = (logits: number[], options: object, recommended?: object, seen = new Set<number>()) => pickToken(Float32Array.from(logits), logits.length, samplingOf(options, recommended), seen);
  const vocab = 300, logits = Array.from({ length: vocab }, () => 0);
  logits[10] = 5; logits[20] = 4.5; logits[30] = 1;

  it('temperature 0 is greedy and the presence penalty moves the pick away from a token that was generated', () => {
    expect(pick(logits, { temperature: 0 })).toBe(10);
    expect(pick(logits, { temperature: 0, presencePenalty: 1.5 }, undefined, new Set([10]))).toBe(20);   // 5 - 1.5 < 4.5
    expect(pick(logits, { temperature: 0, presencePenalty: 0.2 }, undefined, new Set([10]))).toBe(10);
  });

  it('no temperature and no recommendation: greedy', () => {
    expect(samplingOf({})).toEqual({ temperature: 0, topK: 40, topP: 1, presencePenalty: 0 });
    expect(pick(logits, {})).toBe(10);
  });

  it('top-k / top-p restrict the candidates; the recommendation is used only when the caller passes no temperature', () => {
    const recommended = { temperature: 0.7, top_p: 0.8, top_k: 2, presence_penalty: 0 };
    const picks = new Set<number>();
    for (let i = 0; i < 300; i++) picks.add(pick(logits, {}, recommended));   // top-k 2: tokens 10 and 20 only
    expect([...picks].every((t) => t === 10 || t === 20)).toBe(true);
    const nucleus = new Set<number>();
    for (let i = 0; i < 300; i++) nucleus.add(pick(logits, { temperature: 0.7, topP: 0.5, topK: 5 }));
    expect([...nucleus]).toEqual([10]);   // token 10 alone has more than half of the probability mass
    expect(pick(logits, { temperature: 0 }, recommended)).toBe(10);   // an explicit 0 is greedy whatever the package recommends
    expect(samplingOf({ temperature: 0.2 }, recommended)).toEqual({ temperature: 0.2, topK: 40, topP: 1, presencePenalty: 0 });   // the recommendation is not mixed into the caller's own temperature
  });
});
