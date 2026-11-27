// bench/raw/prompt-text.js: prompts of an exact token count for the chat page presets (1k / 2k / 4k / 8k).
import { describe, expect, it } from 'vitest';
import { answerReserve, buildPromptOfTokens, INSTRUCTION, minLengthForPrompt, sourceText } from '../../bench/raw/prompt-text.js';

const count = (text: string) => Math.round(text.length / 3.8) + 30;   // a stand-in for the tokenizer (the page uses the real one, chat format included)

describe('sourceText', () => {
  it('is deterministic, long enough for 8k tokens, numbered and varied (not one repeated sentence)', () => {
    const a = sourceText(), b = sourceText();
    expect(a).toBe(b);
    expect(a.length).toBeGreaterThanOrEqual(80000);
    expect(a.startsWith('Section 1: ')).toBe(true);
    expect(a).toContain('Section 40: ');
    const sentences = a.split(/(?<=\.)\s+/).filter((s) => s.length > 40);
    expect(new Set(sentences).size / sentences.length).toBeGreaterThan(0.95);
  });
});

describe('buildPromptOfTokens', () => {
  it.each([1024, 2048, 4096, 8192])('is within 1%% of %i tokens and ends with the short question', (target) => {
    const { text, tokens } = buildPromptOfTokens(target, count);
    expect(Math.abs(tokens - target)).toBeLessThanOrEqual(target * 0.01);
    expect(count(text)).toBe(tokens);
    expect(text.endsWith(INSTRUCTION)).toBe(true);
    expect(text.startsWith('Section 1: ')).toBe(true);
  });
  it('follows a different tokenizer (a longer text per token gives a longer prompt)', () => {
    const dense = (text: string) => Math.round(text.length / 2.5), sparse = (text: string) => Math.round(text.length / 5);
    expect(buildPromptOfTokens(2048, dense).text.length).toBeLessThan(buildPromptOfTokens(2048, sparse).text.length);
    expect(Math.abs(buildPromptOfTokens(2048, dense).tokens - 2048)).toBeLessThanOrEqual(2048 * 0.01);
  });
});

describe('minLengthForPrompt', () => {
  const lengths = [1024, 2048, 4096, 8192, 16384];
  it('needs the prompt plus min(512, a quarter of the cache) inside max length - 1 rows', () => {
    expect(answerReserve(1023)).toBe(255); expect(answerReserve(8191)).toBe(512);
    expect(minLengthForPrompt(1024, lengths)).toBe(2048);
    expect(minLengthForPrompt(2048, lengths)).toBe(4096);
    expect(minLengthForPrompt(4096, lengths)).toBe(8192);
    expect(minLengthForPrompt(8192, lengths)).toBe(16384);
    expect(minLengthForPrompt(8192, [1024, 2048])).toBe(8705);   // none of the lengths: the exact bound
  });
});
