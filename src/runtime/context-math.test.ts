// bench/raw/context-math.js: the capacity / context arithmetic of the chat page.
import { describe, expect, it } from 'vitest';
import { answerReserve, autoMaxNewTokens, cacheCapacity, cacheText, CONTEXT_FULL_SHARE, CONTEXT_WARN_SHARE, contextLevel, contextText, liveSpeed, reserveRows, rowsUsed } from '../../bench/raw/context-math.js';

describe('capacity', () => {
  it('is max length - 1 rows (the whole max length when the protocol attends over the updated cache), and follows the chosen length', () => {
    expect(cacheCapacity(1024)).toBe(1023);
    expect(cacheCapacity(16384)).toBe(16383);
    expect(cacheCapacity(1024, true)).toBe(1024);
  });
  it('the reserve is min(512, a quarter of the cache), or the explicit maxNewTokens', () => {
    expect(answerReserve(1023)).toBe(255);
    expect(answerReserve(4095)).toBe(512);
    expect(reserveRows(4095)).toBe(512);
    expect(reserveRows(4095, 64)).toBe(64);
  });
  it('an automatic answer may use every row that is left, at least one', () => {
    expect(autoMaxNewTokens(1023, 53)).toBe(970);
    expect(autoMaxNewTokens(1023, 1023)).toBe(1);
    expect(autoMaxNewTokens(1023, 2000)).toBe(1);
  });
});

describe('rows used', () => {
  it('prompt + generated so far, clamped to the capacity, full when it is reached', () => {
    expect(rowsUsed(53, 10, 1023)).toEqual({ used: 63, full: false });
    expect(rowsUsed(1000, 23, 1023)).toEqual({ used: 1023, full: true });
    expect(rowsUsed(1000, 40, 1023)).toEqual({ used: 1023, full: true });
  });
  it('the texts show an unknown count as a dash and mark FULL', () => {
    expect(contextText(null, 4095)).toBe('Context – / 4095');
    expect(contextText(53, 1023)).toBe('Context 53 / 1023');
    expect(cacheText(63, 1023, false)).toBe('63 / 1023');
    expect(cacheText(1200, 1023, true)).toBe('1023 / 1023 · FULL');
  });
});

describe('context level', () => {
  it('is ok below 80% of the capacity, warn from 80%, full from 95%', () => {
    expect([CONTEXT_WARN_SHARE, CONTEXT_FULL_SHARE]).toEqual([0.8, 0.95]);
    expect(contextLevel(0, 1000)).toBe('ok');
    expect(contextLevel(799, 1000)).toBe('ok');
    expect(contextLevel(800, 1000)).toBe('warn');
    expect(contextLevel(949, 1000)).toBe('warn');
    expect(contextLevel(950, 1000)).toBe('full');
    expect(contextLevel(5000, 1000)).toBe('full');
  });
  it('is ok when nothing is known (no count yet, no capacity)', () => {
    expect(contextLevel(null, 1023)).toBe('ok');
    expect(contextLevel(undefined, 1023)).toBe('ok');
    expect(contextLevel(10, 0)).toBe('ok');
  });
});

describe('live speed', () => {
  it('needs two tokens; then the time per token since the first one', () => {
    expect(liveSpeed(1, 500)).toBeNull();
    expect(liveSpeed(5, 0)).toBeNull();
    const speed = liveSpeed(11, 1000);   // 10 intervals in a second
    expect(speed?.tpotMs).toBeCloseTo(100); expect(speed?.tps).toBeCloseTo(10);
  });
});
