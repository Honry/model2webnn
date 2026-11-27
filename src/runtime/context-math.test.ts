// bench/raw/context-math.js: the capacity / context arithmetic of the chat page.
import { describe, expect, it } from 'vitest';
import { answerReserve, autoMaxNewTokens, cacheCapacity, cacheText, CONTEXT_FULL_SHARE, CONTEXT_WARN_SHARE, contextLevel, contextNote, contextPreview, contextText, lastPositionHint, liveSpeed, shownRows, reserveRows, rowsUsed } from '../../bench/raw/context-math.js';

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
    expect(contextText(188, 127)).toBe('Context 127 / 127');   // never more than the capacity
    expect(cacheText(63, 1023, false)).toBe('63 / 1023');
    expect(cacheText(1200, 1023, true)).toBe('1023 / 1023 · FULL');
  });
  it('the denominator shown is the max length, the numerator the real rows (a full cache reads max length - 1 / max length)', () => {
    expect(shownRows(53, 1023, 1024)).toEqual({ used: 53, limit: 1024 });
    expect(shownRows(5000, 1023, 1024)).toEqual({ used: 1023, limit: 1024 });
    expect(shownRows(null, 1023, 1024)).toEqual({ used: null, limit: 1024 });
    expect(contextText(53, 1023, 1024)).toBe('Context 53 / 1024');
    expect(contextText(null, 1023, 1024)).toBe('Context – / 1024');
    expect(contextText(2154, 1023, 1024)).toBe('Context 1023 / 1024');
    expect(cacheText(28, 1023, false, 1024)).toBe('28 / 1024');
    expect(cacheText(1500, 1023, true, 1024)).toBe('1023 / 1024 · FULL');
    expect(lastPositionHint(1023, 1024)).toContain('max length - 1 rows plus the token being computed');
    expect(lastPositionHint(1024, 1024)).toBe('');   // the protocol that attends over the updated cache uses the whole max length
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

describe('context preview (what the next prompt will hold)', () => {
  const reserve = 255, capacity = 1023;
  it('is what fits: no note when nothing is dropped, colours by the share of the capacity', () => {
    const p = contextPreview({ total: 300, kept: 300, own: 40, capacity, reserve });
    expect(p).toEqual({ used: 300, overflow: 0, tooLong: false, level: 'ok' });
    expect(contextNote(p)).toBeNull();
    expect(contextPreview({ total: 900, kept: 900, own: 40, capacity, reserve, }).level).toBe('warn');
  });
  it('trimming: used is what is kept, the overflow is what trimming removes, amber note', () => {
    const p = contextPreview({ total: 1100, kept: 600, own: 80, capacity, reserve });
    expect(p.used).toBe(600);
    expect(p.overflow).toBe(500);
    expect(p.level).toBe('ok');
    expect(contextNote(p)).toEqual({ text: '+500 tokens: older turns will be dropped', level: 'warn' });
  });
  it('a message that does not fit alone is red, "message too long", and used never exceeds the capacity', () => {
    const p = contextPreview({ total: 5000, kept: 5000, own: 5000, capacity, reserve });
    expect(p.used).toBe(capacity);
    expect(p.tooLong).toBe(true);
    expect(p.level).toBe('full');
    expect(contextNote(p)).toEqual({ text: 'message too long', level: 'full' });
    expect(contextPreview({ total: 1008, kept: 1008, own: 1008, capacity, reserve }).tooLong).toBe(true);   // 1008 + 16 > 1023
    expect(contextPreview({ total: 1007, kept: 1007, own: 1007, capacity, reserve }).tooLong).toBe(false);   // 1007 + 16 = 1023
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
