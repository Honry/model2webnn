// bench/raw/autoscroll.js: the chat-style auto-scroll of the message list. The DOM is replaced by a few fakes (a container with scroll metrics, requestAnimationFrame as a queue, the observers as hooks).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createAutoScroll, isPinned } from '../../bench/raw/autoscroll.js';

class FakeContainer {
  scrollHeight = 1000; clientHeight = 400; scrollTop = 600; children: unknown[] = []; writes = 0;
  private listeners: (() => void)[] = [];
  addEventListener(type: string, fn: () => void) { if (type === 'scroll') this.listeners.push(fn); }
  /** the reader scrolls (or the browser reports our own write) */
  scrollTo(top: number) { this.scrollTop = top; this.listeners.forEach((fn) => fn()); }
}
class FakeButton { hidden = true; onclick: (() => void) | null = null; }

let frames: (() => void)[] = [], mutate: (() => void) | null = null;
const runFrame = () => { const queued = frames; frames = []; queued.forEach((fn) => fn()); };

beforeEach(() => {
  frames = []; mutate = null;
  vi.stubGlobal('requestAnimationFrame', (fn: () => void) => { frames.push(fn); return frames.length; });
  vi.stubGlobal('ResizeObserver', class { constructor(public cb: () => void) {} observe() {} });
  vi.stubGlobal('MutationObserver', class { constructor(cb: (records: unknown[]) => void) { mutate = () => cb([]); } observe() {} });
});
afterEach(() => vi.unstubAllGlobals());

// a container whose scrollTop writes are counted (the write lands at the bottom like a browser would clamp it)
function setup() {
  const box = new FakeContainer(), button = new FakeButton();
  let top = box.scrollTop;
  Object.defineProperty(box, 'scrollTop', { get: () => top, set: (value: number) => { top = Math.min(value, box.scrollHeight - box.clientHeight); box.writes++; } });
  const scroll = createAutoScroll(box as unknown as HTMLElement, { threshold: 80, button: button as unknown as HTMLElement });
  return { box, button, scroll };
}

describe('isPinned', () => {
  it('is true within the threshold of the bottom, false above it', () => {
    expect(isPinned({ scrollHeight: 1000, scrollTop: 600, clientHeight: 400 })).toBe(true);
    expect(isPinned({ scrollHeight: 1000, scrollTop: 530, clientHeight: 400 })).toBe(true);    // 70 px from the bottom
    expect(isPinned({ scrollHeight: 1000, scrollTop: 500, clientHeight: 400 })).toBe(false);   // 100 px
    expect(isPinned({ scrollHeight: 300, scrollTop: 0, clientHeight: 400 })).toBe(true);       // nothing to scroll
  });
});

describe('createAutoScroll', () => {
  it('follows the bottom while content grows, with one scroll write per frame', () => {
    const { box, scroll } = setup();
    box.scrollHeight = 1100; mutate?.(); box.scrollHeight = 1200; mutate?.(); scroll.schedule();   // three changes in one frame
    expect(frames.length).toBe(1);
    runFrame();
    expect(box.scrollTop).toBe(800);
    expect(box.writes).toBe(1);
  });

  it('stops following when the reader scrolls up, shows the jump button, and the button pins again', () => {
    const { box, button, scroll } = setup();
    box.scrollTo(200);   // up, far from the bottom
    expect(scroll.pinned).toBe(false);
    expect(button.hidden).toBe(false);
    box.scrollHeight = 1400; mutate?.(); runFrame();
    expect(box.scrollTop).toBe(200);   // the stream does not drag the reader down
    button.onclick?.(); runFrame();
    expect(scroll.pinned).toBe(true);
    expect(button.hidden).toBe(true);
    expect(box.scrollTop).toBe(1000);
  });

  it('pins by itself when the reader scrolls back to the bottom; a small scroll up inside the threshold stays pinned', () => {
    const { box, scroll } = setup();
    box.scrollTo(200); expect(scroll.pinned).toBe(false);
    box.scrollTo(590); expect(scroll.pinned).toBe(true);
    box.scrollTo(560); expect(scroll.pinned).toBe(true);   // 40 px up: still near the bottom
  });

  it('growth of the content is not mistaken for scrolling up (a big block appended before the frame runs)', () => {
    const { box, scroll } = setup();
    box.scrollHeight = 2400;   // a large code block arrives; scrollTop did not move
    box.scrollTo(600);         // a scroll event with the same position (e.g. from the layout)
    expect(scroll.pinned).toBe(true);
    mutate?.(); runFrame();
    expect(box.scrollTop).toBe(2000);
  });

  it('pin() scrolls down at the next frame (sending a message)', () => {
    const { box, scroll } = setup();
    box.scrollTo(100); expect(scroll.pinned).toBe(false);
    scroll.pin(); runFrame();
    expect(box.scrollTop).toBe(600);
    expect(scroll.pinned).toBe(true);
  });
});
