// Pieces shared by the LLM runtimes (llm.ts: Qwen3-like decoders, llm-gemma4.ts: Gemma 4).
/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

export interface Built { name: string; graph: ML; inputs: string[]; outputs: string[] }

/** Per decode step: host = writing the step inputs, enqueue = queueing the dispatches (incl. a flush when due), wait = reading the logits (the device finishing the chain), sampling = picking the token. */
export interface StepTiming { hostMs: number; enqueueMs: number; waitMs: number; samplingMs: number; totalMs: number }
export interface Stats {
  loadMs: number; buildMs: number; weightsMs: number; prefillMs: number; prefillChunks: number; promptTokens: number; decodeSteps: number; decodeMs: number; tpotMs: number;
  stepMs: number[]; steps: StepTiming[]; flushes: number; prefillPlan: number[];
}
export const emptyStats = (): Stats => ({ loadMs: 0, buildMs: 0, weightsMs: 0, prefillMs: 0, prefillChunks: 0, promptTokens: 0, decodeSteps: 0, decodeMs: 0, tpotMs: 0, stepMs: [], steps: [], flushes: 0, prefillPlan: [] });


/** Starts the downloads in order, at most `window` ahead of the consumer (so the whole model is never held in memory at once). */
export class Prefetcher {
  private tasks = new Map<number, Promise<ArrayBuffer>>();
  private started = new Set<number>();
  constructor(private files: string[], private fetchOne: (file: string, index: number) => Promise<ArrayBuffer>, private window: number) {}
  get(index: number): Promise<ArrayBuffer> {
    for (let i = 0; i < Math.min(this.files.length, index + this.window); i++) {
      if (this.started.has(i)) continue;
      this.started.add(i);
      const task = this.fetchOne(this.files[i], i);
      task.catch(() => {});   // reported when consumed
      this.tasks.set(i, task);
    }
    return this.tasks.get(index) as Promise<ArrayBuffer>;
  }
  release(index: number) { this.tasks.delete(index); }
}
