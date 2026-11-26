// Pieces shared by the LLM runtimes (llm.ts: Qwen3-like decoders, llm-gemma4.ts: Gemma 4).
/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

/** WebNN (Chromium) rejects a tensor of more than 2^31 - 1 bytes (an MLTensor as well as an operand inside a graph). */
export const WEBNN_MAX_TENSOR_BYTES = 2 ** 31 - 1;
const BYTES_PER_ELEMENT: Record<string, number> = { float32: 4, float16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8, int8: 1, uint8: 1, int4: 0.5, uint4: 0.5 };

/** Throws a readable error before the browser's own when a tensor would be bigger than WebNN allows (a KV cache of a long max length, a huge activation of a big chunk). */
export function checkTensorSize(dataType: string, shape: number[], what = 'tensor'): void {
  const bytes = Math.ceil(shape.reduce((a, b) => a * b, 1) * (BYTES_PER_ELEMENT[dataType] ?? 4));
  if (bytes > WEBNN_MAX_TENSOR_BYTES) {
    throw new Error(`the ${what} ${dataType}[${shape}] would be ${(bytes / 2 ** 30).toFixed(2)} GiB; WebNN tensors are limited to ${WEBNN_MAX_TENSOR_BYTES} bytes (2 GiB): use a smaller max length or chunk size`);
  }
}

export interface Built { name: string; graph: ML; inputs: string[]; outputs: string[] }

/** Per decode step: host = writing the step inputs, enqueue = queueing the dispatches (incl. a flush when due), wait = reading the logits (the device finishing the chain), sampling = picking the token. */
export interface StepTiming { hostMs: number; enqueueMs: number; waitMs: number; samplingMs: number; totalMs: number }
export interface Stats {
  loadMs: number; buildMs: number; weightsMs: number; prefillMs: number; prefillChunks: number; promptTokens: number; decodeSteps: number; decodeMs: number; tpotMs: number;
  stepMs: number[]; steps: StepTiming[]; flushes: number; prefillPlan: number[];
  graphBuilds: { label: string; ms: number }[];   // builder.build() time of every graph of the current session ("decode head", "prefill c256 block", "lm_head", ...)
  imageTokens?: number; visionMs?: number;        // Qwen3.5 with images: image tokens in the prompt, time of the image tower (inside prefillMs)
}
export const emptyStats = (): Stats => ({ loadMs: 0, buildMs: 0, weightsMs: 0, prefillMs: 0, prefillChunks: 0, promptTokens: 0, decodeSteps: 0, decodeMs: 0, tpotMs: 0, stepMs: [], steps: [], flushes: 0, prefillPlan: [], graphBuilds: [] });


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
