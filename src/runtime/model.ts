// One entry point for every package: picks the runtime that matches the plan.
import { Gemma4LLM } from './llm-gemma4.js';
import { LLM, type GenerateOptions, type LoadOptions, type SessionOptions } from './llm.js';
import { Qwen35LLM } from './llm-qwen35.js';
import type { Stats } from './llm-common.js';
import { openPackage, type ModelPackage, type PackageSource, type Variant } from './package.js';
import type { Half } from './protocol.js';

/** What the pages and apps use of a loaded model (implemented by LLM and Gemma4LLM). */
export interface Model {
  readonly pkg: ModelPackage;
  readonly maxLength: number;
  readonly variant: Variant;
  readonly chunkSizes: number[];
  readonly chunkSize: number;
  stats: Stats;
  planFor(tokenCount: number): number[];
  generate(tokens: number[], options?: GenerateOptions): AsyncGenerator<number>;
  reconfigure(options: SessionOptions): Promise<void>;
  destroy(): void;
}

export async function loadModel(src: string | PackageSource, options: LoadOptions = {}): Promise<Model> {
  const progress = options.onProgress ?? (() => {});
  progress({ stage: 'package', message: 'reading the package index' });
  const pkg = await openPackage(src, options.cache ?? true, { segments: options.downloadSegments });
  if (pkg.plan.architecture_kind === 'qwen35') return Qwen35LLM.loadFrom(pkg, options);
  return pkg.plan.architecture_kind === 'gemma4' ? Gemma4LLM.loadFrom(pkg, options) : LLM.loadFrom(pkg, options);
}

export type { Half };
