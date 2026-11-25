// model2webnn/runtime: run model packages written by `m2w compile` (python/) with WebNN.
export { LLM, resolveVariant, type LoadOptions, type SessionOptions, type GenerateOptions, type Stats, type StepTiming } from './llm.js';
export { planPrefill } from './protocol.js';
export { buildPrompt, chat, countPromptTokens, fillPlaceholders, strftime, tokenizerFromTransformersJs, type ChatOptions, type Tokenizer, type Turn } from './chat.js';
export { CachedSource, DirectorySource, UrlSource, openPackage, sourceFrom, type ModelPackage, type PackageSource, type Progress, type Plan, type UrlSourceOptions, type Variant } from './package.js';
export { buildGraph, loadGraph, parseGraphDoc, type GraphDoc } from './ir-loader.js';
