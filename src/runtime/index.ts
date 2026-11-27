// model2webnn/runtime: run model packages written by `m2w compile` (python/) with WebNN.
export { LLM, resolveVariant, type LoadOptions, type SessionOptions, type GenerateOptions, type Stats, type StepTiming } from './llm.js';
export { Gemma4LLM } from './llm-gemma4.js';
export { Qwen35LLM } from './llm-qwen35.js';
export { loadModel, type Model } from './model.js';
export { planPrefill, chooseGrid, describeGrid, expandImageTokens, imageTokenCount, DETAIL_PATCHES, type Detail, type Grid } from './protocol.js';
export { VisionTower, prepareImage, pixelsFromRGBA, type PreparedImage } from './vision.js';
export { buildPrompt, chat, countPromptTokens, fillPlaceholders, strftime, tokenizerFromTransformersJs, type ChatOptions, type Tokenizer, type Turn } from './chat.js';
export { CachedSource, DirectorySource, UrlSource, openPackage, sourceFrom, type ModelPackage, type PackageSource, type Progress, type Plan, type UrlSourceOptions, type Variant } from './package.js';
export { buildGraph, loadGraph, parseGraphDoc, type GraphDoc } from './ir-loader.js';
export { FIXED_OVERHEAD_BYTES, deviceWeightBytes, kvRowsFor, kvBytes, expectedMemory, formatBytes, looksLikeMemoryError, nextSmallerLength, type MemoryNeed } from './memory.js';
