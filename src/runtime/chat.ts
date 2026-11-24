// Prompts from chat_format.json (pieces the Python compiler derived from the model's chat template; the runtime never runs jinja) and streaming chat on top of LLM.
import type { ChatFormat, ModelPackage } from './package.js';
import type { GenerateOptions, LLM } from './llm.js';

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
}
export interface Turn { user: string; assistant?: string }

const pad2 = (n: number) => String(n).padStart(2, '0');
/** strftime for the placeholders the compiler leaves in chat_format.json (<<date:%Y-%m-%d>> ...): %Y %m %d %H %M %S %% */
export function strftime(format: string, d: Date): string {
  return format.replace(/%([YmdHMS%])/g, (_, c) => ({ Y: String(d.getFullYear()), m: pad2(d.getMonth() + 1), d: pad2(d.getDate()), H: pad2(d.getHours()), M: pad2(d.getMinutes()), S: pad2(d.getSeconds()), '%': '%' }[c as string] as string));
}
export const fillPlaceholders = (text: string, now: Date) => text.replace(/<<date:([^>]*)>>/g, (_, f) => strftime(f, now));

/** The text to feed the model: the turns so far, the last turn without an assistant answer (the model answers next). */
export function buildPrompt(format: ChatFormat, turns: Turn[], system?: string, now = new Date()): string {
  let text = system ? format.first_prefix_with_system.replace('<<system>>', () => system) : format.first_prefix;
  turns.forEach((turn, i) => {
    text += (i ? format.next_user_prefix : '') + turn.user;
    text += turn.assistant === undefined ? format.user_suffix_generation_prompt : format.user_suffix_assistant_prefix + turn.assistant + format.assistant_suffix;
  });
  return fillPlaceholders(text, now);
}

/** number of prompt tokens a single user message becomes (chat format included) */
export const countPromptTokens = (format: ChatFormat | null, tokenizer: Tokenizer, text: string, system?: string) =>
  tokenizer.encode(format ? buildPrompt(format, [{ user: text }], system) : text).length;

export interface ChatOptions extends GenerateOptions { system?: string; now?: Date }

/** Streams the answer to the last turn as text pieces (the stop token is not part of the text). */
export async function* chat(llm: LLM, tokenizer: Tokenizer, turns: Turn[], options: ChatOptions = {}): AsyncGenerator<string> {
  const format = llm.pkg.chatFormat;
  if (!format) throw new Error('this package has no chat_format.json (the model has no usable chat template): use llm.generate() with your own prompt');
  const ids = tokenizer.encode(buildPrompt(format, turns, options.system, options.now));
  const stop = new Set(options.stopTokens ?? llm.pkg.model.eos_token_ids), generated: number[] = [];
  let shown = '';
  for await (const token of llm.generate(ids, options)) {
    if (stop.has(token)) break;
    generated.push(token);
    const text = tokenizer.decode(generated);
    if (text.endsWith('�')) continue;   // an incomplete UTF-8 sequence: wait for the next token
    yield text.slice(shown.length);
    shown = text;
  }
}

/** A tokenizer from the package's tokenizer.json with transformers.js (`lib` = the imported module, e.g. await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4')). */
export async function tokenizerFromTransformersJs(lib: { PreTrainedTokenizer: new (json: unknown, config: unknown) => any }, pkg: ModelPackage): Promise<Tokenizer> {
  const [json, config] = await Promise.all([pkg.source.text('tokenizer/tokenizer.json').then(JSON.parse), pkg.source.text('tokenizer/tokenizer_config.json').then(JSON.parse)]);
  const tokenizer = new lib.PreTrainedTokenizer(json, config);
  return {
    encode: (text) => Array.from(tokenizer.encode(text, { add_special_tokens: false }) as ArrayLike<number>),
    decode: (ids) => tokenizer.decode(ids, { skip_special_tokens: false }),
  };
}
