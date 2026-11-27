export declare const INSTRUCTION: string;
export declare function sourceText(minChars?: number, seed?: number): string;
export declare function buildPromptOfTokens(target: number, count: (text: string) => number, options?: { instruction?: string; tolerance?: number; source?: string }): { text: string; tokens: number };
export declare const answerReserve: (capacity: number) => number;
export declare function minLengthForPrompt(tokens: number, lengths?: number[]): number;
