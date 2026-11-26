// Types of markdown.js (the page code is plain JavaScript; this file only lets the vitest tests under src/ import it).
export function escapeHtml(text: string): string;
export function inlineToHtml(raw: string): string;
export function markdownToHtml(text: string): string;
