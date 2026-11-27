import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

// The bench pages are plain ES modules without a build step: a syntax error only shows in the browser console, so every page script is parsed here.
describe('bench page scripts', () => {
  const dir = join(__dirname, '..', '..', 'bench', 'raw');
  for (const name of readdirSync(dir).filter((f) => f.endsWith('.js'))) {
    it(`${name} parses`, () => {
      const out = ts.transpileModule(readFileSync(join(dir, name), 'utf8'), { reportDiagnostics: true, fileName: name, compilerOptions: { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext } });
      expect((out.diagnostics ?? []).map((d) => ts.flattenDiagnosticMessageText(d.messageText, '\n'))).toEqual([]);
    });
  }
});
