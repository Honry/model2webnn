// bench/raw/markdown.js: the Markdown renderer of the chat page (answers of the Qwen3.5 models). Plain JavaScript without a DOM: the tests look at the HTML string.
import { describe, expect, it } from 'vitest';
import { markdownToHtml as md } from '../../bench/raw/markdown.js';

const VOID = new Set(['br', 'hr', 'input']);
const ALLOWED = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr', 'strong', 'em', 'del', 'code', 'pre', 'ul', 'ol', 'li', 'input', 'blockquote', 'div', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a']);

/** every tag is one of ours, opened and closed in order; the text between the tags has no raw `<` (everything from the input is escaped) */
function checkBalanced(html: string): void {
  const stack: string[] = [];
  const tag = /<(\/?)([a-z0-9]+)((?:\s+[a-z-]+(?:="[^"<>]*")?)*)\s*>/g;
  let last = 0, m: RegExpExecArray | null;
  while ((m = tag.exec(html))) {
    expect(html.slice(last, m.index), 'raw < or > between tags').not.toMatch(/[<>]/);
    last = m.index + m[0].length;
    const [, closing, name] = m;
    expect(ALLOWED.has(name), `tag <${name}> is not allowed`).toBe(true);
    if (VOID.has(name)) { expect(closing).toBe(''); continue; }
    if (closing) expect(stack.pop(), `unbalanced </${name}> in ${html}`).toBe(name);
    else stack.push(name);
  }
  expect(html.slice(last), 'raw < or > after the last tag').not.toMatch(/[<>]/);
  expect(stack, `unclosed tags in ${html}`).toEqual([]);
}
const render = (text: string) => { const html = md(text); checkBalanced(html); return html; };

const ANSWER = [
  'This image is a screenshot of the **WebNN Image Classification** tool, running in a browser.',
  '',
  '---',
  '',
  '### 🖼️ Input Image',
  '',
  'At the top center, there is a thumbnail of a **tiger** walking through grass. It is shown as `224x224` pixels.',
  '',
  '### ⏱️ Performance Metric',
  '',
  '> **7.40 ms**',
  '> inference time on the NPU',
  '',
  '### 📊 Top predictions',
  '',
  '| Rank | Label | Confidence |',
  '|:----:|-------|-----------:|',
  '| 1 | tiger, *Panthera tigris* | 93.1% |',
  '| 2 | tiger cat | 4.2% |',
  '| 3 | jaguar | 0.9% |',
  '',
  '### Details',
  '',
  '1. **Model**: `mobilenet_v2`, 3.4 M parameters',
  '   - input: `[1, 3, 224, 224]`',
  '   - output: ~~1000~~ 1001 classes',
  '2. **Backend**: WebNN on `npu`',
  '3. Steps:',
  '   1. load the weights',
  '   2. run `session.run()`',
  '',
  '- [x] checked',
  '- [ ] not checked',
  '',
  '```python',
  'for i in range(3):',
  '    print("<b>", i * 2 ** 3)',
  '```',
  '',
  'See [the docs](https://webnn.io/en/learn_more?a=1&b=2) or https://example.com/a_b_c. Done — 100% sure. 🙂',
].join('\n');

describe('markdown.js: blocks', () => {
  it('headings 1-6, not without a space, not 7', () => {
    expect(render('# A\n## B\n### C\n#### D\n##### E\n###### F')).toBe('<h1>A</h1><h2>B</h2><h3>C</h3><h4>D</h4><h5>E</h5><h6>F</h6>');
    expect(render('#nospace')).toBe('<p>#nospace</p>');
    expect(render('####### seven')).toBe('<p>####### seven</p>');
    expect(render('## Title ##')).toBe('<h2>Title</h2>');
    expect(render('### 🖼️ Input **Image**')).toBe('<h3>🖼️ Input <strong>Image</strong></h3>');
  });
  it('paragraphs and soft line breaks', () => {
    expect(render('one\ntwo\n\nthree')).toBe('<p>one<br>two</p><p>three</p>');
    expect(render('\r\na\r\nb\r\n')).toBe('<p>a<br>b</p>');
  });
  it('horizontal rules', () => {
    expect(render('a\n\n---\n\nb')).toBe('<p>a</p><hr><p>b</p>');
    expect(render('***')).toBe('<hr>');
    expect(render('- - -')).toBe('<hr>');
    expect(render('--')).toBe('<p>--</p>');
  });
  it('blockquotes, with markup inside and nested', () => {
    expect(render('> **7.40 ms**')).toBe('<blockquote><p><strong>7.40 ms</strong></p></blockquote>');
    expect(render('> a\n> b\n>\n> c')).toBe('<blockquote><p>a<br>b</p><p>c</p></blockquote>');
    expect(render('> a\n> > b')).toBe('<blockquote><p>a</p><blockquote><p>b</p></blockquote></blockquote>');
  });
  it('bullet and numbered lists, nested, with a start number', () => {
    expect(render('- a\n- b\n* c\n+ d')).toBe('<ul><li>a</li><li>b</li><li>c</li><li>d</li></ul>');
    expect(render('1. a\n2. b\n10. c')).toBe('<ol><li>a</li><li>b</li><li>c</li></ol>');
    expect(render('3. a\n4. b')).toBe('<ol start="3"><li>a</li><li>b</li></ol>');
    expect(render('- a\n  - b\n    - c\n- d')).toBe('<ul><li>a<ul><li>b<ul><li>c</li></ul></li></ul></li><li>d</li></ul>');
    expect(render('1. a\n   - b\n   - c\n2. d')).toBe('<ol><li>a<ul><li>b</li><li>c</li></ul></li><li>d</li></ol>');
    expect(render('1. a\n  - b (two spaces only)\n2. c')).toBe('<ol><li>a<ul><li>b (two spaces only)</li></ul></li><li>c</li></ol>');
    expect(render('- a\n\n- b')).toBe('<ul><li>a</li><li>b</li></ul>');   // a blank line between items does not split the list
    expect(render('- a\ncontinued\n\ntext')).toBe('<ul><li>a<br>continued</li></ul><p>text</p>');
    expect(render('- **Name**: x')).toBe('<ul><li><strong>Name</strong>: x</li></ul>');
  });
  it('a list item with two paragraphs, a code fence inside an item, a fence that interrupts a numbered list', () => {
    expect(render('1. a\n\n   more\n2. b')).toBe('<ol><li><p>a</p><p>more</p></li><li>b</li></ol>');
    expect(render('- run:\n  ```\n  ls <x>\n  ```\n- next')).toBe('<ul><li>run:<pre class="md-code"><code>ls &lt;x&gt;</code></pre></li><li>next</li></ul>');
    expect(render('1. Run:\n```bash\nls\n```\n2. Next')).toBe('<ol><li>Run:</li></ol><pre class="md-code" data-lang="bash"><code class="language-bash">ls</code></pre><ol start="2"><li>Next</li></ol>');
  });
  it('task list markers', () => {
    expect(render('- [x] done\n- [ ] todo')).toBe('<ul><li><input type="checkbox" disabled checked> done</li><li><input type="checkbox" disabled> todo</li></ul>');
  });
  it('a list line is not a list without a space or content, and a paragraph is interrupted by a list', () => {
    expect(render('-5 degrees')).toBe('<p>-5 degrees</p>');
    expect(render('**bold** start')).toBe('<p><strong>bold</strong> start</p>');
    expect(render('Items:\n- a\n- b')).toBe('<p>Items:</p><ul><li>a</li><li>b</li></ul>');
  });
  it('tables: header, separator, alignment, rows padded / cut to the header, inline markup, escaped pipes', () => {
    const html = render('| A | B | C |\n|:--|:-:|--:|\n| 1 | **2** | `3` |\n| only |\n| x | y | z | extra |\n\nafter');
    expect(html).toBe('<div class="md-table"><table><thead><tr><th style="text-align:left">A</th><th style="text-align:center">B</th><th style="text-align:right">C</th></tr></thead><tbody>'
      + '<tr><td style="text-align:left">1</td><td style="text-align:center"><strong>2</strong></td><td style="text-align:right"><code>3</code></td></tr>'
      + '<tr><td style="text-align:left">only</td><td style="text-align:center"></td><td style="text-align:right"></td></tr>'
      + '<tr><td style="text-align:left">x</td><td style="text-align:center">y</td><td style="text-align:right">z</td></tr></tbody></table></div><p>after</p>');
    expect(render('a | b\n--|--\n1 \\| 2 | 3')).toContain('<td>1 | 2</td>');
    expect(render('| a |\n|---|')).toBe('<div class="md-table"><table><thead><tr><th>a</th></tr></thead></table></div>');
    expect(render('| a | b |\n| c |')).toBe('<p>| a | b |<br>| c |</p>');   // no separator row: not a table
    expect(render('| a | b |\n|---|')).toBe('<p>| a | b |<br>|---|</p>');   // separator of a different width: not (yet) a table
  });
  it('code fences: language, html inside is escaped, no markup inside, ``` inside a longer fence or a line', () => {
    expect(render('```python\nx = "<b>" **not bold**\n```')).toBe('<pre class="md-code" data-lang="python"><code class="language-python">x = &quot;&lt;b&gt;&quot; **not bold**</code></pre>');
    expect(render('```\na\n\n\nb\n```')).toBe('<pre class="md-code"><code>a\n\n\nb</code></pre>');
    expect(render('````md\n```js\nx\n```\n````\nafter')).toBe('<pre class="md-code" data-lang="md"><code class="language-md">```js\nx\n```</code></pre><p>after</p>');
    expect(render('~~~\n``` not a closing fence\n~~~')).toBe('<pre class="md-code"><code>``` not a closing fence</code></pre>');
    expect(render('use ```inline``` triple\nand more')).toBe('<p>use <code>inline</code> triple<br>and more</p>');
    expect(render('```js x\ncode')).toBe('<pre class="md-code md-open" data-lang="js"><code class="language-js">code</code></pre>');   // unclosed: still open
    expect(render('```<script>\nx\n```')).not.toContain('<script');
  });
});

describe('markdown.js: inline', () => {
  it('bold, italic, bold italic, strike, code', () => {
    expect(render('**b** *i* _i_ ***bi*** ~~s~~ `c` __b__')).toBe('<p><strong>b</strong> <em>i</em> <em>i</em> <strong><em>bi</em></strong> <del>s</del> <code>c</code> <strong>b</strong></p>');
    expect(render('**bold with *nested italic* inside**')).toBe('<p><strong>bold with <em>nested italic</em> inside</strong></p>');
    expect(render('**Note:** text')).toBe('<p><strong>Note:</strong> text</p>');
    expect(render('这是**重要**的信息，*强调*。')).toBe('<p>这是<strong>重要</strong>的信息，<em>强调</em>。</p>');
  });
  it('snake_case, math-like asterisks and underscores stay literal', () => {
    expect(render('use snake_case_word and __init__ and my_var_name')).toBe('<p>use snake_case_word and <strong>init</strong> and my_var_name</p>');
    expect(render('snake_case_word')).toBe('<p>snake_case_word</p>');
    expect(render('2*3*4 = 24')).toBe('<p>2*3*4 = 24</p>');
    expect(render('2 * 3 * 4 and 5 * 6')).toBe('<p>2 * 3 * 4 and 5 * 6</p>');
    expect(render('x**2 + y**2')).toBe('<p>x**2 + y**2</p>');
    expect(render('a * b')).toBe('<p>a * b</p>');
    expect(render('x_1 + y_2 = z_3')).toBe('<p>x_1 + y_2 = z_3</p>');
    expect(render('file_name.txt and _real italic_')).toBe('<p>file_name.txt and <em>real italic</em></p>');
  });
  it('markup inside code spans and escapes is not interpreted', () => {
    expect(render('`**x** <b>`')).toBe('<p><code>**x** &lt;b&gt;</code></p>');
    expect(render('``a ` b``')).toBe('<p><code>a ` b</code></p>');
    expect(render(String.raw`\*not italic\* \_x\_ a\|b`)).toBe('<p>*not italic* _x_ a|b</p>');
    expect(render('*`code` in italic*')).toBe('<p><em><code>code</code> in italic</em></p>');
  });
  it('links: http, https, mailto; autolinks without the trailing punctuation; images are only their alt text', () => {
    const rel = 'target="_blank" rel="noopener noreferrer"';
    expect(render('[x](https://a.com/p_q?a=1&b=2)')).toBe(`<p><a href="https://a.com/p_q?a=1&amp;b=2" ${rel}>x</a></p>`);
    expect(render('[**x**](mailto:a@b.c) and [y](http://h.io/a_(b)) "z"')).toBe(`<p><a href="mailto:a@b.c" ${rel}><strong>x</strong></a> and <a href="http://h.io/a_(b)" ${rel}>y</a> &quot;z&quot;</p>`);
    expect(render('see https://example.com/a_b_c. And (https://x.org/y).')).toBe(`<p>see <a href="https://example.com/a_b_c" ${rel}>https://example.com/a_b_c</a>. And (<a href="https://x.org/y" ${rel}>https://x.org/y</a>).</p>`);
    expect(render('![a tiger](https://x/y.png) after')).toBe('<p>a tiger after</p>');
    expect(render('![](https://x/y.png)')).toBe('<p>[image]</p>');
  });
  it('emoji and other unicode pass through', () => {
    expect(render('🙂 **ok** ✅ – “quoted” é')).toBe('<p>🙂 <strong>ok</strong> ✅ – “quoted” é</p>');
  });
});

describe('markdown.js: XSS', () => {
  const cases = [
    '<script>alert(1)</script>',
    '[x](javascript:alert(1))',
    '[x](JaVaScRiPt:alert(1))',
    '[x](data:text/html;base64,PHNjcmlwdD4=)',
    '[x](//evil.com) [y](/relative) [z](#frag) [w](vbscript:x)',
    '<img src=x onerror=alert(1)>',
    '**<b>x</b>**',
    '![x](javascript:alert(1))',
    '[<img src=x onerror=alert(1)>](https://a.com)',
    '[x](https://a.com" onclick="alert(1))',
    '[x](https://a.com/"onmouseover="alert(1))',
    '<a href="javascript:alert(1)">x</a>',
    '`<script>`',
    '```<img src=x onerror=1>\n<script>alert(1)</script>\n```',
    '| <script> | b |\n|---|---|\n| <img src=x onerror=1> | `<i>` |',
    '> <iframe src=javascript:alert(1)>',
    '- <svg onload=alert(1)>',
    '# <style>x{}</style>',
    '&lt;script&gt; &#60;script&#62; &#x3C;script&#x3E;',
    'https://a.com/"><script>alert(1)</script>',
    'https://a.com/&quot;onmouseover=&quot;x',
    '\u0000<script>\u00000\u0000',
  ];
  for (const text of cases) {
    it(JSON.stringify(text), () => {
      const html = render(text);
      expect(html).not.toMatch(/<(script|img|iframe|svg|style)\b/i);
      for (const [tag] of html.matchAll(/<[^>]*>/g)) expect(tag).not.toMatch(/\son\w+=/i);   // an on... attribute inside a tag (the escaped text between the tags may contain the words)
      expect(html).not.toMatch(/href="(?!https?:\/\/|mailto:)/i);
      for (const [, url] of html.matchAll(/href="([^"]*)"/g)) expect(url).not.toMatch(/[<>"\s]|javascript:/i);
    });
  }
  it('exact outputs', () => {
    expect(md('<script>alert(1)</script>')).toBe('<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
    expect(md('[x](javascript:alert(1))')).toBe('<p>x</p>');
    expect(md('<img src=x onerror=alert(1)>')).toBe('<p>&lt;img src=x onerror=alert(1)&gt;</p>');
    expect(md('**<b>x</b>**')).toBe('<p><strong>&lt;b&gt;x&lt;/b&gt;</strong></p>');
    expect(md('&lt;b&gt;')).toBe('<p>&amp;lt;b&amp;gt;</p>');   // an entity in the input is shown as typed, never decoded
  });
  it('very deep nesting and long input do not overflow the stack', () => {
    checkBalanced(md('> '.repeat(500) + 'x'));
    checkBalanced(md(Array.from({ length: 300 }, (_, i) => ' '.repeat(i) + '- x').join('\n')));
    checkBalanced(md('*a '.repeat(20000)));
    checkBalanced(md('['.repeat(5000) + '](' + '('.repeat(5000)));
  });
});

describe('markdown.js: a realistic answer', () => {
  it('renders every section', () => {
    const html = render(ANSWER);
    expect(html).toContain('<p>This image is a screenshot of the <strong>WebNN Image Classification</strong> tool, running in a browser.</p><hr><h3>🖼️ Input Image</h3>');
    expect(html).toContain('<strong>tiger</strong>');
    expect(html).toContain('<code>224x224</code>');
    expect(html).toContain('<h3>⏱️ Performance Metric</h3><blockquote><p><strong>7.40 ms</strong><br>inference time on the NPU</p></blockquote>');
    expect(html).toContain('<th style="text-align:center">Rank</th><th>Label</th><th style="text-align:right">Confidence</th>');
    expect(html).toContain('<td style="text-align:right">93.1%</td>');
    expect(html).toContain('<em>Panthera tigris</em>');
    expect(html).toContain('<li><strong>Model</strong>: <code>mobilenet_v2</code>, 3.4 M parameters<ul><li>input: <code>[1, 3, 224, 224]</code></li><li>output: <del>1000</del> 1001 classes</li></ul></li>');
    expect(html).toContain('<li>Steps:<ol><li>load the weights</li><li>run <code>session.run()</code></li></ol></li>');
    expect(html).toContain('<pre class="md-code" data-lang="python"><code class="language-python">for i in range(3):\n    print(&quot;&lt;b&gt;&quot;, i * 2 ** 3)</code></pre>');
    expect(html).toContain('href="https://webnn.io/en/learn_more?a=1&amp;b=2"');
    expect(html).toContain('>https://example.com/a_b_c</a>. Done — 100% sure. 🙂</p>');
    expect(html.replace(/<code[^]*?<\/code>/g, '')).not.toContain('*');   // every marker outside code was consumed
  });
  it('every prefix of the answer renders without throwing, with balanced tags', () => {
    for (let n = 0; n <= ANSWER.length; n++) render(ANSWER.slice(0, n));
  });
  it('every prefix of a second answer (nested markup, unicode, odd tables)', () => {
    const text = [
      '## **Steps** – plan', '', '> quote with `code` and *italic*', '> > nested', '',
      '1. First **bold**', '    * deep *a*', '        * deeper _b_', '2. Second', '', '| a | b |', '| - | - |', '| `x|y` | **z** |', '| half', '',
      '````', '```inner```', '````', '', '~~~ts', 'let a = 1 * 2 * 3;', '~~~', '', '***', '', 'tail [link](https://x.y/(z)) ![i](a.png) 你好**世界**。',
    ].join('\n');
    for (let n = 0; n <= text.length; n++) render(text.slice(0, n));
  });
  it('an unclosed construct stays literal until it is closed', () => {
    expect(render('This is **bold')).toBe('<p>This is **bold</p>');
    expect(render('This is **bold*')).toBe('<p>This is **bold*</p>');
    expect(render('This is **bold**')).toBe('<p>This is <strong>bold</strong></p>');
    expect(render('use `code')).toBe('<p>use `code</p>');
    expect(render('see [text](https://exa')).toContain('see [text](');
    expect(render('~~gone')).toBe('<p>~~gone</p>');
    expect(render('| a | b |\n|---|---|\n| 1')).toBe('<div class="md-table"><table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td></td></tr></tbody></table></div>');
  });
  it('the tag checker itself catches broken HTML', () => {
    expect(() => checkBalanced('<p><strong>x</p>')).toThrow();
    expect(() => checkBalanced('<p>a < b</p>')).toThrow();
    expect(() => checkBalanced('<script>x</script>')).toThrow();
  });
});
