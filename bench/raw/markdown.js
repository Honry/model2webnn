// A small, dependency-free Markdown renderer for the chat answers of the bench pages: markdownToHtml(text) -> HTML string (pure, no DOM).
//
// Safe by construction: every piece of text from the input goes through escapeHtml BEFORE the tags are built, so the only tags in the output are the ones this file writes (h1-h6, p, br, hr, strong, em, del, code, pre,
// ul, ol, li, input[type=checkbox disabled], blockquote, table parts, div.md-table, a). Links: only http(s) and mailto, always target=_blank rel="noopener noreferrer". Images are not rendered (the alt text is shown).
// Built for STREAMING: the page renders a partial answer again and again. Nothing throws, and unfinished constructs degrade quietly: an unclosed ** / * / ` / ~~ / [link]( stays literal until it is closed,
// an unclosed ``` fence is a code block that is still open (class md-open), a half-written table row is padded to the header's width.
//
// Supported: ATX headings, paragraphs (a soft line break is a <br>), **bold**, *italic*, _italic_ (not inside words: snake_case_word, 2*3*4), ***both***, ~~strike~~, `code` spans, fenced code blocks (``` and ~~~, optional language),
// nested bullet / numbered lists (start number kept, task markers [ ] / [x]), > blockquotes, --- *** ___ rules, GFM tables with alignment colons, [text](url), bare http(s) URLs, \* \_ \` \~ \# \| escapes.
// Not supported: setext headings, indented code blocks, reference links, raw HTML (it is shown as text), footnotes, math.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);

const MAX_DEPTH = 12;   // nesting of lists / quotes: deeper input is shown as plain paragraphs (no runaway recursion)
const WORD = "[A-Za-z0-9_\\u00C0-\\u024F\\u0370-\\u03FF\\u0400-\\u04FF]";   // letters and digits of alphabets that put spaces between words (CJK is deliberately NOT a word character: 这是**重要**的 works)
const NUL = "\u0000";

// ---------------------------------------------------------------------------------------------------------------------
// inline
// ---------------------------------------------------------------------------------------------------------------------

/** `code` spans -> placeholders (a run of n backticks closes with a run of exactly n; an unclosed run stays literal) */
function extractCodeSpans(text, put) {
    let out = "", i = 0;
    while (i < text.length) {
        const start = text.indexOf("`", i);
        if (start < 0) { out += text.slice(i); break; }
        let end = start; while (text[end] === "`") end++;
        const length = end - start;
        let search = end, close = -1;
        while (search < text.length) {
            const at = text.indexOf("`", search);
            if (at < 0) break;
            let stop = at; while (text[stop] === "`") stop++;
            if (stop - at === length) { close = at; break; }
            search = stop;
        }
        if (close < 0) { out += text.slice(i, end); i = end; continue; }
        let code = text.slice(end, close).replace(/\n/g, " ");
        if (code.length > 2 && code.startsWith(" ") && code.endsWith(" ") && code.trim()) code = code.slice(1, -1);
        out += text.slice(i, start) + put(`<code>${escapeHtml(code)}</code>`);
        i = close + length;
    }
    return out;
}

const SAFE_URL = /^(?:https?:\/\/|mailto:)[^\s]*$/i;
const anchor = (url, inner) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${inner}</a>`;

/** bold / italic / strike on an already escaped string (placeholders inside are fine) */
function emphasis(s) {
    s = s.replace(new RegExp(`(?<!${WORD}|\\*)\\*\\*\\*(?=[^\\s*])([\\s\\S]+?)(?<=[^\\s*])\\*\\*\\*`, "g"), "<strong><em>$1</em></strong>");
    s = s.replace(new RegExp(`(?<!${WORD}|\\*)\\*\\*(?=\\S)([\\s\\S]+?)(?<=\\S)\\*\\*`, "g"), "<strong>$1</strong>");
    s = s.replace(new RegExp(`(?<!${WORD})__(?=\\S)([\\s\\S]+?)(?<=\\S)__(?!${WORD})`, "g"), "<strong>$1</strong>");
    s = s.replace(/~~(?=\S)([^~\n]+?)(?<=\S)~~/g, "<del>$1</del>");
    s = s.replace(new RegExp(`(?<!${WORD}|\\*)\\*(?=[^\\s*])([\\s\\S]+?)(?<=[^\\s*])\\*(?!${WORD}|\\*)`, "g"), "<em>$1</em>");
    s = s.replace(new RegExp(`(?<!${WORD}|_)_(?=[^\\s_])([\\s\\S]+?)(?<=[^\\s_])_(?!${WORD}|_)`, "g"), "<em>$1</em>");
    return s;
}

/** the inline markup of one block of text (lines joined by \n) -> HTML */
export function inlineToHtml(raw) {
    const stash = [];
    const put = (html) => { stash.push(html); return `${NUL}${stash.length - 1}${NUL}`; };
    let s = extractCodeSpans(String(raw).replace(/\u0000/g, ""), put);
    s = escapeHtml(s);
    s = s.replace(/\\([`*_~#|])/g, (_, c) => put(c));   // backslash escapes
    s = s.replace(/!\[([^\]]*)\]\([^)\s]*(?:\s[^)]*)?\)/g, (_, alt) => (alt ? alt : "[image]"));   // images are not rendered: the alt text
    s = s.replace(/\[([^\[\]]+)\]\(\s*((?:[^\s()]|\([^\s()]*\))*)(?:\s+(?:&quot;.*?&quot;|&#39;.*?&#39;))?\s*\)/g, (whole, text, url) => {
        if (!SAFE_URL.test(url)) return text;   // javascript:, data:, relative ... : only the text
        return put(anchor(url, emphasis(text)));
    });
    s = s.replace(/(^|[\s(>*_~\u0000])(https?:\/\/(?:(?!&lt;|&gt;|&quot;|&#39;)[^\s\u0000])+)/g, (whole, before, url) => {
        let tail = "";
        for (;;) {   // trailing punctuation and an unbalanced ) belong to the sentence, not to the link
            const last = url[url.length - 1];
            if (/[.,;:!?*_~]/.test(last) || (last === ")" && url.split(")").length > url.split("(").length)) { tail = last + tail; url = url.slice(0, -1); } else break;
        }
        return url.length > 8 ? before + put(anchor(url, url)) + tail : whole;
    });
    s = emphasis(s);
    const restore = (text) => text.replace(/\u0000(\d+)\u0000/g, (_, index) => restore(stash[+index]));
    return restore(s);
}

// ---------------------------------------------------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------------------------------------------------

const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
const HR = /^ {0,3}([-*_])(?: *\1){2,} *$/;
const QUOTE = /^ {0,3}>[ ]?(.*)$/;
const LIST_ITEM = /^( *)([-*+]|\d{1,9}[.)])( +)(?=\S)(.*)$/;
const indentOf = (line) => line.length - line.trimStart().length;
const isBlank = (line) => line.trim() === "";
const isFence = (line) => { const m = FENCE.exec(line); return !!m && !(m[1][0] === "`" && m[2].includes("`")); };
const isListItem = (line) => LIST_ITEM.test(line) && !HR.test(line);

const splitRow = (line) => {
    let text = line.trim();
    if (text.startsWith("|")) text = text.slice(1);
    if (text.endsWith("|") && !text.endsWith("\\|")) text = text.slice(0, -1);
    return text.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
};
const SEPARATOR_CELL = /^:?-+:?$/;
function isTableStart(lines, i) {
    if (i + 1 >= lines.length || !lines[i].includes("|") || !lines[i + 1].includes("|")) return false;
    const separator = splitRow(lines[i + 1]);
    return separator.every((cell) => SEPARATOR_CELL.test(cell)) && separator.length === splitRow(lines[i]).length && !isBlank(lines[i]);
}
const isBlockStart = (lines, i) => isFence(lines[i]) || HEADING.test(lines[i]) || HR.test(lines[i]) || QUOTE.test(lines[i]) || isListItem(lines[i]) || isTableStart(lines, i);

function parseFence(lines, i) {
    const [, fence, info] = FENCE.exec(lines[i]);
    const indent = indentOf(lines[i]), closing = new RegExp(`^ {0,3}\\${fence[0]}{${fence.length},}[ \\t]*$`);
    const body = [];
    let j = i + 1;
    while (j < lines.length && !closing.test(lines[j])) { body.push(lines[j].slice(Math.min(indent, indentOf(lines[j])))); j++; }
    const closed = j < lines.length;
    const language = (info.trim().split(/\s+/)[0] ?? "").replace(/[^\w+#.-]/g, "").slice(0, 24);
    const html = `<pre class="md-code${closed ? "" : " md-open"}"${language ? ` data-lang="${language}"` : ""}><code${language ? ` class="language-${language}"` : ""}>${escapeHtml(body.join("\n"))}</code></pre>`;
    return [html, closed ? j + 1 : j];
}

function parseTable(lines, i) {
    const header = splitRow(lines[i]), alignment = splitRow(lines[i + 1]).map((cell) => (cell.startsWith(":") ? (cell.endsWith(":") ? "center" : "left") : cell.endsWith(":") ? "right" : ""));
    const cell = (tag, text, column) => `<${tag}${alignment[column] ? ` style="text-align:${alignment[column]}"` : ""}>${inlineToHtml(text)}</${tag}>`;
    let html = `<div class="md-table"><table><thead><tr>${header.map((text, column) => cell("th", text, column)).join("")}</tr></thead>`;
    let j = i + 2;
    const rows = [];
    while (j < lines.length && !isBlank(lines[j]) && lines[j].includes("|")) {
        const cells = splitRow(lines[j]).slice(0, header.length);
        while (cells.length < header.length) cells.push("");   // a half-written row
        rows.push(`<tr>${cells.map((text, column) => cell("td", text, column)).join("")}</tr>`);
        j++;
    }
    if (rows.length) html += `<tbody>${rows.join("")}</tbody>`;
    return [html + "</table></div>", j];
}

function parseQuote(lines, i, depth) {
    const inner = [];
    let j = i;
    while (j < lines.length && QUOTE.test(lines[j])) { inner.push(QUOTE.exec(lines[j])[1]); j++; }
    return [`<blockquote>${parseBlocks(inner, depth + 1).join("")}</blockquote>`, j];
}

function parseList(lines, i, depth) {
    const first = LIST_ITEM.exec(lines[i]), ordered = /\d/.test(first[2][0]);
    const start = ordered ? parseInt(first[2], 10) : 1;
    const items = [];
    let j = i;
    for (;;) {
        const m = LIST_ITEM.exec(lines[j]);
        const indent = m[1].length, spaces = m[3].length > 4 ? 1 : m[3].length;
        const contentIndent = indent + m[2].length + spaces, nestedFrom = Math.min(contentIndent, indent + 2);
        const body = [m[4]];
        j++;
        while (j < lines.length) {
            const line = lines[j];
            if (isBlank(line)) {   // a blank line stays inside the item only when the next text is indented under it
                let k = j; while (k < lines.length && isBlank(lines[k])) k++;
                if (k < lines.length && indentOf(lines[k]) >= nestedFrom && (indentOf(lines[k]) >= contentIndent || isListItem(lines[k]))) { while (j < k) { body.push(""); j++; } continue; }
                break;
            }
            const at = indentOf(line);
            if (at >= contentIndent) body.push(line.slice(contentIndent));
            else if (at >= nestedFrom && isListItem(line)) body.push(line.trimStart());
            else if (!isListItem(line) && !isBlank(body[body.length - 1]) && !isBlockStart(lines, j) && !isFence(body[body.length - 1])) body.push(line.trimStart());   // lazy continuation of the item's text
            else break;
            j++;
        }
        items.push(body);
        let next = j; while (next < lines.length && isBlank(lines[next])) next++;
        if (next < lines.length && isListItem(lines[next]) && /\d/.test(LIST_ITEM.exec(lines[next])[2][0]) === ordered) { j = next; continue; }
        break;
    }
    const html = items.map((body) => {
        let task = "";
        const m = /^\[([ xX])\][ \t]+(?=\S)/.exec(body[0]);
        if (m) { task = `<input type="checkbox" disabled${m[1] === " " ? "" : " checked"}> `; body[0] = body[0].slice(m[0].length); }
        const blocks = parseBlocks(body, depth + 1);
        if (blocks.length && blocks[0].startsWith("<p>") && blocks.filter((block) => block.startsWith("<p>")).length === 1) blocks[0] = blocks[0].slice(3, -4);   // tight item: no <p> around a lone paragraph
        return `<li>${task}${blocks.join("")}</li>`;
    }).join("");
    return [`<${ordered ? "ol" : "ul"}${ordered && start !== 1 ? ` start="${start}"` : ""}>${html}</${ordered ? "ol" : "ul"}>`, j];
}

/** lines -> an array of HTML blocks */
function parseBlocks(lines, depth) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        if (isBlank(line)) { i++; continue; }
        let m, result = null;
        if (depth < MAX_DEPTH) {
            if (isFence(line)) result = parseFence(lines, i);
            else if ((m = HEADING.exec(line))) result = [`<h${m[1].length}>${inlineToHtml(m[2])}</h${m[1].length}>`, i + 1];
            else if (HR.test(line)) result = ["<hr>", i + 1];
            else if (QUOTE.test(line)) result = parseQuote(lines, i, depth);
            else if (isListItem(line)) result = parseList(lines, i, depth);
            else if (isTableStart(lines, i)) result = parseTable(lines, i);
        }
        if (result) { out.push(result[0]); i = result[1]; continue; }
        const paragraph = [line.trim()];
        i++;
        while (i < lines.length && !isBlank(lines[i]) && (depth >= MAX_DEPTH || !isBlockStart(lines, i))) { paragraph.push(lines[i].trim()); i++; }
        out.push(`<p>${inlineToHtml(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
    }
    return out;
}

/** Markdown text -> HTML string (safe to assign to innerHTML; never throws) */
export function markdownToHtml(text) {
    const source = String(text ?? "");
    try {
        const lines = source.replace(/\r\n?/g, "\n").split("\n").map((line) => line.replace(/^[ \t]+/, (white) => white.replace(/\t/g, "    ")));
        return parseBlocks(lines, 0).join("");
    } catch {
        return `<p>${escapeHtml(source).replace(/\n/g, "<br>")}</p>`;
    }
}
