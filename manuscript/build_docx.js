// Generic builder: content blocks (see ms_content.js) -> .docx with docx-js.
// Inline markup: *italic*, **bold**, ^{sup}, _{sub}, [@key; @key2] citations,
// {Fig:key} / {Tab:key} cross references, {SFig:key} / {STab:key} for SI items.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, BorderStyle, LineNumberRestartFormat, PageBreak, ShadingType, VerticalAlign,
  TabStopType,
} = require("docx");

const [, , contentFile, outFile] = process.argv;
const content = require(path.resolve(contentFile));
const sizes = JSON.parse(fs.readFileSync(path.resolve(path.dirname(contentFile), "image_sizes.json")));

const FONT = "Calibri";
const BODY_PT = 22;      // half-points: 11 pt
const TABLE_PT = 17;     // 8.5 pt
const CAPTION_PT = 19;   // 9.5 pt
const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;  // US Letter, 1 in
const TEXT_W = PAGE_W - 2 * MARGIN;                    // 9360 DXA

// ---------------------------------------------------------------- numbering
const figNum = {}, tabNum = {}, sfigNum = {}, stabNum = {}, refNum = {};
let nf = 0, nt = 0, nsf = 0, nst = 0, nr = 0;
for (const b of content.blocks) {
  if (b.fig) figNum[b.fig] = ++nf;
  if (b.table) tabNum[b.table] = ++nt;
  if (b.sfig) sfigNum[b.sfig] = ++nsf;
  if (b.stable) stabNum[b.stable] = ++nst;
}
const extNum = content.external || {};
if (content.numbersOut) {
  const out = {};
  Object.entries(sfigNum).forEach(([k, v]) => out["SFig:" + k] = v);
  Object.entries(stabNum).forEach(([k, v]) => out["STab:" + k] = v);
  fs.writeFileSync(path.resolve(path.dirname(contentFile), content.numbersOut), JSON.stringify(out, null, 1));
}   // numbers of items defined in the other document
function refIndex(key) {
  if (!(key in content.refs)) throw new Error("unknown reference key: " + key);
  if (!(key in refNum)) refNum[key] = ++nr;
  return refNum[key];
}
// pass over all text in order to fix citation numbers
function scanText(s) {
  (s.match(/\[@[^\]]+\]/g) || []).forEach(m => m.slice(2, -1).split(";").forEach(k => refIndex(k.trim().replace(/^@/, ""))));
}
for (const b of content.blocks) {
  for (const k of ["p", "caption", "h1", "h2", "h3", "li", "quote"]) if (b[k]) scanText(b[k]);
  if (b.rows) b.rows.forEach(r => r.forEach(c => scanText(String(c))));
  if (b.header) b.header.forEach(c => scanText(String(c)));
  if (b.items) b.items.forEach(scanText);
}

function resolveRefs(s) {
  return s
    .replace(/\{Fig:([^}]+)\}/g, (_, k) => { if (!(k in figNum)) throw new Error("fig " + k); return "Fig. " + figNum[k]; })
    .replace(/\{Tab:([^}]+)\}/g, (_, k) => { if (!(k in tabNum)) throw new Error("tab " + k); return "Table " + tabNum[k]; })
    .replace(/\{SFig:([^}]+)\}/g, (_, k) => { const n = sfigNum[k] || extNum["SFig:" + k]; if (!n) throw new Error("sfig " + k); return "Fig. S" + n; })
    .replace(/\{STab:([^}]+)\}/g, (_, k) => { const n = stabNum[k] || extNum["STab:" + k]; if (!n) throw new Error("stab " + k); return "Table S" + n; })
    .replace(/\[@([^\]]+)\]/g, (_, body) => {
      const nums = body.split(";").map(k => refIndex(k.trim().replace(/^@/, ""))).sort((a, b) => a - b);
      // compress consecutive runs: 3,4,5 -> 3–5
      const out = [];
      for (let i = 0; i < nums.length; i++) {
        let j = i;
        while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
        out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join(","));
        i = j;
      }
      return "[" + out.join(",") + "]";
    });
}

// ---------------------------------------------------------------- inline parser
function runs(text, base = {}) {
  const s = resolveRefs(text);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})/g;
  let last = 0, m;
  const push = (t, extra) => { if (t) out.push(new TextRun({ text: t, font: FONT, size: base.size || BODY_PT, ...base, ...extra })); };
  while ((m = re.exec(s)) !== null) {
    push(s.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("*")) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith("^{")) push(tok.slice(2, -1), { superScript: true });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subScript: true });
    last = m.index + tok.length;
  }
  push(s.slice(last), {});
  return out;
}

const para = (text, opts = {}, base = {}) => new Paragraph({
  children: runs(text, base), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts,
});

function image(file, widthMm) {
  const abs = path.resolve(path.dirname(contentFile), file);
  const sz = sizes[file];
  if (!sz) throw new Error("no size for " + file);
  const wpx = Math.round(widthMm / 25.4 * 96);
  const hpx = Math.round(wpx * sz[1] / sz[0]);
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(abs), transformation: { width: wpx, height: hpx } })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60 }, keepNext: true,
  });
}

function table(b) {
  const ncol = b.header.length;
  const rel = b.widths || Array(ncol).fill(1);
  const tot = rel.reduce((a, c) => a + c, 0);
  const widths = rel.map(w => Math.round(TEXT_W * w / tot));
  const border = { style: BorderStyle.SINGLE, size: 4, color: "52514e" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cell = (txt, i, isHead, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead ? border : none, bottom: (isHead || isLast) ? border : none, left: none, right: none },
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: runs(String(txt), { size: TABLE_PT, bold: isHead }), spacing: { after: 0, line: 240 },
      alignment: (b.align && b.align[i] === "l") ? AlignmentType.LEFT : (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER),
    })],
  });
  const rows = [new TableRow({ tableHeader: true, children: b.header.map((h, i) => cell(h, i, true, false)) })];
  b.rows.forEach((r, ri) => rows.push(new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === b.rows.length - 1)) })));
  return new Table({ rows, width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths });
}

// ---------------------------------------------------------------- build
const children = [];
for (const b of content.blocks) {
  if (b.title) children.push(new Paragraph({ children: runs(b.title, { size: 32, bold: true }), spacing: { after: 200 }, alignment: AlignmentType.LEFT }));
  else if (b.authors) children.push(para(b.authors, { alignment: AlignmentType.LEFT, spacing: { after: 60 } }));
  else if (b.affil) children.push(para(b.affil, { alignment: AlignmentType.LEFT, spacing: { after: 40 } }, { size: 20, italics: true }));
  else if (b.h1) children.push(new Paragraph({ children: runs(b.h1, { size: 26, bold: true }), heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 120 } }));
  else if (b.h2) children.push(new Paragraph({ children: runs(b.h2, { size: 23, bold: true, italics: true }), heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 } }));
  else if (b.h3) children.push(new Paragraph({ children: runs(b.h3, { size: 22, italics: true }), heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 } }));
  else if (b.p) children.push(para(b.p));
  else if (b.plain) children.push(para(b.plain, { alignment: AlignmentType.LEFT }));
  else if (b.eq) children.push(new Paragraph({
    children: [...runs(b.eq, { italics: false }), new TextRun({ text: "\t" + b.num, font: FONT, size: BODY_PT })],
    tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
  }));
  else if (b.items) b.items.forEach(t => children.push(new Paragraph({ children: runs(t), bullet: { level: 0 }, spacing: { after: 60, line: 264 }, alignment: AlignmentType.LEFT })));
  else if (b.fig || b.sfig) {
    children.push(image(b.file, b.width || 90));
    const label = b.fig ? `Figure ${figNum[b.fig]}.` : `Figure S${sfigNum[b.sfig]}.`;
    children.push(new Paragraph({ children: [new TextRun({ text: label + " ", bold: true, font: FONT, size: CAPTION_PT }), ...runs(b.caption, { size: CAPTION_PT })], spacing: { after: 240, line: 252 }, alignment: AlignmentType.JUSTIFIED }));
  } else if (b.table || b.stable) {
    const label = b.table ? `Table ${tabNum[b.table]}.` : `Table S${stabNum[b.stable]}.`;
    children.push(new Paragraph({ children: [new TextRun({ text: label + " ", bold: true, font: FONT, size: CAPTION_PT }), ...runs(b.caption, { size: CAPTION_PT })], spacing: { before: 200, after: 80, line: 252 }, alignment: AlignmentType.JUSTIFIED, keepNext: true }));
    children.push(table(b));
    if (b.note) children.push(para(b.note, { spacing: { before: 60, after: 240 } }, { size: TABLE_PT }));
    else children.push(new Paragraph({ children: [], spacing: { after: 160 } }));
  } else if (b.pagebreak) children.push(new Paragraph({ children: [new PageBreak()] }));
  else if (b.references) {
    const ordered = Object.entries(refNum).sort((a, b) => a[1] - b[1]);
    for (const [key, n] of ordered) {
      children.push(new Paragraph({ children: [new TextRun({ text: `[${n}] `, font: FONT, size: 20 }), ...runs(content.refs[key], { size: 20 })], spacing: { after: 60, line: 252 }, alignment: AlignmentType.LEFT, indent: { left: 480, hanging: 480 } }));
    }
  }
}

const doc = new Document({
  creator: "C. H. Guerrero Navarro",
  title: content.docTitle || "",
  styles: { default: { document: { run: { font: FONT, size: BODY_PT } } } },
  sections: [{
    properties: {
      page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      lineNumbers: content.lineNumbers ? { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS } : undefined,
    },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outFile, buf); console.log("wrote", outFile, "refs:", nr, "figs:", nf, "tables:", nt, "S-figs:", nsf, "S-tables:", nst); });
