import fs from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const report = JSON.parse(fs.readFileSync(path.join(site, '.cache/reference/report.json'), 'utf8'));
const build = JSON.parse(fs.readFileSync(path.join(site, 'dist/build-info.json'), 'utf8'));
const model = JSON.parse(fs.readFileSync(path.join(site, 'dist/reference/api/reference.json'), 'utf8'));
if (report.commit !== build.commit || model.generatedFrom?.sourceCommit !== build.commit) throw new Error('API reference and documentation source commits differ.');
let anchors = 0;
let largest = { route: '', html: 0, gzip: 0 };
for (const page of report.routes) {
  const route = page.route.replace(/^\/helia-rt\//, '');
  const html = fs.readFileSync(path.join(site, 'dist', route, 'index.html'), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  for (const anchor of page.anchors) {
    if (ids.filter(id => id === anchor).length !== 1) throw new Error(`Missing or duplicate API anchor ${page.route}#${anchor}`);
    anchors++;
  }
  const bytes = Buffer.byteLength(html);
  const compressed = gzipSync(html).length;
  if (bytes > 250_000 || compressed > 40_000) throw new Error(`API page exceeds budget: ${page.route} (${bytes} HTML, ${compressed} gzip bytes)`);
  if (bytes > largest.html) largest = {route: page.route, html: bytes, gzip: compressed};
}
if (anchors !== report.symbols) throw new Error(`Rendered ${anchors} anchors for ${report.symbols} symbols.`);
console.log(`Checked ${report.pages} static API pages and ${anchors} unique rendered anchors. Largest: ${JSON.stringify(largest)}`);
