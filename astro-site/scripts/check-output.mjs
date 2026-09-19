import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(site, 'dist');
const build = JSON.parse(fs.readFileSync(path.join(dist, 'build-info.json'), 'utf8'));
const operators = JSON.parse(fs.readFileSync(path.join(site, 'src/data/operators.json'), 'utf8'));
const inventory = fs.readFileSync(path.join(dist, 'operators.md'), 'utf8');
if (!inventory.includes(build.commit)) throw new Error('Operator export provenance mismatch');
for (const op of operators) {
  if (!inventory.includes(`## ${op.name}\n`) || !inventory.includes(op.notes)) {
    throw new Error(`Operator export missing contract: ${op.name}`);
  }
}
const operatorMarkdown = fs.readFileSync(path.join(dist, 'guide/operators/index.md'), 'utf8');
if (!operatorMarkdown.includes('/helia-rt/operators.md')) throw new Error('Operator export is not discoverable');
if (!/^[a-f0-9]{40}$/.test(build.commit) || !build.runtimeVersion) throw new Error('Invalid source provenance');
const redirects = JSON.parse(fs.readFileSync(path.join(site, 'src/data/redirects.json'), 'utf8'));
for (const [route, target] of Object.entries(redirects)) {
  const html = fs.readFileSync(path.join(dist, route, 'index.html'), 'utf8');
  if (!html.includes(`url=${target}`)) throw new Error(`Redirect mismatch: ${route}`);
}
for (const route of ['guide/operators', 'reference/runtime']) {
  const html = fs.readFileSync(path.join(dist, route, 'index.html'));
  const compressed = gzipSync(html).length;
  if (html.length > 250_000 || compressed > 40_000) throw new Error(`Reference page exceeds size budget: ${route}`);
  console.log(`${route}: ${html.length} HTML bytes, ${compressed} gzip bytes`);
}
console.log(`Validated provenance and ${Object.keys(redirects).length} redirects.`);
