import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(site, 'dist');
const build = JSON.parse(fs.readFileSync(path.join(dist, 'build-info.json'), 'utf8'));
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
