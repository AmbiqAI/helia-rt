import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(fs.readFileSync(path.join(site, 'src/data/api-manifest.json'), 'utf8'));
const names = new Set();
for (const entry of manifest) {
  if (names.has(entry.name)) throw new Error(`Duplicate API entry ${entry.name}`);
  names.add(entry.name);
  const source = fs.readFileSync(path.join(site, '..', entry.header), 'utf8');
  if (!new RegExp(`\\b${entry.name}\\b`).test(source)) throw new Error(`Missing API declaration ${entry.name}`);
}
console.log(`Checked ${names.size} API index entries against source headers.`);
