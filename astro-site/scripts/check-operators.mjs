import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(site, '..');
const dir = path.join(root, 'tensorflow/lite/micro/kernels/helia');
const rows = JSON.parse(fs.readFileSync(path.join(site, 'src/data/operators.json'), 'utf8'));
const registered = new Set();
for (const file of fs.readdirSync(dir).filter(file => file.endsWith('.cc') && !file.endsWith('_test.cc'))) {
  const text = fs.readFileSync(path.join(dir, file), 'utf8');
  for (const match of text.matchAll(/\bRegister_([A-Z][A-Z_0-9]+)\(/g)) {
    if (!/_INT(?:4|8|16)(?:_INT16)?$/.test(match[1])) registered.add(match[1]);
  }
}
const names = new Set();
for (const row of rows) {
  if (names.has(row.name)) throw new Error(`Duplicate operator ${row.name}`);
  names.add(row.name);
  const source = fs.readFileSync(path.join(dir, row.source), 'utf8');
  if (!row.types.length || !row.notes || !row.family) throw new Error(`Incomplete entry ${row.name}`);
  if (row.name !== 'QUANTIZE' && !source.includes(`Register_${row.name}(`)) throw new Error(`Registration missing: ${row.name} in ${row.source}`);
  if (row.name === 'QUANTIZE' && !fs.readFileSync(path.join(dir, '../quantize.cc'), 'utf8').includes('Register_QUANTIZE(')) throw new Error('QUANTIZE registration missing');
}
for (const name of registered) if (!names.has(name)) throw new Error(`Undocumented HELIA registration: ${name}`);
console.log(`Checked ${rows.length} operator records against HELIA source registrations.`);
