import fs from 'node:fs';
import operators from '../src/data/operators.json' with { type: 'json' };
import build from '../src/data/build-info.json' with { type: 'json' };

const source = `https://github.com/AmbiqAI/helia-rt/blob/${build.commit}/tensorflow/lite/micro/kernels/helia/`;
const markdown = [
  '# HELIA operator inventory',
  `Source: ${build.runtimeVersion}, commit ${build.commit}.`,
  'Types below describe inputs. Support depends on shapes, output and weight types, backend and build settings. A listed type can use optimized, reference or storage-only execution.',
  ...operators.map(op => `## ${op.name}\n\nFamily: ${op.family}\n\nInput types: ${op.types.join(', ')}\n\n${op.notes}\n\n[Adapter source](${source}${op.source})`),
].join('\n\n') + '\n';
fs.writeFileSync(new URL('../public/operators.md', import.meta.url), markdown);
console.log(`Exported ${operators.length} operator contracts for readers and agents.`);
