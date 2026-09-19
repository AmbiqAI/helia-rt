import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(siteRoot, '..');
const cache = path.join(siteRoot, '.cache/reference-trial');
const headers = [
  'micro_interpreter.h',
  'micro_mutable_op_resolver.h',
  'micro_op_resolver.h',
  'micro_profiler_interface.h',
  'helia_rt_version.h',
];
fs.mkdirSync(cache, { recursive: true });
const doxygen = spawnSync('doxygen', ['--version'], { encoding: 'utf8' });
if (doxygen.status !== 0 || doxygen.stdout.trim() !== '1.17.0') {
  throw new Error('The reference trial requires Doxygen 1.17.0.');
}
const quoted = (value) => `"${value.replaceAll('\\', '\\\\').replaceAll('"', '\\"')}"`;
const config = [
  'PROJECT_NAME = heliaRT',
  `INPUT = ${headers.map((header) => quoted(path.join(repoRoot, 'tensorflow/lite/micro', header))).join(' ')}`,
  `OUTPUT_DIRECTORY = ${quoted(cache)}`,
  'GENERATE_XML = YES',
  'XML_PROGRAMLISTING = NO',
  'GENERATE_HTML = NO',
  'GENERATE_LATEX = NO',
  'EXTRACT_ALL = YES',
  'EXTRACT_PRIVATE = NO',
  'EXTRACT_STATIC = NO',
  'ENABLE_PREPROCESSING = YES',
  'MACRO_EXPANSION = YES',
  'EXPAND_ONLY_PREDEF = YES',
  'PREDEFINED = TF_LITE_REMOVE_VIRTUAL_DELETE=',
  'QUIET = YES',
  'WARN_IF_UNDOCUMENTED = NO',
].join('\n');
const doxyfile = path.join(cache, 'Doxyfile');
fs.writeFileSync(doxyfile, config);
const extraction = spawnSync('doxygen', [doxyfile], { encoding: 'utf8' });
fs.writeFileSync(path.join(cache, 'doxygen.log'), extraction.stderr ?? '');
if (extraction.status !== 0) throw new Error(extraction.stderr || 'Doxygen failed');

const packageRoot = path.join(siteRoot, 'node_modules/@ambiqai/helia-ui');
const { extractModel, readDoxygenXml } = await import(
  path.join(packageRoot, 'scripts/lib/doxyref-extract.mjs')
);
const dump = await readDoxygenXml(path.join(cache, 'xml'));
const { model, warnings } = extractModel(dump, { language: 'cpp', sourceRoot: repoRoot, name: 'heliaRT' });
fs.writeFileSync(path.join(cache, 'model.json'), JSON.stringify(model, null, 2) + '\n');
fs.writeFileSync(path.join(cache, 'warnings.json'), JSON.stringify(warnings, null, 2) + '\n');
console.log(`Trial XML and model: ${path.relative(siteRoot, cache)}`);
console.log(`Extractor warnings: ${warnings.length}`);
