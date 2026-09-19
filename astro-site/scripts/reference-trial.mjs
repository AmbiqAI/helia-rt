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

const symbols = [];
function collect(value) {
  if (Array.isArray(value)) value.forEach(collect);
  else if (value && typeof value === 'object') {
    if (typeof value.id === 'string' && typeof value.signature === 'string') symbols.push(value);
    Object.values(value).forEach(collect);
  }
}
collect(model);
const interpreter = symbols.find((symbol) => symbol.id === 'tflite::MicroInterpreter');
const constructors = interpreter?.members.filter((symbol) => symbol.name === 'MicroInterpreter') ?? [];
const resolver = symbols.find((symbol) => symbol.id === 'tflite::MicroMutableOpResolver');
const typedInput = symbols.find((symbol) => symbol.id === 'tflite::MicroInterpreter::typed_input_tensor');
const checks = [
  { name: 'Both interpreter constructors extracted', pass: constructors.length === 2 },
  { name: 'No overload identity warnings', pass: !warnings.some((warning) => warning.includes('duplicate id')) },
  { name: 'Resolver template declaration retained', pass: /template\s*</.test(resolver?.signature ?? '') },
  { name: 'Typed accessor template declaration retained', pass: /template\s*</.test(typedInput?.signature ?? '') },
  { name: 'Constructor ownership contract retained', pass: constructors.some((symbol) => /ownership remains with the caller/.test(symbol.description ?? '')) },
];
const report = {
  doxygen: doxygen.stdout.trim(),
  packageVersion: JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8')).version,
  headers,
  checks,
  warnings,
  complete: checks.every((check) => check.pass),
};
fs.writeFileSync(path.join(cache, 'report.json'), JSON.stringify(report, null, 2) + '\n');
for (const check of checks) console.log(`${check.pass ? 'PASS' : 'GAP'} ${check.name}`);
console.log(report.complete ? 'Representative checks passed; full coverage still requires a public API manifest.' : 'The generated reference is not ready to publish.');
if (process.argv.includes('--require-complete') && !report.complete) process.exitCode = 1;
