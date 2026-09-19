import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(site, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(site, 'src/data/api-manifest.json'), 'utf8'));
const cache = path.join(site, '.cache/reference');
const candidate = process.argv.indexOf('--candidate-package');
const packageRoot = candidate >= 0 ? path.resolve(process.argv[candidate + 1]) : path.join(site, 'node_modules/@ambiqai/helia-ui');
const emit = candidate < 0 || process.argv.includes('--emit-site');
const run = (command, args) => {
  const result = spawnSync(command, args, { cwd: repo, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(result.error?.message || result.stderr);
  return result.stdout.trim();
};
const doxygenVersion = run('doxygen', ['--version']);
if (doxygenVersion.split(/\s+/)[0] !== '1.17.0') throw new Error(`Reference generation requires Doxygen 1.17.0; found ${doxygenVersion}.`);
fs.mkdirSync(cache, { recursive: true });
const quote = value => `"${value.replaceAll('\\', '\\\\').replaceAll('"', '\\"')}"`;
const headers = [...new Set(manifest.map(entry => entry.header))];
const doxyfile = path.join(cache, 'Doxyfile');
fs.writeFileSync(doxyfile, [
  'PROJECT_NAME = heliaRT',
  `INPUT = ${headers.map(header => quote(path.join(repo, header))).join(' ')}`,
  `OUTPUT_DIRECTORY = ${quote(cache)}`,
  `INPUT_FILTER = ${quote(`python3 ${path.join(site, 'scripts/reference/comment-filter.py')}`)}`,
  'GENERATE_XML = YES', 'XML_PROGRAMLISTING = NO', 'GENERATE_HTML = NO', 'GENERATE_LATEX = NO',
  'EXTRACT_ALL = YES', 'EXTRACT_PRIVATE = NO', 'EXTRACT_STATIC = NO',
  'ENABLE_PREPROCESSING = YES', 'MACRO_EXPANSION = YES', 'EXPAND_ONLY_PREDEF = YES',
  'PREDEFINED = TF_LITE_REMOVE_VIRTUAL_DELETE= FLATBUFFERS_FINAL_CLASS=final TF_LITE_STATIC_MEMORY=1', 'QUIET = YES', 'WARN_IF_UNDOCUMENTED = NO',
].join('\n'));
run('doxygen', [doxyfile]);
const { extractModel, readDoxygenXml } = await import(path.join(packageRoot, 'scripts/lib/doxyref-extract.mjs'));
const { renderReference } = await import(path.join(packageRoot, 'scripts/lib/reference-render.mjs'));
const commit = run('git', ['rev-parse', 'HEAD']);
const dump = await readDoxygenXml(path.join(cache, 'xml'));
const textOf = node => typeof node === 'string' ? node : (node?.children ?? []).map(textOf).join('');
const child = (node, name) => node.children.find(node => node.name === name);
const selected = new Set(manifest.map(entry => entry.symbol));
const names = new Set(manifest.map(entry => entry.name));
// The generated FlatBuffer schema also declares builders and every operator's options.
// Only the model view and explicitly selected application entry points belong here.
dump.compounds = dump.compounds.filter(compound => ['file', 'namespace', 'group'].includes(compound.kind) || selected.has(textOf(child(compound.def, 'compoundname'))));
const compoundNames = new Set(dump.compounds.filter(compound => ['struct', 'class'].includes(compound.kind)).map(compound => textOf(child(compound.def, 'compoundname'))));
for (const compound of dump.compounds.filter(compound => ['file', 'namespace', 'group'].includes(compound.kind))) {
  for (const section of compound.def.children.filter(node => node.name === 'sectiondef')) {
    section.children = section.children.filter(node => {
      if (node.name !== 'memberdef') return true;
      const name = textOf(child(node, 'name'));
      // A C typedef with the struct's own name is the same published type.
      if (node.attrs.kind === 'typedef' && compoundNames.has(name)) return false;
      return names.has(name);
    });
  }
}
const { model: extracted, warnings } = extractModel(dump, {
  language: 'cpp', sourceRoot: repo, name: 'heliaRT',
  sourceUrl: `https://github.com/AmbiqAI/helia-rt/blob/${commit}/{path}#L{line}`,
}, { sourceCommit: commit });
const all = [];
const visit = module => { all.push(...module.symbols); for (const child of module.submodules ?? []) visit(child); };
for (const module of extracted.modules) visit(module);
const modules = manifest.map(entry => {
  const symbol = all.find(symbol => symbol.id === entry.symbol);
  if (!symbol) throw new Error(`Missing public API symbol: ${entry.symbol}; candidates: ${all.filter(symbol => symbol.name === entry.name).map(symbol => symbol.id).join(', ')}`);
  if (symbol.source.path !== entry.header) throw new Error(`Wrong declaration source: ${entry.symbol}`);
  return { path: entry.name.toLowerCase(), name: entry.name, summary: entry.role, description: entry.role, symbols: [symbol], submodules: [] };
});
const symbols = modules.flatMap(module => module.symbols.flatMap(symbol => [symbol, ...(symbol.members ?? [])]));
const ids = new Set();
for (const symbol of symbols) {
  if (ids.has(symbol.id)) throw new Error(`Duplicate API identity: ${symbol.id}`);
  ids.add(symbol.id);
}
const interpreter = symbols.find(symbol => symbol.id === 'tflite::MicroInterpreter');
const constructors = interpreter.members.filter(symbol => symbol.name === 'MicroInterpreter');
if (constructors.length !== 2 || !constructors.some(symbol => /ownership remains with the caller/.test(symbol.description))) throw new Error('Interpreter overload or ownership contract lost.');
for (const id of ['tflite::MicroMutableOpResolver', 'tflite::MicroInterpreter::typed_input_tensor']) {
  if (!/template\s*</.test(symbols.find(symbol => symbol.id === id)?.signature ?? '')) throw new Error(`Template contract lost: ${id}. Install a released helia-ui with C++ template support.`);
}
const resolverInterface = symbols.find(symbol => symbol.id === 'tflite::MicroOpResolver');
if (!resolverInterface.members.filter(member => member.name === 'FindOp').every(member => /= 0$/.test(member.signature))) throw new Error('Pure-virtual resolver contract lost.');
if (!/public tflite::MicroInterpreter/.test(symbols.find(symbol => symbol.id === 'tflite::RecordingMicroInterpreter').signature)) throw new Error('Interpreter inheritance contract lost.');
if (warnings.length) throw new Error(warnings.join('\n'));
const resolverModule = modules.find(module => module.name === 'MicroMutableOpResolver');
const resolver = resolverModule.symbols[0];
const registrations = resolver.members.filter(member => /^Add[A-Z]/.test(member.name) && member.name !== 'AddCustom');
resolver.members = resolver.members.filter(member => !registrations.includes(member));
for (const [suffix, range] of [['a-f', /^[A-F]/], ['g-p', /^[G-P]/], ['q-z', /^[Q-Z]/]]) {
  resolverModule.submodules.push({
    path: `micromutableopresolver.${suffix}`,
    name: `Operator registration ${suffix.toUpperCase()}`,
    summary: 'Builtin registration methods on MicroMutableOpResolver.',
    description: 'Builtin registration methods on [MicroMutableOpResolver](/helia-rt/reference/api/micromutableopresolver/). Each method adds an operator implementation to the resolver. Resolver capacity counts registrations.',
    symbols: registrations.filter(member => range.test(member.name.slice(3))),
    submodules: [],
  });
}
const model = { ...extracted, modules };
const rendered = renderReference(model, { base: '/helia-rt/', site: 'https://ambiqai.github.io', routePrefix: 'reference/api' });
if (rendered.warnings.length) throw new Error(rendered.warnings.join('\n'));
fs.writeFileSync(path.join(cache, 'extraction-warnings.json'), JSON.stringify(warnings, null, 2) + '\n');
const out = emit ? path.join(site, 'src/content/docs/reference/api') : path.join(cache, 'pages');
const publicDir = emit ? path.join(site, 'public') : path.join(cache, 'public');
fs.rmSync(out, { recursive: true, force: true });
fs.rmSync(path.join(publicDir, 'reference/api'), { recursive: true, force: true });
for (const page of rendered.pages) {
  const target = path.join(out, page.path);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const jumpLinks = page.anchors.map(id => {
    const symbol = symbols.find(symbol => symbol.id === id);
    return { id, name: symbol.name, kind: symbol.kind };
  });
  const mdx = page.mdx.replace('sidebar:', 'tableOfContents: false\nsidebar:')
    .replace(/<RefMembers label="On this page" items=\{.*\} \/>/, `<RefMembers label="On this page" items={${JSON.stringify(jumpLinks)}} />`);
  fs.writeFileSync(target, mdx);
}
for (const artifact of rendered.artifacts) {
  const target = path.join(publicDir, artifact.path);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, artifact.contents);
}
console.log(run('python3', [path.join(site, 'scripts/reference/check-coverage.py'), path.join(cache, 'xml'), path.join(site, 'src/data/api-manifest.json'), path.join(publicDir, 'reference/api/reference.json')]));
const report = { commit, headers, pages: rendered.pages.length, symbols: symbols.length, routes: rendered.pages.map(page => ({route: page.route, anchors: page.anchors})), candidate: candidate >= 0, extractionWarnings: warnings, ids: [...ids] };
fs.writeFileSync(path.join(cache, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(`Generated ${rendered.pages.length} static API pages with ${symbols.length} symbols from ${headers.length} public headers${emit ? '' : ' (candidate diagnostic only)'}.`);
