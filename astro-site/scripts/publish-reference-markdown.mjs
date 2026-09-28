import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  RENDER_DEFAULTS, buildIndex, flatten, renderModuleMarkdown, slugSegment,
} from '../node_modules/@ambiqai/helia-ui/scripts/lib/reference-render.mjs';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const model = JSON.parse(fs.readFileSync(path.join(dist, 'reference/api/reference.json'), 'utf8'));
const catalog = JSON.parse(fs.readFileSync(path.join(dist, 'content-index.json'), 'utf8'));
const options = { ...RENDER_DEFAULTS, base: catalog.base, site: catalog.site, routePrefix: 'reference/api' };
const index = buildIndex(model, options);

// Source-based MDX renditions cannot recover contracts held in component props.
for (const module of flatten(model)) {
  const route = module.path.split('.').map(slugSegment).join('/');
  const target = path.join(dist, options.routePrefix, route, 'index.md');
  fs.writeFileSync(target, renderModuleMarkdown(module, { index, options }));
}

const paths = JSON.parse(fs.readFileSync(new URL('../src/data/integration-paths.json', import.meta.url), 'utf8'));
for (const [route, detailed, marker] of [
  ['getting-started', false, 'Have a runtime library'],
  ['getting-started/choose-your-path', true, '## One source of truth'],
]) {
  const target = path.join(dist, route, 'index.md');
  const markdown = fs.readFileSync(target, 'utf8');
  const cards = paths.map(item => `${detailed ? '##' : '###'} ${item.title}\n\n${item.summary}\n\n${detailed ? item.detail + '\n\n' : ''}[View setup guide](${catalog.base.replace(/\/$/, '')}/getting-started/${item.route}/)`).join('\n\n');
  if (!markdown.includes(marker)) throw new Error(`Missing integration card insertion point: ${route}`);
  fs.writeFileSync(target, markdown.replace(marker, `${cards}\n\n${marker}`));
}

const bundle = catalog.routes.map(page => {
  if (!page.markdown.startsWith(catalog.base)) throw new Error(`Unexpected Markdown route: ${page.markdown}`);
  const markdown = fs.readFileSync(path.join(dist, page.markdown.slice(catalog.base.length)), 'utf8');
  return `<!-- ${page.url} -->\n\n${markdown.trimEnd()}`;
}).join('\n\n---\n\n') + '\n';
fs.writeFileSync(path.join(dist, 'llms-full.txt'), bundle);
console.log(`Published complete Markdown contracts for ${flatten(model).length} API pages and rebuilt the site agent bundle.`);
