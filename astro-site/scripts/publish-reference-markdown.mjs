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

const bundle = catalog.routes.map(page => {
  if (!page.markdown.startsWith(catalog.base)) throw new Error(`Unexpected Markdown route: ${page.markdown}`);
  const markdown = fs.readFileSync(path.join(dist, page.markdown.slice(catalog.base.length)), 'utf8');
  return `<!-- ${page.url} -->\n\n${markdown.trimEnd()}`;
}).join('\n\n---\n\n') + '\n';
fs.writeFileSync(path.join(dist, 'llms-full.txt'), bundle);
console.log(`Published complete Markdown contracts for ${flatten(model).length} API pages and rebuilt the site agent bundle.`);
