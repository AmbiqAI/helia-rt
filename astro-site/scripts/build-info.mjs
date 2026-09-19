import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(site, '..');
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
const header = fs.readFileSync(path.join(root, 'tensorflow/lite/micro/helia_rt_version.h'), 'utf8');
const version = header.match(/#define HELIA_RT_VERSION "([^"]+)"/)?.[1];
if (!version) throw new Error('Runtime version missing');
const commit = git('rev-parse', 'HEAD');
const tags = git('tag', '--points-at', 'HEAD').split('\n').filter(Boolean);
const buildInfo = {
  runtimeVersion: version,
  commit,
  shortCommit: commit.slice(0, 8),
  sourceUrl: `https://github.com/AmbiqAI/helia-rt/tree/${commit}`,
  commitTime: git('show', '-s', '--format=%cI', 'HEAD'),
  releaseTag: tags.find(tag => [version, `helia-rt-${version}`, `heliaRT-${version}`, `HeliaRT-${version}`].includes(tag)) || null,
  modified: Boolean(git('status', '--porcelain', '--untracked-files=normal')),
};
const json = JSON.stringify(buildInfo, null, 2) + '\n';
fs.mkdirSync(path.join(site, 'src/data'), { recursive: true });
fs.writeFileSync(path.join(site, 'src/data/build-info.json'), json);
fs.writeFileSync(path.join(site, 'public/build-info.json'), json);
console.log(`Docs source: ${version}, ${buildInfo.shortCommit}${buildInfo.modified ? ' (local modifications)' : ''}`);
