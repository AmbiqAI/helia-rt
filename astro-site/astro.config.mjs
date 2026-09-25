// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import redirects from './src/data/redirects.json' with { type: 'json' };
import apiManifest from './src/data/api-manifest.json' with { type: 'json' };
import buildInfo from './src/data/build-info.json' with { type: 'json' };
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';

const base = '/helia-rt';

export default defineConfig({
  site: 'https://ambiqai.github.io',
  base,
  redirects,
  integrations: [
    starlight({
      title: 'heliaRT',
      description: "Ambiq's optimized LiteRT for Microcontrollers runtime.",
      favicon: '/helia-rt-favicon.svg',
      customCss: ['./src/styles/header.css'],
      plugins: [heliaStarlight({
        accent: 'helia-rt',
        sidebar: 'docs',
        header: {
          title: 'heliaRT',
          hub: { label: 'HELIA', href: 'https://ambiqai.github.io/helia-developer-hub/' },
        },
        sections: [
          { label: 'Home', href: `${base}/`, sidebar: false },
          {
            label: 'Getting started', href: `${base}/getting-started/`,
            sidebar: [
              { label: 'Overview', slug: 'getting-started' },
              { label: 'Run your first model', slug: 'getting-started/first-model' },
              { label: 'Choose your path', slug: 'getting-started/choose-your-path' },
              { label: 'Zephyr', slug: 'getting-started/zephyr' },
              { label: 'neuralSPOT-X', slug: 'getting-started/neuralspot-x' },
              { label: 'Build from source', slug: 'getting-started/source' },
              { label: 'Prebuilt archive', slug: 'getting-started/cmake' },
              { label: 'CMSIS-Pack', slug: 'getting-started/cmsis-pack' },
              { label: 'First inference', slug: 'getting-started/first-inference' },
              { label: 'Migrate from LiteRT', slug: 'getting-started/migrate-from-litert' },
            ],
          },
          {
            label: 'User guide', href: `${base}/guide/`,
            sidebar: [
              { label: 'Overview', slug: 'guide' },
              { label: 'Runtime and model', items: [
                { label: 'Runtime concepts', slug: 'guide/runtime' },
                { label: 'Model compatibility', slug: 'guide/model-compatibility' },
                { label: 'Operator explorer', slug: 'guide/operators' },
                { label: 'Target selection', slug: 'guide/targets' },
              ] },
              { label: 'Build and configure', items: [
                { label: 'Build configuration', slug: 'guide/build-configuration' },
                { label: 'SPEED and SIZE', slug: 'guide/kernel-profiles' },
                { label: 'Floating point', slug: 'guide/floating-point' },
                { label: 'Build options', slug: 'guide/build-options' },
                { label: 'Toolchains', slug: 'guide/toolchains' },
              ] },
              { label: 'Measure and diagnose', items: [
                { label: 'Memory and profiling', slug: 'guide/memory-and-profiling' },
                { label: 'Benchmarks', slug: 'guide/benchmarks' },
                { label: 'Troubleshooting', slug: 'guide/troubleshooting' },
              ] },
              { label: 'Maintenance', collapsed: true, items: [
                { label: 'Architecture', slug: 'guide/maintenance/architecture' },
                { label: 'Testing and CI', slug: 'guide/maintenance/testing' },
                { label: 'Upstream sync', slug: 'guide/maintenance/upstream-sync' },
                { label: 'Releases', slug: 'guide/maintenance/releases' },
                { label: 'Support', slug: 'guide/maintenance/support' },
                { label: 'Attribution', slug: 'guide/maintenance/attribution' },
              ] },
            ],
          },
          {
            label: 'API reference', href: `${base}/reference/`,
            sidebar: [
              { label: 'Overview', slug: 'reference' },
              { label: 'Runtime API', slug: 'reference/runtime' },
              ...Array.from(new Set(apiManifest.map(entry => entry.group))).map(group => ({
                label: group,
                collapsed: true,
                items: apiManifest.filter(entry => entry.group === group).map(entry => ({
                  label: entry.name, slug: entry.route,
                })),
              })),
              { label: 'Builtin registrations', collapsed: true, items: ['a-f', 'g-p', 'q-z'].map(range => ({
                label: range.toUpperCase(), slug: `reference/api/micromutableopresolver/${range}`,
              })) },
              { label: 'Documentation version', slug: 'reference/build' },
            ],
          },
        ],
        discoverability: { ogImage: true, jsonLd: true, markdown: true, llms: true },
        footer: {
          links: [
            { label: `Docs: ${buildInfo.runtimeVersion} · ${buildInfo.shortCommit}${buildInfo.modified ? ' (modified)' : ''}`, href: `${base}/reference/build/` },
            { label: 'Getting started', href: `${base}/getting-started/` },
            { label: 'User guide', href: `${base}/guide/` },
            { label: 'API reference', href: `${base}/reference/` },
            { label: 'GitHub', href: 'https://github.com/AmbiqAI/helia-rt' },
          ],
          tagline: 'Ambiq Micro, Inc. Built on LiteRT for Microcontrollers.',
          logo: 'ambiq',
        },
      })],
    }),
  ],
});
