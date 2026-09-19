// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';

const base = '/helia-rt';

export default defineConfig({
  site: 'https://ambiqai.github.io',
  base,
  integrations: [
    starlight({
      title: 'heliaRT',
      description: "Ambiq's optimized LiteRT for Microcontrollers runtime.",
      favicon: '/helia-rt-favicon.svg',
      plugins: [heliaStarlight({
        accent: 'helia-rt',
        sidebar: 'always',
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
              { label: 'Choose your path', slug: 'getting-started/choose-your-path' },
              { label: 'First inference', slug: 'getting-started/first-inference' },
            ],
          },
          {
            label: 'User guide', href: `${base}/guide/`,
            sidebar: [
              { label: 'Overview', slug: 'guide' },
              { label: 'Runtime concepts', slug: 'guide/runtime' },
            ],
          },
          {
            label: 'API reference', href: `${base}/reference/`,
            sidebar: [
              { label: 'Overview', slug: 'reference' },
              { label: 'Runtime API', slug: 'reference/runtime' },
            ],
          },
        ],
        discoverability: { ogImage: true, jsonLd: true, markdown: true, llms: true },
        footer: {
          links: [
            { label: 'Getting started', href: `${base}/getting-started/` },
            { label: 'User guide', href: `${base}/guide/` },
            { label: 'API reference', href: `${base}/reference/` },
            { label: 'GitHub', href: 'https://github.com/AmbiqAI/helia-rt' },
            { label: 'Upstream LiteRT for Microcontrollers', href: 'https://github.com/tensorflow/tflite-micro' },
          ],
          tagline: 'Ambiq Micro, Inc. Built on LiteRT for Microcontrollers.',
          logo: 'ambiq',
        },
      })],
    }),
  ],
});
