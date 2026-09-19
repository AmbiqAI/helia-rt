# heliaRT documentation

Astro/Starlight documentation for heliaRT. Authored pages live in `src/content/docs`; generated API pages and build metadata are ignored. Edit source headers for API contracts and authored pages for product guidance.

## Local development

Use Node 24 or later, npm 11 or later, Python 3.11 or later, and Doxygen 1.17.0.

```sh
cd astro-site
npm ci
npm run dev
```

The site uses `/helia-rt/`. Preparation validates the operator/header inventories, records the actual checkout commit and runtime header version, and generates the application C++ reference. The extraction-only comment filter preserves header line numbers without editing upstream files.

```sh
npm run build
npm run check
npm run check:links
npm run check:output
npm run check:reference
npm run preview
```

Site-wide search uses the Pagefind index created by the production build. Test it in preview, and reload an open tab after rebuilding. Operator filters also work in development.

## Reference scope and checks

`src/data/api-manifest.json` selects application-facing declarations for execution, model loading, registration, allocation, planning, profiling and platform initialization. Internal kernel helpers, test utilities and FlatBuffer builders are excluded. Add a manifest entry when expanding that scope.

The generator requires complete constructor/ownership/template contracts and no extraction warnings. An independent XML pass compares selected class members and overload counts. The output check verifies every generated symbol anchor and limits each page to 250 KB HTML and 40 KB gzip. Resolver registrations are split alphabetically to stay within that budget. API data is rendered statically; no browser-side parser loads the complete reference model.

For shared-tooling development, `npm run reference:build -- --candidate-package /absolute/path/to/helia-ui` produces diagnostic output under `.cache/reference`. Add `--emit-site` only for local rendered validation. This option does not change the production dependency. Production consumes a released helia-ui tag and lockfile; do not patch installed files.

`npm run reference:trial` runs the smaller representative extraction check. Both diagnostic paths require Doxygen 1.17.0.

## Delivery

`.github/workflows/docs.yml` builds and validates one artifact, then deploys that artifact on eligible main updates. Release automation invokes the same workflow with the release tag. The deployment freshness guard prevents an older queued artifact from replacing a newer main build. Pull requests validate without deploying.

Authored redirects live in `src/data/redirects.json`. Keep one current public site and retain useful old routes when reorganizing content. `public/build-info.json` records the exact source revision; it does not redefine the contents of an older runtime archive.
