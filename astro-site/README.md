# heliaRT documentation

Astro/Starlight site for AmbiqAI/helia-rt#297. The migration is a draft. The proposed workflow builds and validates one Astro artifact, then deploys that artifact for eligible main or release builds. Keep the PR in draft until generated C++ reference coverage and retirement of superseded MkDocs instructions are complete.

## Local development

Use Node 24 or later and npm 11 or later.

```sh
cd astro-site
npm ci
npm run dev
```

The development URL is printed by Astro. The site uses the `/helia-rt/` base path.

```sh
npm run build
npm run check
npm run check:links
npm run check:output
```

Site-wide search uses the Pagefind index created by `npm run build`. Use `npm run preview` to test it; the development server does not build that index. Reload an open preview after rebuilding. Operator filters also work in development.

The prepare step checks the 59-entry operator inventory and public header index against source, and generates commit/version metadata. `check:output` verifies authored redirects, metadata and reference-page payload budgets. Source links use the actual build commit.

## C++ extraction trial

Install Doxygen 1.17.0, then run:

```sh
npm run reference:trial
```

The trial writes Doxygen XML, the extracted model warnings and a pass/gap report under `.cache/reference-trial/`. It uses the pinned helia-ui package and selected RT headers without modifying the headers. An extraction-only filter promotes adjacent ordinary header comments without changing source files or line counts. These artifacts are diagnostic and are not published.

To evaluate a local shared-tooling candidate, pass `--candidate-package /absolute/path/to/helia-ui`. This diagnostic option does not change the released dependency pin or production build.

The authored runtime API map remains a source-linked guide until the C++ extraction contract is complete. Migration acceptance and remaining scope are tracked in issue #297 and the draft PR.

Use `npm run reference:trial -- --require-complete` to return a failing status when a representative contract is missing. This is not part of the site build until the extraction gaps are resolved.
