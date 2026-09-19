# heliaRT documentation

Astro/Starlight site for AmbiqAI/helia-rt#297. The migration is in progress; the existing MkDocs workflow remains the publishing path until content, generated reference, redirects and delivery checks are complete.

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
```

## C++ extraction trial

Install Doxygen 1.17.0, then run:

```sh
npm run reference:trial
```

The trial writes Doxygen XML, the extracted model warnings and a pass/gap report under `.cache/reference-trial/`. It uses the pinned helia-ui package and selected RT headers without modifying the headers. These artifacts are diagnostic and are not published.

The authored runtime API map remains a source-linked guide until the C++ extraction contract is complete. Migration acceptance and remaining scope are tracked in issue #297 and the draft PR.

Use `npm run reference:trial -- --require-complete` to return a failing status when a representative contract is missing. This is not part of the site build until the extraction gaps are resolved.
