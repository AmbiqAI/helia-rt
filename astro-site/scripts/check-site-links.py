from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "dist"
BASE = "/helia-rt/"
ORIGIN = "https://ambiqai.github.io"


class Page(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = set()
        self.duplicates = set()
        self.links = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        anchor = attrs.get("id")
        if anchor:
            if anchor in self.ids:
                self.duplicates.add(anchor)
            self.ids.add(anchor)
        for attribute in ("href", "src"):
            if attrs.get(attribute) and (attribute != "href" or tag == "a" or attrs.get("rel") == "stylesheet"):
                self.links.append(attrs[attribute])


pages = {path: Page(path.read_text()) for path in ROOT.rglob("*.html")}
if not pages:
    raise SystemExit("No built HTML found. Run npm run build first.")
errors = []
for path, page in pages.items():
    route = BASE + path.relative_to(ROOT).as_posix().removesuffix("index.html")
    for anchor in page.duplicates:
        errors.append(f"{route}: duplicate id {anchor}")
    for link in page.links:
        url = urlsplit(urljoin(ORIGIN + route, link))
        if url.scheme not in ("http", "https") or url.netloc != "ambiqai.github.io":
            continue
        if not url.path.startswith(BASE):
            continue
        target = ROOT / unquote(url.path[len(BASE):])
        if url.path.endswith("/"):
            target /= "index.html"
        if not target.is_file():
            errors.append(f"{route}: missing {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f"{route}: missing anchor {link}")
if errors:
    raise SystemExit("\n".join(sorted(set(errors))))
print(f"Checked links, assets and unique anchors in {len(pages)} HTML pages.")
