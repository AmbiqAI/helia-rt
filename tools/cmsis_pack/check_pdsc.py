"""Contract check for the heliaRT CMSIS-Pack PDSC.

Verifies that the generated pack's pdsc preserves the public identity
contract consumers depend on:

  - package vendor/name
  - one <component> per backend with the expected
    (Cclass, Cgroup, Csub, Cvariant, Cversion)
  - HELIA variant is gated by a <condition> on ns-cmsis-nn/heliaCORE
  - the <require> inside that condition targets the exact identity
    ns-cmsis-nn ships
  - the advertised minimum satisfies the matching compiled HELIA source floor

Run modes:

  # A staged pdsc uses its sibling source tree, or an explicit source root.
    python3 tools/cmsis_pack/check_pdsc.py path/to/Ambiq.helia-rt.pdsc
    python3 tools/cmsis_pack/check_pdsc.py file.pdsc --repo-root path/to/helia-rt

  # A .pack uses its own packaged guard, never the tool checkout's sources.
    python3 tools/cmsis_pack/check_pdsc.py path/to/Ambiq.helia-rt.<version>.pack

Exit code is 0 on success, 1 on contract violation (with a diff-style
report on stderr).
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# Shared package identity and guard parser; no source tree is read on import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack import (  # noqa: E402
    BACKENDS,
    CCLASS,
    CGROUP,
    CSUB,
    CORE_VERSION_GUARD,
    NS_CMSIS_NN_CCLASS,
    NS_CMSIS_NN_CGROUP,
    NS_CMSIS_NN_CSUB,
    NS_CMSIS_NN_CVARIANT,
    NS_CMSIS_NN_VENDOR,
    PACK_NAME,
    PACK_VENDOR,
    core_source_min_version,
    parse_core_source_min_version,
)


def _load_pdsc(path: Path, repo_root: Path | None = None) -> tuple[ET.Element, str]:
    if path.suffix == ".pack":
        if repo_root is not None:
            raise ValueError(
                "--repo-root applies to plain PDSC files; .pack owns its sources"
            )
        with zipfile.ZipFile(path) as zf:
            pdsc_name = next(
                (n for n in zf.namelist() if n.endswith(".pdsc")), None
            )
            if not pdsc_name:
                raise SystemExit(f"no .pdsc inside {path}")
            source_floor = parse_core_source_min_version(
                zf.read(CORE_VERSION_GUARD).decode("utf-8")
            )
            return ET.fromstring(zf.read(pdsc_name)), source_floor
    source_floor = core_source_min_version(repo_root or path.parent)
    return ET.parse(path).getroot(), source_floor


def _check(failures: list[str], cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def check_contract(pdsc: ET.Element, *, source_floor: str) -> list[str]:
    """Check metadata against the floor parsed from its matching source guard."""
    failures: list[str] = []

    _check(failures, pdsc.tag == "package", f"root tag = {pdsc.tag!r}, want 'package'")
    vendor = pdsc.findtext("vendor")
    name = pdsc.findtext("name")
    _check(failures, vendor == PACK_VENDOR, f"<vendor>={vendor!r}, want {PACK_VENDOR!r}")
    _check(failures, name == PACK_NAME, f"<name>={name!r}, want {PACK_NAME!r}")

    # ---- components -------------------------------------------------------
    components = pdsc.findall("components/component")
    by_variant = {c.get("Cvariant"): c for c in components}
    expected_variants = {cvariant for _backend, cvariant, _descr in BACKENDS}
    _check(
        failures,
        set(by_variant) == expected_variants,
        f"Cvariant set={sorted(by_variant)}, want {sorted(expected_variants)}",
    )

    pack_version = next(
        (r.get("version") for r in pdsc.findall("releases/release")), None
    )
    for cvariant, comp in by_variant.items():
        for attr, want in (
            ("Cclass", CCLASS),
            ("Cgroup", CGROUP),
            ("Csub", CSUB),
            ("Cversion", pack_version),
        ):
            got = comp.get(attr)
            _check(
                failures,
                got == want,
                f"variant {cvariant!r}: {attr}={got!r}, want {want!r}",
            )

    # ---- HELIA condition --------------------------------------------------
    helia = by_variant.get("HELIA")
    if helia is not None:
        cond_id = helia.get("condition")
        _check(
            failures,
            bool(cond_id),
            "HELIA component is missing a condition= attribute",
        )
        if cond_id:
            cond = pdsc.find(f"conditions/condition[@id='{cond_id}']")
            _check(
                failures,
                cond is not None,
                f"<condition id={cond_id!r}> not declared",
            )
            if cond is not None:
                req = cond.find("require")
                _check(failures, req is not None, "HELIA condition has no <require>")
                if req is not None:
                    for attr, want in (
                        ("Cvendor", NS_CMSIS_NN_VENDOR),
                        ("Cclass", NS_CMSIS_NN_CCLASS),
                        ("Cgroup", NS_CMSIS_NN_CGROUP),
                        ("Csub", NS_CMSIS_NN_CSUB),
                        ("Cvariant", NS_CMSIS_NN_CVARIANT),
                    ):
                        got = req.get(attr)
                        _check(
                            failures,
                            got == want,
                            f"heliaCORE <require>: {attr}={got!r}, want {want!r}",
                        )
                    advertised = req.get("Cversion", "")
                    _check(
                        failures,
                        bool(re.fullmatch(r"\d+\.\d+\.\d+", advertised))
                        and tuple(map(int, advertised.split(".")))
                        >= tuple(map(int, source_floor.split("."))),
                        f"heliaCORE advertised floor {advertised!r} is below "
                        f"compiled HELIA source floor {source_floor}",
                    )

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        help="Matching source root for a plain PDSC (default: its directory); .pack uses its own sources",
    )
    ap.add_argument(
        "path",
        type=Path,
        help="Path to a .pdsc file or a .pack archive containing one.",
    )
    args = ap.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 1

    try:
        pdsc, source_floor = _load_pdsc(args.path, args.repo_root)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"error: cannot establish matching source floor: {exc}", file=sys.stderr)
        return 1
    failures = check_contract(pdsc, source_floor=source_floor)
    if failures:
        print("CMSIS-Pack contract check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    context = (
        "packaged sources" if args.path.suffix == ".pack"
        else str(args.repo_root or args.path.parent)
    )
    print(
        f"OK: {args.path.name} matches heliaRT pack contract; "
        f"source floor {source_floor} from {context}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
