"""Contract check for the heliaRT CMSIS-Pack PDSC.

Verifies that the generated pack's pdsc preserves the public identity
contract consumers depend on:

  - package vendor/name
  - one <component> per backend with the expected
    (Cclass, Cgroup, Csub, Cvariant, Cversion)
  - HELIA variant is gated by a <condition> on ns-cmsis-nn/heliaCORE
  - the <require> inside that condition targets the exact identity
    ns-cmsis-nn ships and pins Cversion to the agreed minimum

Run modes:

  # Use an existing pdsc file (the workflow's preferred path).
  python3 tools/cmsis_pack/check_pdsc.py path/to/Ambiq.heliaRT.pdsc

  # Or: read the pdsc out of a built .pack archive.
  python3 tools/cmsis_pack/check_pdsc.py path/to/Ambiq.heliaRT.1.13.1.pack

Exit code is 0 on success, 1 on contract violation (with a diff-style
report on stderr).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# Single source of truth for the contract — keep imports lazy so the
# script remains usable from a checkout that has not configured anything.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack import (  # noqa: E402
    BACKENDS,
    CCLASS,
    CGROUP,
    CSUB,
    NS_CMSIS_NN_CCLASS,
    NS_CMSIS_NN_CGROUP,
    NS_CMSIS_NN_CSUB,
    NS_CMSIS_NN_CVARIANT,
    NS_CMSIS_NN_MIN_VERSION,
    NS_CMSIS_NN_VENDOR,
    PACK_NAME,
    PACK_VENDOR,
)


def _load_pdsc(path: Path) -> ET.Element:
    if path.suffix == ".pack":
        with zipfile.ZipFile(path) as zf:
            pdsc_name = next(
                (n for n in zf.namelist() if n.endswith(".pdsc")), None
            )
            if not pdsc_name:
                raise SystemExit(f"no .pdsc inside {path}")
            return ET.fromstring(zf.read(pdsc_name))
    return ET.parse(path).getroot()


def _check(failures: list[str], cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def check_contract(pdsc: ET.Element) -> list[str]:
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
                        ("Cversion", NS_CMSIS_NN_MIN_VERSION),
                    ):
                        got = req.get(attr)
                        _check(
                            failures,
                            got == want,
                            f"heliaCORE <require>: {attr}={got!r}, want {want!r}",
                        )

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        type=Path,
        help="Path to a .pdsc file or a .pack archive containing one.",
    )
    args = ap.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 1

    pdsc = _load_pdsc(args.path)
    failures = check_contract(pdsc)
    if failures:
        print("CMSIS-Pack contract check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"OK: {args.path.name} matches heliaRT pack contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
