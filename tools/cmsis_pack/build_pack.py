"""Build a CMSIS-Pack (.pack) for heliaRT.

Phase 3 of issue #147. Consumes the canonical source manifest emitted by
``cmake -P cmake/dump_manifest.cmake`` (one invocation per backend) and
produces a CMSIS-Pack archive with one component variant per backend.

Layout inside the .pack archive::

    Ambiq.helia-rt.pdsc                # pack description (XML)
    LICENSE
    src/                               # all .cc source files referenced by
        ...                            # the manifest, mirroring the repo
    include/                           # public headers (whole repo trees
        ...                            # listed in include_dirs)

Usage::

    python tools/cmsis_pack/build_pack.py --output dist/

The output filename follows CMSIS-Pack convention:
``<vendor>.<name>.<version>.pack``.

The script is dependency-free (Python stdlib only) so it can run in any
CI image without an extra ``pip install`` step.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Pack metadata. Tightly coupled to the heliaRT release identity; bump in
# lockstep with cmake/helia_rt_sources.cmake (HELIA_RT_VERSION).
# ---------------------------------------------------------------------------

PACK_VENDOR = "Ambiq"
PACK_NAME = "helia-rt"
PACK_DESCRIPTION = (
    "Ambiq-optimized TensorFlow Lite for Microcontrollers (TFLM) runtime."
)
PACK_URL = "https://github.com/AmbiqAI/helia-rt/releases/download/"
PACK_LICENSE_FILE = "LICENSE"

# Backend → (Cvariant, human description). Cclass/Cgroup/Csub are shared
# across all variants — see CCLASS/CGROUP/CSUB below.
#
# CMSIS-Pack component identity follows the same 4-level convention used by
# Ambiq's ns-cmsis-nn pack so consumers can pin heliaRT with a parallel
# <require> tag, e.g.::
#
#     <require Cclass="Machine Learning"
#              Cgroup="TFLM Runtime"
#              Csub="helia-rt"
#              Cvendor="Ambiq"
#              Cversion="1.13.1"/>     <!-- or "1.13.1:2.0.0" range -->
#
# The Cvariant axis selects the kernel backend; consumers pick exactly one.
BACKENDS: list[tuple[str, str, str]] = [
    ("reference", "Reference", "Portable reference TFLM kernels"),
    ("cmsis_nn", "CMSIS-NN", "Arm CMSIS-NN optimized kernels (open source)"),
    ("helia", "HELIA", "Ambiq-optimized HELIA kernels (requires ns-cmsis-nn)"),
]

CCLASS = "Machine Learning"
CGROUP = "TFLM Runtime"
CSUB = "helia-rt"

# Identity of the cross-pack dependency (Ambiq ns-cmsis-nn / heliaCORE).
# Pinned to the ns-cmsis-nn release that aligned its CMake / Zephyr / NSX /
# CMSIS-Pack surfaces with heliaRT. Bump the lower bound when a newer
# ns-cmsis-nn release introduces a breaking contract change; widen to a
# range (e.g. "7.25.0:8.0.0") once the next-major compatibility window is
# known. The CI guard at tools/cmsis_pack/check_pdsc.py asserts this value.
#
# 7.25.0 introduced a Cvariant split ("Source" vs "Prebuilt") on the
# heliaCORE component. heliaRT itself ships as source via CMSIS-Pack, so
# we pin our require to the "Source" variant to keep the stack source-
# consistent. Consumers who need the prebuilt heliaCORE for binary-size
# reasons can override at integration time.
NS_CMSIS_NN_VENDOR = "Ambiq"
NS_CMSIS_NN_CCLASS = "Machine Learning"
NS_CMSIS_NN_CGROUP = "NN Lib"
NS_CMSIS_NN_CSUB = "heliaCORE"
NS_CMSIS_NN_CVARIANT = "Source"
NS_CMSIS_NN_MIN_VERSION = "7.25.0"


# ---------------------------------------------------------------------------
# Manifest extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendManifest:
    backend: str
    version: str
    include_dirs: tuple[str, ...]
    common_sources: tuple[str, ...]
    kernel_sources: tuple[str, ...]
    backend_defines: tuple[str, ...]


def dump_manifest(repo_root: Path, backend: str) -> BackendManifest:
    """Run cmake/dump_manifest.cmake for *backend* and parse its JSON."""
    script = repo_root / "cmake" / "dump_manifest.cmake"
    if not script.is_file():
        raise SystemExit(f"dump_manifest.cmake not found at {script}")

    with tempfile.NamedTemporaryFile(
        "w+", suffix=".json", delete=False
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        try:
            subprocess.run(
                [
                    "cmake",
                    f"-DBACKEND={backend}",
                    f"-DMANIFEST_OUT={out_path}",
                    "-P",
                    str(script),
                ],
                check=True,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            # Surface cmake's diagnostics; the default CalledProcessError
            # repr hides them, which makes manifest regressions opaque.
            sys.stderr.write(
                f"cmake -P dump_manifest.cmake failed for backend={backend!r}"
                f" (exit {exc.returncode}):\n"
            )
            if exc.stdout:
                sys.stderr.write(exc.stdout.decode(errors="replace"))
            raise SystemExit(1) from exc
        data = json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)

    if data.get("schema") != "helia-rt-manifest-v1":
        raise SystemExit(
            f"unexpected manifest schema: {data.get('schema')!r}"
        )
    if data["backend"] != backend:
        raise SystemExit(
            f"manifest backend mismatch: requested {backend} got {data['backend']}"
        )
    return BackendManifest(
        backend=data["backend"],
        version=data["version"],
        include_dirs=tuple(data["include_dirs"]),
        common_sources=tuple(data["common_sources"]),
        kernel_sources=tuple(data["kernel_sources"]),
        backend_defines=tuple(data["backend_defines"]),
    )


# ---------------------------------------------------------------------------
# File staging
# ---------------------------------------------------------------------------


def _iter_repo_files(
    repo_root: Path, rel_dir: str, *, suffixes: tuple[str, ...]
) -> Iterable[Path]:
    """Yield repo-relative files under *rel_dir* with one of *suffixes*.

    Skips noisy subtrees (downloads/, tools/make/, examples, tests) so the
    pack does not balloon with TFLM make-system cruft and demo apps.
    """
    base = repo_root / rel_dir
    if not base.exists():
        return
    deny_components = {
        "downloads",
        "examples",
        "integration_tests",
        "testing",
        "tests",
        "test",
        "third_party",  # TFLM-vendored copies; we ship our own third_party_static/.
    }
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in suffixes:
            continue
        rel = path.relative_to(repo_root)
        parts = set(rel.parts)
        if parts & deny_components:
            continue
        # Filenames ending in _test.* are unit tests; skip.
        stem = path.stem
        if stem.endswith("_test") or stem.endswith("_tests"):
            continue
        yield rel


def stage_pack_files(
    repo_root: Path,
    manifests: list[BackendManifest],
    stage_root: Path,
) -> tuple[set[str], set[str], set[str]]:
    """Copy every file referenced by *manifests* into *stage_root*.

    Returns ``(all_sources, all_headers, include_dirs)`` as sets of POSIX
    repo-relative paths. The stage layout mirrors the repo so existing
    ``#include`` directives still resolve when consumers add the staged
    include dirs to their search path.
    """
    sources: set[str] = set()
    headers: set[str] = set()
    include_dirs: set[str] = set()

    # Sources from manifests (per-backend).
    for m in manifests:
        for rel in (*m.common_sources, *m.kernel_sources):
            sources.add(rel)
        for inc in m.include_dirs:
            include_dirs.add(inc if inc != "." else "")

    # Header staging strategy:
    #   * For "narrow" include dirs (everything under the repo root except
    #     the root itself), enumerate every header — these are typically
    #     small self-contained third-party trees (flatbuffers, gemmlowp,
    #     ruy, kissfft) where the consumer needs the whole include set.
    #   * For the bare "" (repo-root) include dir, never recurse the entire
    #     repo. Instead derive header roots from the parent directory of
    #     every source file in the manifest, then enumerate *.h/.hpp/.inc
    #     under each. This captures the headers paired with the staged
    #     sources without dragging in unrelated trees (codegen/, docs/,
    #     site/, gen/, third_party/, ...).
    narrow_include_dirs = {d for d in include_dirs if d}
    needs_root_include = "" in include_dirs

    for inc in narrow_include_dirs:
        for rel in _iter_repo_files(
            repo_root, inc, suffixes=(".h", ".hpp", ".inc")
        ):
            headers.add(rel.as_posix())

    if needs_root_include:
        # Walk each unique source-parent directory.
        source_dirs: set[str] = set()
        for rel in sources:
            parent = os.path.dirname(rel)
            if parent:
                source_dirs.add(parent)
        for sd in source_dirs:
            for rel in _iter_repo_files(
                repo_root, sd, suffixes=(".h", ".hpp", ".inc")
            ):
                headers.add(rel.as_posix())
        # tensorflow/lite/{c,core,kernels,schema} carry headers used by the
        # public TFLM API surface that aren't always in the same dir as a
        # source file. Pull their full header trees so #include "tensorflow/
        # lite/..." resolves cleanly for consumers.
        for tl_subtree in (
            "tensorflow/lite/c",
            "tensorflow/lite/core",
            "tensorflow/lite/kernels",
            "tensorflow/lite/micro",
            "tensorflow/lite/schema",
            "tensorflow/compiler/mlir/lite",
        ):
            for rel in _iter_repo_files(
                repo_root, tl_subtree, suffixes=(".h", ".hpp", ".inc")
            ):
                headers.add(rel.as_posix())

    # Copy sources + headers into stage.
    for rel in sources | headers:
        src = repo_root / rel
        if not src.is_file():
            raise SystemExit(f"manifest references missing file: {rel}")
        dst = stage_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Top-level LICENSE (CMSIS-Pack convention).
    license_src = repo_root / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, stage_root / "LICENSE")

    return sources, headers, include_dirs


# ---------------------------------------------------------------------------
# PDSC generation
# ---------------------------------------------------------------------------


def _ind(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indent for ElementTree (in-place, ET ≥ 3.9)."""
    if hasattr(ET, "indent"):
        ET.indent(elem, space="  ", level=level)


def build_pdsc(
    *,
    version: str,
    manifests: dict[str, BackendManifest],
    sources: set[str],
    headers: set[str],
    include_dirs: set[str],
) -> ET.ElementTree:
    """Build the in-memory ``<package>`` XML tree for the pack."""
    pkg = ET.Element("package", schemaVersion="1.7.36")

    ET.SubElement(pkg, "vendor").text = PACK_VENDOR
    ET.SubElement(pkg, "name").text = PACK_NAME
    ET.SubElement(pkg, "description").text = PACK_DESCRIPTION
    ET.SubElement(pkg, "url").text = PACK_URL
    ET.SubElement(pkg, "license").text = PACK_LICENSE_FILE

    # Releases: minimum viable — current version dated today.
    releases = ET.SubElement(pkg, "releases")
    ET.SubElement(
        releases,
        "release",
        version=version,
        date=date.today().isoformat(),
    ).text = f"heliaRT {version} — generated by tools/cmsis_pack/build_pack.py"

    # Keywords help pack indexers surface heliaRT.
    keywords = ET.SubElement(pkg, "keywords")
    for k in ("Ambiq", "TFLM", "TensorFlow Lite Micro", "Machine Learning"):
        ET.SubElement(keywords, "keyword").text = k

    # ----- conditions --------------------------------------------------
    # The HELIA backend depends on the Ambiq ns-cmsis-nn pack (which exposes
    # the heliaCORE NN-Lib component). CMSIS-Pack expresses inter-pack
    # dependencies through <conditions>; components reference a condition by
    # id via the ``condition`` attribute.
    conditions = ET.SubElement(pkg, "conditions")
    ns_cmsis_nn_cond = ET.SubElement(
        conditions, "condition", id="ns-cmsis-nn present"
    )
    ET.SubElement(ns_cmsis_nn_cond, "description").text = (
        "Requires Ambiq ns-cmsis-nn (heliaCORE) component"
    )
    ET.SubElement(
        ns_cmsis_nn_cond,
        "require",
        Cvendor=NS_CMSIS_NN_VENDOR,
        Cclass=NS_CMSIS_NN_CCLASS,
        Cgroup=NS_CMSIS_NN_CGROUP,
        Csub=NS_CMSIS_NN_CSUB,
        Cvariant=NS_CMSIS_NN_CVARIANT,
        Cversion=NS_CMSIS_NN_MIN_VERSION,
    )

    # ----- components --------------------------------------------------
    components = ET.SubElement(pkg, "components")

    for backend, cvariant, descr in BACKENDS:
        m = manifests[backend]
        comp_attrs = dict(
            Cclass=CCLASS,
            Cgroup=CGROUP,
            Csub=CSUB,
            Cvariant=cvariant,
            Cversion=version,
        )
        if backend == "helia":
            comp_attrs["condition"] = "ns-cmsis-nn present"
        comp = ET.SubElement(components, "component", **comp_attrs)
        ET.SubElement(comp, "description").text = descr
        files = ET.SubElement(comp, "files")

        # Include dirs (each gets a "header" category dir entry).
        for inc in sorted(include_dirs):
            attrs = {"category": "include"}
            attrs["name"] = (inc + "/") if inc else "./"
            ET.SubElement(files, "file", **attrs)

        # Sources — common first (deterministic order), then backend-only.
        backend_sources = sorted(
            set(m.common_sources) | set(m.kernel_sources)
        )
        for rel in backend_sources:
            ET.SubElement(files, "file", category="source", name=rel)

        # Backend-specific compile definitions.
        if m.backend_defines:
            for d in m.backend_defines:
                # preIncludeGlobal stubs encode the variant choice and must
                # stay immutable in consumer projects, so we deliberately do
                # NOT mark them attr="config" (which would copy them into
                # the project tree as user-editable files).
                ET.SubElement(
                    files,
                    "file",
                    category="preIncludeGlobal",
                    name=f".cmsis_pack/define_{d}.h",
                )

    # ----- enumerate every staged header for the indexer ----------------
    # CMSIS-Pack consumers use <file category="header"> for IDE intellisense.
    # Listing them here (outside any <component>) doesn't add them to a
    # build, but does make them browsable. Skipped for now to keep the pdsc
    # compact; revisit if intellisense fidelity matters.
    _ = headers  # reserved for a future pass.

    tree = ET.ElementTree(pkg)
    _ind(pkg)
    return tree


def write_define_stubs(stage_root: Path, manifests: dict[str, BackendManifest]) -> None:
    """Emit the tiny ``define_<X>.h`` files referenced by the pdsc."""
    out_dir = stage_root / ".cmsis_pack"
    out_dir.mkdir(exist_ok=True)
    seen: set[str] = set()
    for m in manifests.values():
        for d in m.backend_defines:
            if d in seen:
                continue
            seen.add(d)
            (out_dir / f"define_{d}.h").write_text(
                f"/* heliaRT CMSIS-Pack: backend define for {d}. */\n"
                f"#ifndef {d}\n#define {d} 1\n#endif\n"
            )


# ---------------------------------------------------------------------------
# Pack archive
# ---------------------------------------------------------------------------


def build_pack(
    repo_root: Path,
    output_dir: Path,
    *,
    version: str | None = None,
    keep_stage: bool = False,
) -> Path:
    """Top-level driver. Returns the path to the produced .pack file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {b: dump_manifest(repo_root, b) for b, *_ in BACKENDS}

    resolved_version = version or manifests["reference"].version
    pack_path = (
        output_dir / f"{PACK_VENDOR}.{PACK_NAME}.{resolved_version}.pack"
    )

    with tempfile.TemporaryDirectory(prefix="helia-rt-pack-") as td:
        stage = Path(td)
        sources, headers, include_dirs = stage_pack_files(
            repo_root, list(manifests.values()), stage
        )
        write_define_stubs(stage, manifests)
        tree = build_pdsc(
            version=resolved_version,
            manifests=manifests,
            sources=sources,
            headers=headers,
            include_dirs=include_dirs,
        )
        pdsc_path = stage / f"{PACK_VENDOR}.{PACK_NAME}.pdsc"
        tree.write(pdsc_path, encoding="utf-8", xml_declaration=True)

        if keep_stage:
            keep_dst = output_dir / f"{PACK_VENDOR}.{PACK_NAME}.{resolved_version}.stage"
            if keep_dst.exists():
                shutil.rmtree(keep_dst)
            shutil.copytree(stage, keep_dst)

        with zipfile.ZipFile(
            pack_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())

    return pack_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="heliaRT repo root (default: inferred from this script's location)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dist"),
        help="output directory for the .pack file (default: ./dist)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="override pack version (default: HELIA_RT_VERSION from the manifest)",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="also copy the unzipped pack contents next to the .pack",
    )
    args = parser.parse_args(argv)

    if not (args.repo_root / "cmake" / "helia_rt_sources.cmake").is_file():
        parser.error(
            f"--repo-root {args.repo_root} does not look like a heliaRT clone"
        )

    pack = build_pack(
        args.repo_root,
        args.output,
        version=args.version,
        keep_stage=args.keep_stage,
    )
    size_kb = pack.stat().st_size / 1024
    print(f"OK: wrote {pack} ({size_kb:,.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
