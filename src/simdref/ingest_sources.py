"""Source acquisition for intrinsics and instruction catalogs."""

from __future__ import annotations

import json
import html
import io
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx

from simdref.ingest_pdf import _sha256_file as _sha256
from simdref.models import SourceVersion


class SourceUnavailableError(RuntimeError):
    """Raised when every candidate source for a feed is unreachable.

    The installer / CI surfaces this with a hint to use the pre-built
    release catalog via ``simdref update``.
    """


UOPS_XML_URL = "https://uops.info/instructions.xml"
INTEL_OFFLINE_ZIP_URL = "https://cdrdv2.intel.com/v1/dl/getContent/764289?fileName=Intel-Intrinsics-Guide-Offline-3.6.4.zip"
INTEL_INDEX_URL = "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html"
INTEL_CANDIDATE_DATA_URLS = [
    "https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/files/data.js",
    "https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/files/intrinsics.json",
]
ARM_ACLE_REPO_URL = "https://github.com/ARM-software/acle"
ARM_ACLE_DOC_URL = "https://arm-software.github.io/acle/main/"
ARM_NEON_DOC_URL = "https://arm-software.github.io/acle/neon_intrinsics/advsimd.html"
ARM_A64_DOC_URL = "https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions"
ARM_A64_ARCHIVE_URL = (
    "https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-OS-Machine-Readable-Data/"
    "AARCHMRS_BSD/AARCHMRS_OPENSOURCE_A_profile_FAT-2026-06.tar.gz"
)
ARM_ACLE_ARCHIVE_URL = "https://codeload.github.com/ARM-software/acle/zip/refs/heads/main"
ARM_INTRINSICS_DATA_BASE_URL = (
    "https://developer.arm.com/architectures/instruction-sets/intrinsics/data/"
)
ARM_INTRINSICS_JSON_URL = ARM_INTRINSICS_DATA_BASE_URL + "intrinsics.json"
ARM_INTRINSICS_OPERATIONS_JSON_URL = ARM_INTRINSICS_DATA_BASE_URL + "operations.json"
ARM_INTRINSICS_EXAMPLES_JSON_URL = ARM_INTRINSICS_DATA_BASE_URL + "examples.json"
RISCV_UNIFIED_DB_REPO_URL = "https://github.com/riscv-software-src/riscv-unified-db"
RISCV_RVV_INTRINSICS_REPO_URL = "https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc"
RISCV_UNIFIED_DB_CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/riscv-software-src/riscv-unified-db/main/build/instructions.json",
    "https://raw.githubusercontent.com/riscv-software-src/riscv-unified-db/main/generated/instructions.json",
    "https://raw.githubusercontent.com/riscv-software-src/riscv-unified-db/main/artifacts/instructions.json",
]
# The dead static intrinsics.json URLs are gone: upstream no longer ships one.
# tools/vendor_riscv_sources.py clones the upstream repos and generates the
# bundle, and is the single source of truth for RVV data (see #65 item N1).
PTX_ISA_DOC_URL = "https://docs.nvidia.com/cuda/parallel-thread-execution/"
ARM_ACLE_NEON_DB_PATH = "tools/intrinsic_db/advsimd.csv"
ARM_ACLE_NEON_CLASSIFICATION_PATH = "tools/intrinsic_db/advsimd_classification.csv"
ARM_ACLE_MAIN_MD_PATH = "main/acle.md"
ARM_ACLE_NEON_MD_PATH = "neon_intrinsics/advsimd.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_INTEL_ARCHIVES = [
    _REPO_ROOT / "vendor" / "intel" / "Intel-Intrinsics-Guide-Offline-3.6.9.zip",
]
LOCAL_INTEL_CACHE_ARCHIVES = [
    _REPO_ROOT / "vendor" / "intel-cache" / "intel-intrinsics.zip",
]
LOCAL_UOPS_XMLS = [
    _REPO_ROOT / "vendor" / "uops" / "instructions.xml",
]
LOCAL_ARM_ACLE_JSONS = [
    _REPO_ROOT / "vendor" / "arm" / "acle_intrinsics.json",
]
LOCAL_ARM_INTRINSICS_JSONS = [
    _REPO_ROOT / "vendor" / "arm" / "intrinsics.json",
]
LOCAL_ARM_OPERATIONS_JSONS = [
    _REPO_ROOT / "vendor" / "arm" / "operations.json",
]
LOCAL_ARM_EXAMPLES_JSONS = [
    _REPO_ROOT / "vendor" / "arm" / "examples.json",
]
LOCAL_ARM_ACLE_ARCHIVES = [
    _REPO_ROOT / "vendor" / "arm" / "acle-main.zip",
]
LOCAL_ARM_A64_JSONS = [
    _REPO_ROOT / "vendor" / "arm" / "a64_instructions.json",
]
ARM_A64_ARCHIVE_CACHE = _REPO_ROOT / "vendor" / "arm" / Path(ARM_A64_ARCHIVE_URL).name
LOCAL_ARM_A64_ARCHIVES = [
    ARM_A64_ARCHIVE_CACHE,
    # Vendored 2025-09 release, checked into the repo for offline builds.
    _REPO_ROOT / "vendor" / "arm" / "AARCHMRS_OPENSOURCE_A_profile_FAT-2025-09_ASL0.tar.gz",
    _REPO_ROOT / "vendor" / "arm" / "AARCHMRS_BSD.tar.gz",
    _REPO_ROOT / "vendor" / "arm" / "aarchmrs_bsd.tar.gz",
    _REPO_ROOT / "vendor" / "arm" / "aarchmrs-bsd.tar.gz",
]
# A64 ISA XML release (Exploration Tools). The XML <desc><brief> paragraphs
# are the one-line summaries for rows with no sourced text (issue #65 m1).
ARM_A64_XML_URL = (
    "https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/"
    "ISA_A64/ISA_A64_xml_A_profile-2026-06.tar.gz"
)
ARM_A64_XML_SIZE = 34182920
ARM_A64_XML_SHA256 = "63a01a1696483bbe2edfef9e0f0cd053d6c1c619ec0587876cb7a60bb344f354"
ARM_A64_XML_CACHE = _REPO_ROOT / "vendor" / "arm" / Path(ARM_A64_XML_URL).name
LOCAL_ARM_A64_XML_ARCHIVES = [ARM_A64_XML_CACHE]
LOCAL_RISCV_UNIFIED_DB_JSONS = [
    _REPO_ROOT / "vendor" / "riscv" / "unified_db_bundle.json",
    _REPO_ROOT / "vendor" / "riscv" / "unified_db_instructions.json",
    _REPO_ROOT / "vendor" / "riscv" / "instructions.json",
]
LOCAL_RISCV_RVV_INTRINSICS_JSONS = [
    _REPO_ROOT / "vendor" / "riscv" / "rvv_intrinsics_bundle.json",
    _REPO_ROOT / "vendor" / "riscv" / "rvv_intrinsics.json",
    _REPO_ROOT / "vendor" / "riscv" / "intrinsics.json",
]
LOCAL_RISCV_DOCS_JSONS = [
    _REPO_ROOT / "vendor" / "riscv" / "docs_pages.json",
]
LOCAL_PTX_BUNDLES = [
    _REPO_ROOT / "vendor" / "ptx" / "ptx_instructions.json",
]
LOCAL_PTX_DOC_HTMLS = [
    _REPO_ROOT / "vendor" / "ptx" / "parallel-thread-execution.html",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fetch_text(url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=20.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _read_local_intel_archive() -> tuple[str, SourceVersion] | None:
    for archive_path in LOCAL_INTEL_ARCHIVES:
        if not archive_path.exists():
            continue
        with zipfile.ZipFile(archive_path) as zf:
            text = _extract_zip_text(zf, "data.js")
        return text, SourceVersion(
            source="intel-intrinsics-guide",
            version=f"offline-package:{archive_path.stem}",
            fetched_at=now_iso(),
            url=str(archive_path),
        )
    return None


def _read_local_intel_cache_archive() -> tuple[str, SourceVersion] | None:
    for archive_path in LOCAL_INTEL_CACHE_ARCHIVES:
        if not archive_path.exists():
            continue
        try:
            with zipfile.ZipFile(archive_path) as zf:
                text = _extract_zip_text(zf, "data.js")
                fetched_at = json.loads(zf.read("SOURCE.json"))["fetched_at"]
        except Exception:
            continue
        return text, SourceVersion(
            source="intel-intrinsics-guide",
            version=f"upstream-cache:{fetched_at}",
            fetched_at=now_iso(),
            url=str(archive_path),
        )
    return None


def fetch_uops_xml() -> tuple[str | Path, SourceVersion]:
    for xml_path in LOCAL_UOPS_XMLS:
        if xml_path.exists():
            return xml_path, SourceVersion(
                source="uops.info",
                version=f"local-xml:{xml_path.name}",
                fetched_at=now_iso(),
                url=str(xml_path),
            )
    try:
        text = _fetch_text(UOPS_XML_URL)
    except Exception as exc:
        raise SourceUnavailableError(
            f"uops.info instruction XML is unreachable at {UOPS_XML_URL}"
        ) from exc
    return text, SourceVersion(
        source="uops.info",
        version="live",
        fetched_at=now_iso(),
        url=UOPS_XML_URL,
    )


def _files_asset_links(html_text: str) -> list[str]:
    # "files/" asset links: the run up to a quote or backslash, kept when
    # some prefix of the run ends in .js, .json or .xml.
    # shortcut: "files/x.json" keeps the full name; the removed regex
    # truncated it to "files/x.js" (live Intel HTML has no such links).
    links: list[str] = []
    marker = "files/"
    start = html_text.find(marker)
    while start != -1:
        end = start + len(marker)
        while end < len(html_text) and html_text[end] not in "'\"\\":
            end += 1
        run = html_text[start + len(marker) : end]
        best = 0
        for ext in (".json", ".js", ".xml"):
            index = run.rfind(ext)
            if index >= 1 and index + len(ext) > best:
                best = index + len(ext)
        if best:
            links.append(html_text[start : start + len(marker) + best])
            start = html_text.find(marker, end)
        else:
            start = html_text.find(marker, start + 1)
    return links


def fetch_intel_data() -> tuple[str, SourceVersion]:
    local_archive = _read_local_intel_archive()
    if local_archive is not None:
        return local_archive

    for url in INTEL_CANDIDATE_DATA_URLS:
        try:
            text = _fetch_text(url)
            return text, SourceVersion(
                source="intel-intrinsics-guide",
                version="live",
                fetched_at=now_iso(),
                url=url,
            )
        except Exception:
            pass
    try:
        html = _fetch_text(INTEL_INDEX_URL)
        matches = _files_asset_links(html)
        for match in matches:
            candidate = f"https://www.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide/{match}"
            try:
                text = _fetch_text(candidate)
                return text, SourceVersion(
                    source="intel-intrinsics-guide",
                    version="live",
                    fetched_at=now_iso(),
                    url=candidate,
                )
            except Exception:
                pass
    except Exception:
        pass

    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(INTEL_OFFLINE_ZIP_URL)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    if name.endswith("data.js") or name.endswith("data.json"):
                        text = zf.read(name).decode("utf-8", "replace")
                        return text, SourceVersion(
                            source="intel-intrinsics-guide",
                            version="offline-zip-download",
                            fetched_at=now_iso(),
                            url=INTEL_OFFLINE_ZIP_URL,
                        )
    except Exception:
        pass

    cached = _read_local_intel_cache_archive()
    if cached is not None:
        return cached

    raise SourceUnavailableError(
        "Intel intrinsics data is unreachable (network + local archives + upstream cache all failed). "
        "Install llvm-mca + a working network, or run `simdref update` to "
        "use the pre-built release catalog."
    )


def _read_local_text(
    paths: list[Path], source: str, version_prefix: str
) -> tuple[str, SourceVersion] | None:
    for text_path in paths:
        if not text_path.exists():
            continue
        return text_path.read_text(), SourceVersion(
            source=source,
            version=f"{version_prefix}:{text_path.name}",
            fetched_at=now_iso(),
            url=str(text_path),
        )
    return None


def _riscv_missing_semantics_urls(payload: dict) -> list[str]:
    instructions = payload.get("instructions") or payload.get("records") or []
    urls: list[str] = []
    for item in instructions:
        if not isinstance(item, dict):
            continue
        sections = item.get("description") or item.get("doc_sections") or item.get("sections") or {}
        if not isinstance(sections, dict):
            sections = {}
        has_description = str(sections.get("Description") or "").strip()
        has_operation = str(sections.get("Operation") or "").strip()
        if has_description and has_operation:
            continue
        url = str(item.get("url") or item.get("reference_url") or "").strip()
        if url.startswith("https://docs.riscv.org/"):
            urls.append(url)
    return list(dict.fromkeys(urls))


def _augment_riscv_unified_db_payload_with_docs(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, dict):
        return text
    if payload.get("docs_pages"):
        return text

    docs_pages: dict[str, str] = {}
    local_docs = _read_local_text(LOCAL_RISCV_DOCS_JSONS, "docs.riscv.org", "local-json")
    if local_docs is not None:
        try:
            parsed = json.loads(local_docs[0])
            if isinstance(parsed, dict):
                docs_pages.update(
                    {
                        str(key).strip(): str(value).strip()
                        for key, value in parsed.items()
                        if str(key).strip() and str(value).strip()
                    }
                )
        except Exception:
            pass

    for url in _riscv_missing_semantics_urls(payload):
        if url in docs_pages:
            continue
        base = url.split("#", 1)[0]
        if base in docs_pages:
            docs_pages[url] = docs_pages[base]
            continue
        try:
            page = _fetch_text(base)
        except Exception:
            continue
        if page.strip():
            docs_pages[base] = page
            docs_pages[url] = page

    if not docs_pages:
        return text
    payload["docs_pages"] = docs_pages
    return json.dumps(payload)


def _arm_acle_bundle_payload(
    intrinsics_csv: str,
    classification_csv: str,
    acle_markdown: str = "",
    neon_markdown: str = "",
) -> str:
    return json.dumps(
        {
            "format": "acle-neon-csv-v1",
            "intrinsics_csv": intrinsics_csv,
            "classification_csv": classification_csv,
            "acle_markdown": acle_markdown,
            "neon_markdown": neon_markdown,
        }
    )


def _arm_intrinsics_bundle_payload(
    intrinsics_json: str, operations_json: str, examples_json: str = "[]"
) -> str:
    return json.dumps(
        {
            "format": "arm-intrinsics-json-v1",
            "intrinsics_json": intrinsics_json,
            "operations_json": operations_json,
            "examples_json": examples_json,
        }
    )


def _extract_zip_text(zf: zipfile.ZipFile, suffix: str) -> str:
    for name in zf.namelist():
        if name.endswith(suffix):
            return zf.read(name).decode("utf-8", "replace")
    raise KeyError(suffix)


def _extract_zip_text_by_match(zf: zipfile.ZipFile, predicate) -> tuple[str, str]:
    for name in zf.namelist():
        if predicate(name):
            return zf.read(name).decode("utf-8", "replace"), name
    raise KeyError("zip member")


def _extract_tar_text_by_match(tf: tarfile.TarFile, predicate) -> tuple[str, str]:
    for member in tf.getmembers():
        if not member.isfile() or not predicate(member.name):
            continue
        extracted = tf.extractfile(member)
        if extracted is None:
            continue
        return extracted.read().decode("utf-8", "replace"), member.name
    raise KeyError("tar member")


def _read_local_arm_acle_archive() -> tuple[str, SourceVersion] | None:
    for archive_path in LOCAL_ARM_ACLE_ARCHIVES:
        if not archive_path.exists():
            continue
        with zipfile.ZipFile(archive_path) as zf:
            return _arm_acle_bundle_payload(
                _extract_zip_text(zf, ARM_ACLE_NEON_DB_PATH),
                _extract_zip_text(zf, ARM_ACLE_NEON_CLASSIFICATION_PATH),
                _extract_zip_text(zf, ARM_ACLE_MAIN_MD_PATH),
                _extract_zip_text(zf, ARM_ACLE_NEON_MD_PATH),
            ), SourceVersion(
                source="arm-acle",
                version=f"archive:{archive_path.name}",
                fetched_at=now_iso(),
                url=str(archive_path),
            )
    return None


def _read_local_arm_intrinsics_bundle() -> tuple[str, SourceVersion] | None:
    intrinsics = _read_local_text(LOCAL_ARM_INTRINSICS_JSONS, "arm-intrinsics-site", "local-json")
    operations = _read_local_text(LOCAL_ARM_OPERATIONS_JSONS, "arm-intrinsics-site", "local-json")
    if intrinsics is None or operations is None:
        return None
    examples = _read_local_text(LOCAL_ARM_EXAMPLES_JSONS, "arm-intrinsics-site", "local-json")
    payload = _arm_intrinsics_bundle_payload(
        intrinsics[0],
        operations[0],
        examples[0] if examples is not None else "[]",
    )
    return payload, SourceVersion(
        source="arm-intrinsics-site",
        version="local-json-bundle",
        fetched_at=now_iso(),
        url=str(LOCAL_ARM_INTRINSICS_JSONS[0].parent),
    )


def _looks_like_arm_instruction_json(path: str) -> bool:
    lowered = path.casefold()
    if not lowered.endswith(".json"):
        return False
    basename = lowered.rsplit("/", 1)[-1]
    if basename == "instructions.json" and "/" not in lowered.strip("/"):
        return True
    if "schema" in lowered or "register" in lowered or "system" in lowered:
        return False
    return "instruction" in lowered and ("a64" in lowered or basename == "instructions.json")


def _arm_instruction_bundle_payload(instructions_json: str) -> str:
    return json.dumps(
        {
            "format": "arm-aarchmrs-instructions-v1",
            "instructions_json": instructions_json,
        }
    )


def _read_local_arm_instruction_archive() -> tuple[str, SourceVersion] | None:
    for archive_path in LOCAL_ARM_A64_ARCHIVES:
        if not archive_path.exists():
            continue
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                payload, member_name = _extract_zip_text_by_match(
                    zf, _looks_like_arm_instruction_json
                )
        else:
            with tarfile.open(archive_path, "r:*") as tf:
                payload, member_name = _extract_tar_text_by_match(
                    tf, _looks_like_arm_instruction_json
                )
        return _arm_instruction_bundle_payload(payload), SourceVersion(
            source="arm-a64",
            version=f"archive:{archive_path.name}:{member_name}",
            fetched_at=now_iso(),
            url=str(archive_path),
        )
    return None


def refresh_local_arm_a64_archive(
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download the AARCHMRS BSD archive into ``vendor/arm/`` if not already present.

    The archive is large (~hundreds of MB) and pinned to a specific
    release via its URL. The cache filename is derived from the URL's
    basename, so bumping ``ARM_A64_ARCHIVE_URL`` automatically triggers a
    fresh download without requiring the user to clean ``vendor/arm/``.

    ``on_progress(bytes_downloaded, total_bytes)`` fires on each chunk;
    ``total_bytes`` is ``None`` if the server omits ``Content-Length``.
    """
    target = ARM_A64_ARCHIVE_CACHE
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with httpx.Client(follow_redirects=True, timeout=600.0) as client:
        with client.stream("GET", ARM_A64_ARCHIVE_URL) as response:
            response.raise_for_status()
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if on_progress is not None:
                        on_progress(downloaded, total)
    tmp.replace(target)
    return target


def refresh_local_arm_a64_xml_archive(
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download the pinned A64 ISA XML release into ``vendor/arm/``.

    The cached file must match the pinned size and sha256; a download that
    fails either check is deleted and raises so the cache never keeps a
    partial or a swapped file.
    """
    target = ARM_A64_XML_CACHE
    if target.exists() and target.stat().st_size == ARM_A64_XML_SIZE:
        if _sha256(target) == ARM_A64_XML_SHA256:
            return target
    tmp = target.with_suffix(target.suffix + ".part")
    with httpx.Client(follow_redirects=True, timeout=600.0) as client:
        with client.stream("GET", ARM_A64_XML_URL) as response:
            response.raise_for_status()
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if on_progress is not None:
                        on_progress(downloaded, total)
    if tmp.stat().st_size != ARM_A64_XML_SIZE or _sha256(tmp) != ARM_A64_XML_SHA256:
        tmp.unlink(missing_ok=True)
        raise SourceUnavailableError("A64 ISA XML download failed the pinned size or sha256 check.")
    tmp.replace(target)
    return target


def a64xml_briefs_from_xml(text: str) -> dict[str, str]:
    """``<desc><brief>`` text of one A64 ISA XML file, keyed by its names.

    Keys are the instruction id, the title and heading without the ``-- A64``
    suffix, and every ``docvar mnemonic`` value, uppercased and with the XML's
    entity spelling kept (``CBB&lt;cc&gt;`` -> ``CBB&LT;CC&GT;``) so the map
    reproduces the release's own alias names.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # The release directory also holds .dtd and .css members that are
        # not standalone XML; they carry no briefs.
        return {}
    if root.tag != "instructionsection":
        return {}
    para = root.find("./desc/brief/para")
    if para is None:
        return {}
    brief = " ".join("".join(para.itertext()).split())
    if not brief:
        return {}
    title = (root.get("title") or "").strip()
    if title.endswith("A64"):
        head = title[: -len("A64")].rstrip()
        if head.endswith("--"):
            title = head[:-2].rstrip()
    names = [
        root.get("id") or "",
        title,
        (root.findtext("heading") or "").strip(),
    ]
    names += [
        docvar.get("value") or ""
        for docvar in root.iter("docvar")
        if docvar.get("key") == "mnemonic"
    ]
    keys = {html.escape(name, quote=False).upper() for name in names if name.strip()}
    return dict.fromkeys(keys, html.escape(brief, quote=False))


def parse_a64xml_briefs(archive: Path) -> tuple[dict[str, str], dict[str, str]]:
    """A64 ISA XML briefs keyed by file name and, unambiguous ones, by name.

    Only the release directory the pinned URL names is read, so a later
    release never mixes its files in. Returns ``(by_file, by_name)``.
    ``by_file`` keys each brief by its XML file name, uppercased; a row's
    ``operation_id`` joins here. ``by_name`` keys a name only when every XML
    file naming it gives the same brief. A name several files describe
    differently carries no entry, so no lookup ever falls back to file order.
    """
    release = Path(ARM_A64_XML_URL).name.removesuffix(".tar.gz")
    prefix = f"{release}/"
    by_file: dict[str, str] = {}
    by_name: dict[str, str] = {}
    conflicts: set[str] = set()
    with tarfile.open(archive, "r:*") as tf:
        names = sorted(
            member.name
            for member in tf.getmembers()
            if member.name.startswith(prefix) and "/" not in member.name[len(prefix) :]
        )
        for name in names:
            fh = tf.extractfile(name)
            if fh is None:
                continue
            briefs = a64xml_briefs_from_xml(fh.read().decode("utf-8"))
            if not briefs:
                continue
            # One file, one brief; every key of the file maps to it.
            brief = next(iter(briefs.values()))
            by_file.setdefault(name[len(prefix) :].removesuffix(".xml").upper(), brief)
            for key in briefs:
                if key in conflicts:
                    continue
                if by_name.setdefault(key, brief) != brief:
                    del by_name[key]
                    conflicts.add(key)
    return by_file, by_name


def read_local_arm_a64_xml_briefs() -> tuple[dict[str, str], dict[str, str]]:
    for archive in LOCAL_ARM_A64_XML_ARCHIVES:
        if archive.exists():
            return parse_a64xml_briefs(archive)
    return {}, {}


def refresh_local_arm_intrinsics_bundle(
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    target_dir = LOCAL_ARM_INTRINSICS_JSONS[0].parent
    target_dir.mkdir(parents=True, exist_ok=True)
    downloads = [
        (ARM_INTRINSICS_JSON_URL, LOCAL_ARM_INTRINSICS_JSONS[0]),
        (ARM_INTRINSICS_OPERATIONS_JSON_URL, LOCAL_ARM_OPERATIONS_JSONS[0]),
        (ARM_INTRINSICS_EXAMPLES_JSON_URL, LOCAL_ARM_EXAMPLES_JSONS[0]),
    ]
    written: list[Path] = []
    total = len(downloads)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for url, path in downloads:
            response = client.get(url)
            response.raise_for_status()
            path.write_text(response.text)
            written.append(path)
            if on_progress is not None:
                on_progress(len(written), total)
    return written


def fetch_arm_acle_data() -> tuple[str, SourceVersion]:
    local_intrinsics_bundle = _read_local_arm_intrinsics_bundle()
    if local_intrinsics_bundle is not None:
        return local_intrinsics_bundle
    local_archive = _read_local_arm_acle_archive()
    if local_archive is not None:
        return local_archive
    local = _read_local_text(LOCAL_ARM_ACLE_JSONS, "arm-acle", "local-json")
    if local is not None:
        return local
    try:
        payload = _arm_intrinsics_bundle_payload(
            _fetch_text(ARM_INTRINSICS_JSON_URL),
            _fetch_text(ARM_INTRINSICS_OPERATIONS_JSON_URL),
            _fetch_text(ARM_INTRINSICS_EXAMPLES_JSON_URL),
        )
        return payload, SourceVersion(
            source="arm-intrinsics-site",
            version="live-json",
            fetched_at=now_iso(),
            url=ARM_INTRINSICS_JSON_URL,
        )
    except Exception:
        pass
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(ARM_ACLE_ARCHIVE_URL)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                payload = _arm_acle_bundle_payload(
                    _extract_zip_text(zf, ARM_ACLE_NEON_DB_PATH),
                    _extract_zip_text(zf, ARM_ACLE_NEON_CLASSIFICATION_PATH),
                    _extract_zip_text(zf, ARM_ACLE_MAIN_MD_PATH),
                    _extract_zip_text(zf, ARM_ACLE_NEON_MD_PATH),
                )
            return payload, SourceVersion(
                source="arm-acle",
                version="archive-download",
                fetched_at=now_iso(),
                url=ARM_ACLE_ARCHIVE_URL,
            )
    except Exception:
        pass
    raise SourceUnavailableError("Arm ACLE intrinsics data is unreachable.")


def fetch_arm_a64_data() -> tuple[str, SourceVersion]:
    try:
        # Best effort: rows without sourced text take the A64 XML briefs.
        refresh_local_arm_a64_xml_archive()
    except Exception:
        pass
    local = _read_local_text(LOCAL_ARM_A64_JSONS, "arm-a64", "local-json")
    if local is not None:
        return local
    archive = _read_local_arm_instruction_archive()
    if archive is not None:
        return archive
    try:
        refresh_local_arm_a64_archive()
    except Exception as exc:
        raise SourceUnavailableError(f"Arm A64 instruction data is unreachable: {exc}") from exc
    archive = _read_local_arm_instruction_archive()
    if archive is not None:
        return archive
    raise SourceUnavailableError("Arm A64 instruction data is unreachable.")


def fetch_riscv_unified_db_data() -> tuple[str, SourceVersion]:
    local = _read_local_text(LOCAL_RISCV_UNIFIED_DB_JSONS, "riscv-unified-db", "local-json")
    if local is not None:
        return _augment_riscv_unified_db_payload_with_docs(local[0]), local[1]
    for url in RISCV_UNIFIED_DB_CANDIDATE_URLS:
        try:
            text = _fetch_text(url)
            return _augment_riscv_unified_db_payload_with_docs(text), SourceVersion(
                source="riscv-unified-db",
                version="live",
                fetched_at=now_iso(),
                url=url,
            )
        except Exception:
            pass
    raise SourceUnavailableError("RISC-V unified-db data is unreachable.")


def _build_rvv_bundle_locally() -> None:
    """Run tools/vendor_riscv_sources.py to clone upstream and write the vendor/riscv bundles."""
    script = _REPO_ROOT / "tools" / "vendor_riscv_sources.py"
    bundle = _REPO_ROOT / "vendor" / "riscv" / "rvv_intrinsics_bundle.json"
    if not script.exists():
        # The wheel ships no tools/ directory, so an installed simdref cannot build the bundle.
        raise SourceUnavailableError(
            f"RISC-V RVV intrinsics bundle is missing at {bundle}. "
            "Make it from a source checkout with 'python tools/vendor_riscv_sources.py'."
        )
    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            timeout=600,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        stderr = getattr(exc, "stderr", None) or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        tail = "\n".join(stderr.splitlines()[-20:])
        raise SourceUnavailableError(
            "RISC-V RVV intrinsics data is unreachable: "
            f"tools/vendor_riscv_sources.py failed: {exc}\n{tail}"
        ) from exc


def fetch_riscv_rvv_intrinsics_data() -> tuple[str, SourceVersion]:
    local = _read_local_text(LOCAL_RISCV_RVV_INTRINSICS_JSONS, "rvv-intrinsic-doc", "local-json")
    if local is not None:
        return local
    # No local bundle: the dead static URLs are gone, so build it by running
    # the same vendor step CI uses (clones upstream, writes vendor/riscv).
    _build_rvv_bundle_locally()
    local = _read_local_text(LOCAL_RISCV_RVV_INTRINSICS_JSONS, "rvv-intrinsic-doc", "local-json")
    if local is not None:
        return local
    raise SourceUnavailableError("RISC-V RVV intrinsics data is unreachable.")


def fetch_ptx_isa_data() -> tuple[str, SourceVersion]:
    local = _read_local_text(LOCAL_PTX_BUNDLES, "ptx-isa", "local-bundle")
    if local is not None:
        return local
    local_doc = _read_local_text(LOCAL_PTX_DOC_HTMLS, "ptx-isa", "local-doc")
    if local_doc is not None:
        return local_doc
    try:
        text = _fetch_text(PTX_ISA_DOC_URL)
    except Exception as exc:
        raise SourceUnavailableError(
            f"NVIDIA PTX ISA reference is unreachable at {PTX_ISA_DOC_URL}"
        ) from exc
    return text, SourceVersion(
        source="ptx-isa",
        version="live",
        fetched_at=now_iso(),
        url=PTX_ISA_DOC_URL,
    )
