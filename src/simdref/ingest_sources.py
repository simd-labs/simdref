"""Source acquisition for intrinsics and instruction catalogs."""

from __future__ import annotations

import json
import io
import re
import tarfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx

from simdref.models import SourceVersion


class SourceUnavailableError(RuntimeError):
    """Raised when every candidate source for a feed is unreachable.

    The installer / CI surfaces this with a hint to use the pre-built
    release catalog via ``simdref update`` instead of ``--build``.
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
RISCV_RVV_INTRINSICS_CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/riscv-non-isa/riscv-rvv-intrinsic-doc/main/auto-generated/intrinsics.json",
    "https://raw.githubusercontent.com/riscv-non-isa/riscv-rvv-intrinsic-doc/main/generated/intrinsics.json",
    "https://raw.githubusercontent.com/riscv-non-isa/riscv-rvv-intrinsic-doc/main/intrinsics.json",
]
ARM_ACLE_NEON_DB_PATH = "tools/intrinsic_db/advsimd.csv"
ARM_ACLE_NEON_CLASSIFICATION_PATH = "tools/intrinsic_db/advsimd_classification.csv"
ARM_ACLE_MAIN_MD_PATH = "main/acle.md"
ARM_ACLE_NEON_MD_PATH = "neon_intrinsics/advsimd.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_INTEL_ARCHIVES = [
    _REPO_ROOT / "vendor" / "intel" / "Intel-Intrinsics-Guide-Offline-3.6.9.zip",
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
            text = zf.read("Intel Intrinsics Guide/files/data.js").decode("utf-8", "replace")
        return text, SourceVersion(
            source="intel-intrinsics-guide",
            version=f"offline-package:{archive_path.stem}",
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
        matches = re.findall(r'files/[^"\']+\.(?:js|json|xml)', html)
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

    raise SourceUnavailableError(
        "Intel intrinsics data is unreachable (network + local archives all failed). "
        "Install llvm-mca + a working network, or run `simdref update` (no --build) to "
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


def fetch_riscv_rvv_intrinsics_data() -> tuple[str, SourceVersion]:
    local = _read_local_text(LOCAL_RISCV_RVV_INTRINSICS_JSONS, "rvv-intrinsic-doc", "local-json")
    if local is not None:
        return local
    for url in RISCV_RVV_INTRINSICS_CANDIDATE_URLS:
        try:
            text = _fetch_text(url)
            return text, SourceVersion(
                source="rvv-intrinsic-doc",
                version="live",
                fetched_at=now_iso(),
                url=url,
            )
        except Exception:
            pass
    raise SourceUnavailableError("RISC-V RVV intrinsics data is unreachable.")
