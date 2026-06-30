from __future__ import annotations

import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request

from tqdm import tqdm


_CHUNK = 256 * 1024
_UA = {"User-Agent": "gvoice/0.1"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def download(url: str, dest: Path, *, desc: str = "download", sha256: str | None = None) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        verify_sha256(dest, sha256)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 9):
        existing = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers=dict(_UA))
        if existing:
            req.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", resp.getcode())
                if status == 206:
                    total_header = resp.headers.get("Content-Range", "")
                    total = int(total_header.split("/")[-1]) if "/" in total_header else None
                    mode = "ab"
                else:
                    total = int(resp.headers.get("Content-Length", "0")) or None
                    existing = 0
                    mode = "wb"
                with tmp.open(mode) as f, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as bar:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(dest)
            verify_sha256(dest, sha256)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and tmp.exists():
                tmp.replace(dest)
                verify_sha256(dest, sha256)
                return dest
            if attempt == 8:
                raise
        except (urllib.error.URLError, OSError, TimeoutError, ConnectionError):
            if attempt == 8:
                raise
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"download failed: {url}")
