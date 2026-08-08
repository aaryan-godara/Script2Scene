"""Persistent transcription cache, keyed by AUDIO IDENTITY + transcription config.

Unlike the legacy job-local ``transcription.json`` cache, this cache lives
outside any single job (``workspace/cache/transcriptions/<cache_key>/``) so a
successful recovery survives across generations. Two jobs that use the same
narration bytes and the same transcription configuration reuse the same cached
transcription (and any recovered improvements) instead of re-running Whisper.

Cache identity is deliberately NOT based on filename, project name or job id.
It depends on:

    * audio SHA-256 (byte identity of the narration)
    * a deterministic, serialized transcription configuration (provider, model,
      no_speech_threshold, chunking, tail recovery, failed-region recovery)
    * a transcription schema/CACHE version

The cache is stored under ``transcription.json`` with associated
``metadata.json`` and ``recovery_log.json``. Updates are atomic: we write to a
unique temporary file inside the same directory and ``os.replace`` it into
place, so an interrupted write never leaves a truncated cache behind. A small
cross-platform file lock serializes concurrent writers.

Design invariants:
    * every component of the identity is deterministic and cross-platform;
    * never reuse a cache whose config no longer matches;
    * never reuse a cache whose audio bytes differ;
    * writes are atomic (no half-written JSON);
    * the raw ASR evidence (Whisper words) is never rewritten by numeric
      alignment logic — only recovery may add/merge words.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .provider_base import TranscriptionResult


class TranscriptionCacheError(RuntimeError):
    pass


def audio_sha256(path: str | Path) -> str:
    """Deterministic SHA-256 of an audio file for cache identity."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Schema/version for the cache layout + the transcription identity. Users who
# change the on-disk layout or the recovery algorithm that contributes words to
# the cache must bump one of these so old caches are not silently reused.
TRANSCRIPTION_CACHE_SCHEMA_VERSION = "1.0"
# Recovery contributes words to the cache; changing the recovery formula can
# change results, so it participates in the identity.
RECOVERY_ALGORITHM_VERSION = "1.0"


class TranscriptionIdentityConfig(BaseModel):
    """The transcription-affecting configuration that defines cache identity.

    Values here MUST only be things that change the WORD content of a
    transcription (Whisper provider/model/threshold, chunking, tail recovery,
    failed-region recovery). Pure alignment-side settings (review threshold,
    numeric normalization) do NOT belong here because they never change what
    Whisper returned.
    """

    provider: str = "stable_whisper"
    model: str = "base"
    no_speech_threshold: float = 0.9

    chunking_enabled: bool = True
    chunk_duration_seconds: float = 180.0
    overlap_seconds: float = 10.0
    long_audio_threshold_seconds: float = 300.0
    dedup_time_tolerance: float = 1.5

    tail_recovery_enabled: bool = True
    tail_gap_trigger_seconds: float = 2.0
    tail_recovery_context_seconds: float = 90.0

    # Failed-region recovery affects the transcript that lands in the cache so
    # it is part of the identity.
    failed_region_recovery_enabled: bool = True
    max_recovery_passes: int = 2
    context_before_seconds: float = 15.0
    context_after_seconds: float = 15.0
    min_window_seconds: float = 30.0
    preferred_window_seconds: float = 60.0
    max_window_seconds: float = 90.0
    acoustic_check_enabled: bool = True


def _deterministic_json(value: Any) -> str:
    """Serializes nested dict/list/scalar deterministically (sorted keys)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def transcription_cache_key(audio_sha256: str,
                            identity: TranscriptionIdentityConfig,
                            schema_version: str = TRANSCRIPTION_CACHE_SCHEMA_VERSION,
                            recovery_version: str = RECOVERY_ALGORITHM_VERSION) -> str:
    """Deterministic cache key from audio bytes + serialized transcription config."""
    payload = _deterministic_json({
        "audio_sha256": audio_sha256,
        "identity": json.loads(identity.model_dump_json()),
        "schema_version": schema_version,
        "recovery_version": recovery_version,
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PersistentTranscriptionCache:
    """Persistent, identity-keyed, atomic transcription cache."""

    def __init__(self, cache_root: Path | str,
                 identity: Optional[TranscriptionIdentityConfig] = None,
                 schema_version: str = TRANSCRIPTION_CACHE_SCHEMA_VERSION,
                 recovery_version: str = RECOVERY_ALGORITHM_VERSION,
                 lock_timeout_ms: int = 60000):
        self.cache_root = Path(cache_root)
        self.identity = identity or TranscriptionIdentityConfig()
        self.schema_version = schema_version
        self.recovery_version = recovery_version
        self.lock_timeout_ms = lock_timeout_ms

    # ------------------------------------------------------------------ paths
    def entry_dir(self, audio_sha256: str) -> Path:
        key = transcription_cache_key(
            audio_sha256, self.identity, self.schema_version, self.recovery_version)
        return self.cache_root / key

    # ------------------------------------------------------------------ lookup
    def load(self, audio_sha256: str) -> Optional[TranscriptionResult]:
        """Returns cached transcription if it exists and matches audio identity.

        The cache DIRECTORY key already encodes the full transcription config +
        audio SHA, so an unrelated config or different audio resolves to a
        different key and ``load`` returns None. We additionally verify the
        stored audio SHA-256 matches the requested identity as a defense in depth.
        """
        entry = self.entry_dir(audio_sha256)
        if not entry.is_dir():
            return None
        from .transcription_cache import load_transcription
        try:
            result = load_transcription(entry / "transcription.json", audio_path=None)
        except Exception:  # noqa: BLE001 - any cache defect -> treat as miss
            return None
        if result.audio_sha256 != audio_sha256:
            return None
        return result

    def has(self, audio_sha256: str) -> bool:
        return self.load(audio_sha256) is not None

    # ------------------------------------------------------------------ write
    def save(self, result: TranscriptionResult, audio_sha256: str,
             source: str, metadata_extra: Optional[Dict] = None,
             recovery_log: Optional[Dict] = None):
        """Atomically persists a transcription (and metadata) for this identity."""
        entry = self.entry_dir(audio_sha256)
        entry.mkdir(parents=True, exist_ok=True)
        payload_path = entry / "transcription.json"
        metadata_path = entry / "metadata.json"
        log_path = entry / "recovery_log.json"

        with self._lock(entry):
            result.audio_sha256 = audio_sha256
            metadata = self._build_metadata(result, source, metadata_extra)
            _atomic_write_text(payload_path, result.model_dump_json(indent=2))
            _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
            _atomic_write_text(log_path, json.dumps(recovery_log or {}, indent=2))
        return payload_path

    def _build_metadata(self, result: TranscriptionResult, source: str,
                        metadata_extra: Optional[Dict]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "audio_sha256": result.audio_sha256,
            "provider": result.provider,
            "model": result.model,
            "transcription_config": self.identity.model_dump(),
            "cache_schema_version": self.schema_version,
            "recovery_version": self.recovery_version,
            "cache_key": transcription_cache_key(
                result.audio_sha256, self.identity,
                self.schema_version, self.recovery_version),
            "created_at": None,
            "updated_at": _now_iso(),
            "base_word_count": len(result.words),
            "current_word_count": len(result.words),
            "recovery_applied": bool(metadata_extra and metadata_extra.get("recovery_applied")),
            "recovery_algorithm_version": self.recovery_version,
            "recovery_passes": (metadata_extra or {}).get("recovery_passes", 0),
            "recovered_regions": (metadata_extra or {}).get("recovered_regions", []),
            "recovered_scene_ids": (metadata_extra or {}).get("recovered_scene_ids", []),
            "tail_recovery_triggered": bool((metadata_extra or {}).get("tail_recovery_triggered")),
            "numeric_validation_version": (metadata_extra or {}).get("numeric_validation_version"),
            "source": source,
        }
        return meta

    # ------------------------------------------------------------------ lock
    def _lock(self, entry: Path):
        return _CacheLock(entry / ".lock", timeout_ms=self.lock_timeout_ms)


class _CacheLock:
    """A small cross-platform file lock used to serialize concurrent writers.

    Uses msvcrt.locking on Windows and fcntl.flock elsewhere. Falls back to an
    atomic O_EXCL lockfile marker if neither is available. The lock only guards
    the atomic-write section; because writes are also atomic via os.replace the
    worst failure mode is a slightly stale read, never a corrupted cache.
    """

    def __init__(self, path: Path, timeout_ms: int = 60000):
        self.path = path
        self.timeout_ms = timeout_ms
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_ms / 1000.0
        while True:
            try:
                self._fh = open(self.path, "a+b")
                try:
                    import msvcrt  # Windows
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return True
                except ImportError:
                    try:
                        import fcntl  # POSIX
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except ImportError:
                        # Fallback: exclusive-create sentinel.
                        return self._exclusive_fallback()
            except OSError:
                pass
            if time.time() >= deadline:
                return False
            time.sleep(0.05)

    def _lock_fallback_path(self):
        return self.path.with_suffix(self.path.suffix + ".excl")

    def _exclusive_fallback(self):
        lock = self._lock_fallback_path()
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def __enter__(self):
        if not self.acquire():
            raise TranscriptionCacheError(
                f"Could not acquire persistent cache lock: {self.path}")
        return self

    def __exit__(self, *exc):
        try:
            import msvcrt
            self._fh.seek(0)
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        except ImportError:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except ImportError:
                self._exclusive_release()
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._exclusive_release()
        return False

    def _exclusive_release(self):
        lock = self._lock_fallback_path()
        try:
            Path(lock).unlink()
        except OSError:
            pass


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (unique temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_cached_metadata(entry_dir: Path) -> Dict[str, Any]:
    """Best-effort read of metadata.json (never raises)."""
    p = Path(entry_dir) / "metadata.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
