"""Transcription cache safety.

A transcription.json may only be reused when the audio it was built from is
byte-identical to the current narration. Identity is established via the
audio's SHA-256, which is stored in the transcription metadata at write time
and verified at load time. Merely existing is never enough to trust a cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .provider_base import TranscriptionResult


class TranscriptionCacheError(RuntimeError):
    pass


def audio_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_transcription(result: TranscriptionResult, path: str | Path,
                       audio_path: str | Path | None = None) -> Path:
    """Serializes a TranscriptionResult to transcription.json.

    Always stamps audio_sha256 so a later load can be validated. If the result
    already carries an audio_sha256 that matches the audio, it is kept.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path is not None:
        result.audio_sha256 = audio_sha256(audio_path)
    elif result.audio_sha256 is None:
        raise TranscriptionCacheError(
            "Cannot save transcription without an audio SHA-256 identity.")
    path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return path


def load_transcription(path: str | Path, audio_path: str | Path | None = None
                       ) -> TranscriptionResult:
    """Loads and optionally validates a cached transcription.

    Raises TranscriptionCacheError when the file is missing, malformed, or its
    audio SHA-256 does not match ``audio_path``.
    """
    path = Path(path)
    if not path.is_file():
        raise TranscriptionCacheError(f"Transcription cache not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = TranscriptionResult(**data)
    except Exception as e:  # noqa: BLE001 - any malformed cache is rejected
        raise TranscriptionCacheError(f"Cached transcription is malformed: {path}") from e

    if audio_path is not None:
        actual = audio_sha256(audio_path)
        cached = result.audio_sha256
        if not cached:
            raise TranscriptionCacheError(
                f"Cached transcription has no audio identity; refusing to reuse: {path}")
        if cached != actual:
            raise TranscriptionCacheError(
                f"Cached transcription does not match current narration "
                f"(audio changed). Refusing to reuse: {path}")
    return result


def transcription_is_current(path: str | Path, audio_path: str | Path) -> bool:
    """True only when a cache exists and its audio SHA-256 matches the audio."""
    try:
        load_transcription(path, audio_path)
        return True
    except TranscriptionCacheError:
        return False
