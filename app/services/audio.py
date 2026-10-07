"""Audio helpers for telephony.

Twilio Media Streams carry 8 kHz, mono, 8-bit G.711 mu-law audio in base64.
Python 3.13 removed `audioop`, so mu-law encoding is implemented with numpy
(vectorised, ~microseconds per 20 ms frame).
"""

import struct

import numpy as np

TWILIO_SAMPLE_RATE = 8000
FRAME_MS = 20
MULAW_FRAME_BYTES = TWILIO_SAMPLE_RATE * FRAME_MS // 1000  # 160 bytes = 20 ms

_BIAS = 0x84
_CLIP = 32635


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Encode signed 16-bit little-endian PCM to G.711 mu-law."""
    if not pcm:
        return b""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    sign = np.where(samples < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(samples), _CLIP) + _BIAS
    # exponent = position of highest set bit above bit 7 (0..7)
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    mulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return mulaw.astype(np.uint8).tobytes()


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 mu-law to signed 16-bit PCM (used in tests/tools)."""
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa << 3) + _BIAS) << exponent
    sample = np.where(sign != 0, _BIAS - magnitude, magnitude - _BIAS)
    return sample.astype("<i2").tobytes()


def strip_wav_header(data: bytes) -> bytes:
    """Return raw sample bytes if `data` is a RIFF/WAV container, else unchanged."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack("<I", data[offset + 4 : offset + 8])
        if chunk_id == b"data":
            return data[offset + 8 :]
        offset += 8 + size + (size & 1)
    return b""


class FrameBuffer:
    """Accumulates bytes and yields fixed-size frames (Twilio likes 20 ms)."""

    def __init__(self, frame_bytes: int = MULAW_FRAME_BYTES):
        self.frame_bytes = frame_bytes
        self._buf = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        frames = []
        while len(self._buf) >= self.frame_bytes:
            frames.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return frames

    def flush(self) -> bytes:
        rest = bytes(self._buf)
        self._buf.clear()
        return rest

    def clear(self) -> None:
        self._buf.clear()


def frames_duration_s(num_bytes: int) -> float:
    return num_bytes / TWILIO_SAMPLE_RATE
