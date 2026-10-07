import struct

import numpy as np

from app.services.audio import FrameBuffer, mulaw_to_pcm16, pcm16_to_mulaw, strip_wav_header


def pcm(values):
    return np.array(values, dtype="<i2").tobytes()


def test_mulaw_reference_values():
    # ITU-T G.711 reference points
    assert pcm16_to_mulaw(pcm([0])) == b"\xff"
    assert pcm16_to_mulaw(pcm([32767])) == b"\x80"
    assert pcm16_to_mulaw(pcm([-32768])) == b"\x00"


def test_mulaw_roundtrip_error_is_small():
    t = np.arange(8000) / 8000
    wave = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    decoded = np.frombuffer(mulaw_to_pcm16(pcm16_to_mulaw(wave.tobytes())), dtype="<i2")
    # mu-law is logarithmic: error relative to amplitude stays within ~3%
    rel = np.abs(decoded.astype(int) - wave.astype(int)) / (np.abs(wave.astype(int)) + 64)
    assert rel.max() < 0.07
    assert len(decoded) == len(wave)


def test_odd_length_pcm_is_safe():
    assert len(pcm16_to_mulaw(b"\x00\x01\x02")) == 1


def test_strip_wav_header():
    body = b"\x01\x02" * 10
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    wav = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(body)) + b"WAVE"
    wav += b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
    assert strip_wav_header(wav) == body
    assert strip_wav_header(body) == body


def test_frame_buffer():
    fb = FrameBuffer(160)
    assert fb.push(b"a" * 100) == []
    frames = fb.push(b"b" * 300)
    assert [len(f) for f in frames] == [160, 160]
    assert len(fb.flush()) == 80
