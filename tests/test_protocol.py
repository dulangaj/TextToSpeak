import io
import json
import struct

import pytest

from speakctl import protocol as p


def reader(data: bytes):
    buf = io.BytesIO(data)
    return lambda n: buf.read(n)


ALL_FRAMES = [
    p.request_frame(1, "Hello there.", "af_heart", 1.25),
    p.request_frame(2, "no speed", "am_michael"),
    p.cancel_frame(3),
    p.hello_frame({"protocol": 1, "sample_rate": 24000}),
    p.start_frame(4, 24000),
    p.audio_frame(5, b"\x01\x00\xff\x7f"),
    p.end_frame(6),
    p.error_frame(7, p.ErrorCode.ENGINE_ERROR, "bad things"),
]


@pytest.mark.parametrize("frame", ALL_FRAMES, ids=lambda f: f.type.name)
def test_round_trip(frame):
    data = p.encode(frame)
    assert len(data) == p.HEADER_SIZE + len(frame.payload)
    assert p.read_frame(reader(data)) == frame


def test_round_trip_back_to_back():
    data = b"".join(p.encode(f) for f in ALL_FRAMES)
    read = reader(data)
    assert [p.read_frame(read) for _ in ALL_FRAMES] == ALL_FRAMES
    assert p.read_frame(read) is None


def test_header_layout():
    data = p.encode(p.Frame(p.FrameType.AUDIO, 0x01020304, b"xy"))
    assert data == b"\x12\x01\x02\x03\x04\x00\x00\x00\x02xy"


def test_empty_payload():
    data = p.encode(p.end_frame(9))
    assert len(data) == 9
    assert p.read_frame(reader(data)) == p.Frame(p.FrameType.END, 9, b"")


def test_clean_eof_returns_none():
    assert p.read_frame(reader(b"")) is None


def test_truncated_header():
    with pytest.raises(p.ProtocolError):
        p.read_frame(reader(p.encode(p.end_frame(1))[:5]))


def test_truncated_payload():
    with pytest.raises(p.ProtocolError):
        p.read_frame(reader(p.encode(p.audio_frame(1, b"abcdef"))[:-2]))


def test_oversize_payload_rejected_before_reading():
    header = struct.pack("!BII", 0x01, 1, 1 << 30)
    with pytest.raises(p.ProtocolError, match="exceeds"):
        p.read_frame(reader(header))
    with pytest.raises(p.ProtocolError):
        p.read_frame(reader(p.encode(p.audio_frame(1, b"x" * 11))), max_payload=10)


def test_unknown_type():
    with pytest.raises(p.ProtocolError, match="unknown frame type"):
        p.read_frame(reader(struct.pack("!BII", 0x7F, 1, 0)))
    with pytest.raises(p.ProtocolError):
        p.decode_header(struct.pack("!BII", 0x00, 1, 0))


def test_parse_request():
    spec = p.parse_request(p.request_frame(1, "Hi.", "af_heart", 2))
    assert spec == p.RequestSpec("Hi.", "af_heart", 2.0)
    assert p.parse_request(p.request_frame(1, "Hi.", "af_heart")).speed is None


@pytest.mark.parametrize(
    "body",
    [
        {"voice": "af_heart"},
        {"text": 5, "voice": "af_heart"},
        {"text": "hi", "voice": 7},
        {"text": "hi"},
        {"text": "hi", "voice": "af_heart", "speed": "fast"},
        {"text": "hi", "voice": "af_heart", "speed": True},
        ["not", "an", "object"],
    ],
)
def test_parse_request_rejects(body):
    frame = p.Frame(p.FrameType.REQUEST, 1, json.dumps(body).encode())
    with pytest.raises(p.ProtocolError):
        p.parse_request(frame)


def test_parse_request_rejects_bad_json():
    with pytest.raises(p.ProtocolError):
        p.parse_request(p.Frame(p.FrameType.REQUEST, 1, b"{nope"))
    with pytest.raises(p.ProtocolError):
        p.parse_request(p.Frame(p.FrameType.REQUEST, 1, b"\xff\xfe"))


def test_error_frame_truncates_and_handles_newlines_and_unicode():
    message = "line one\nline two — naïve ☃ " + "x" * 1000
    frame = p.error_frame(3, p.ErrorCode.ENGINE_ERROR, message)
    body = p.parse_json(p.read_frame(reader(p.encode(frame))))
    assert body["code"] == "engine_error"
    assert len(body["message"]) == p.ERROR_MESSAGE_LIMIT
    assert body["message"] == message[: p.ERROR_MESSAGE_LIMIT]
