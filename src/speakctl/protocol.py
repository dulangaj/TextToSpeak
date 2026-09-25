"""Binary framing for speakctl's server mode. Pure encode/decode, no I/O.

Every frame is a 9-byte big-endian header followed by `payload_len` bytes:

    type        u8    FrameType
    request_id  u32   chosen by the client; 0 is reserved for HELLO
    payload_len u32   length of the payload that follows

JSON payloads are UTF-8. AUDIO payloads are raw s16le mono PCM.
"""

from __future__ import annotations

import enum
import json
import struct
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

HEADER = struct.Struct("!BII")
HEADER_SIZE = HEADER.size
MAX_INBOUND_PAYLOAD = 1 << 20
PROTOCOL_VERSION = 1
ERROR_MESSAGE_LIMIT = 500


class FrameType(enum.IntEnum):
    # client -> server
    REQUEST = 0x01
    CANCEL = 0x02
    # server -> client
    HELLO = 0x10
    START = 0x11
    AUDIO = 0x12
    END = 0x13
    ERROR = 0x14


class ErrorCode:
    BAD_REQUEST = "bad_request"
    TOO_MANY_REQUESTS = "too_many_requests"
    CANCELLED = "cancelled"
    ENGINE_ERROR = "engine_error"
    PROTOCOL_ERROR = "protocol_error"


class ProtocolError(Exception):
    """The peer sent bytes that don't form a valid frame or payload."""


@dataclass(frozen=True)
class Frame:
    type: FrameType
    request_id: int
    payload: bytes = b""


@dataclass(frozen=True)
class RequestSpec:
    text: str
    voice: str
    speed: Optional[float] = None


def encode(frame: Frame) -> bytes:
    return HEADER.pack(int(frame.type), frame.request_id, len(frame.payload)) + frame.payload


def decode_header(data: bytes) -> Tuple[FrameType, int, int]:
    if len(data) != HEADER_SIZE:
        raise ProtocolError(f"header must be {HEADER_SIZE} bytes, got {len(data)}")
    type_byte, request_id, length = HEADER.unpack(data)
    try:
        frame_type = FrameType(type_byte)
    except ValueError:
        raise ProtocolError(f"unknown frame type 0x{type_byte:02x}") from None
    return frame_type, request_id, length


def read_frame(read_exact: Callable[[int], bytes], max_payload: int = MAX_INBOUND_PAYLOAD) -> Frame | None:
    """Read one frame. Returns None on a clean EOF between frames.

    `read_exact(n)` must return exactly n bytes, or fewer (typically b"") if the
    stream ended first.
    """
    header = read_exact(HEADER_SIZE)
    if not header:
        return None
    if len(header) < HEADER_SIZE:
        raise ProtocolError("connection closed mid-header")
    frame_type, request_id, length = decode_header(header)
    if length > max_payload:
        raise ProtocolError(f"payload of {length} bytes exceeds limit of {max_payload}")
    payload = read_exact(length) if length else b""
    if len(payload) < length:
        raise ProtocolError("connection closed mid-payload")
    return Frame(frame_type, request_id, payload)


def _json(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def request_frame(request_id: int, text: str, voice: str, speed: float | None = None) -> Frame:
    body = {"text": text, "voice": voice}
    if speed is not None:
        body["speed"] = speed
    return Frame(FrameType.REQUEST, request_id, _json(body))


def cancel_frame(request_id: int) -> Frame:
    return Frame(FrameType.CANCEL, request_id)


def hello_frame(info: dict) -> Frame:
    return Frame(FrameType.HELLO, 0, _json(info))


def start_frame(request_id: int, sample_rate: int) -> Frame:
    return Frame(FrameType.START, request_id, _json({"sample_rate": sample_rate, "channels": 1, "format": "s16le"}))


def audio_frame(request_id: int, pcm: bytes) -> Frame:
    return Frame(FrameType.AUDIO, request_id, pcm)


def end_frame(request_id: int) -> Frame:
    return Frame(FrameType.END, request_id)


def error_frame(request_id: int, code: str, message: str) -> Frame:
    return Frame(FrameType.ERROR, request_id, _json({"code": code, "message": message[:ERROR_MESSAGE_LIMIT]}))


def parse_json(frame: Frame) -> dict:
    try:
        obj = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON payload: {exc}") from None
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload must be an object")
    return obj


def parse_request(frame: Frame) -> RequestSpec:
    body = parse_json(frame)
    text = body.get("text")
    voice = body.get("voice")
    speed = body.get("speed")
    if not isinstance(text, str):
        raise ProtocolError("'text' must be a string")
    if not isinstance(voice, str):
        raise ProtocolError("'voice' must be a string")
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float))):
        raise ProtocolError("'speed' must be a number")
    return RequestSpec(text=text, voice=voice, speed=None if speed is None else float(speed))
