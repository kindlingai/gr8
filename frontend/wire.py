"""PROTOCOL.md framing: u32 big-endian length + UTF-8 JSON object. Shared by the front end and the mock engine."""
import asyncio
import json
import struct

MAX_FRAME = 256 << 20
_HDR = struct.Struct('>I')


class FrameError(Exception):
    pass


def encode(obj: dict) -> bytes:
    body = json.dumps(obj, separators=(',', ':'), allow_nan=False).encode()
    if len(body) > MAX_FRAME:
        raise FrameError(f'frame too large: {len(body)} bytes')
    return _HDR.pack(len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict | None:
    """Next frame, or None on a clean EOF at a frame boundary."""
    try:
        hdr = await reader.readexactly(4)
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise FrameError('EOF inside frame header') from e
    (n,) = _HDR.unpack(hdr)
    if n > MAX_FRAME:
        raise FrameError(f'frame too large: {n} bytes')
    try:
        body = await reader.readexactly(n)
    except asyncio.IncompleteReadError as e:
        raise FrameError('EOF inside frame body') from e
    try:
        obj = json.loads(body)
    except ValueError as e:
        raise FrameError(f'bad JSON: {e}') from e
    if not isinstance(obj, dict) or not isinstance(obj.get('type'), str):
        raise FrameError('frame is not an object with a "type"')
    return obj


def write_frame(writer: asyncio.StreamWriter, obj: dict) -> None:
    writer.write(encode(obj))
