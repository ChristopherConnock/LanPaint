"""Read and write LanPaint mask metadata in MP4 files via PyAV.

The metadata tag ``lanpaint-mask`` holds a UTF-8 JSON payload:
{
  "version": 1,
  "video": "<source video filename>",
  "fps": <number>,
  "keyframes": {"<frame_idx>": "<base64 PNG (no data: prefix)>", ...},
  "audio_intervals": [{"start": <float>, "end": <float>}, ...]
}

Keyframes are base64-encoded PNG masks (RGBA, alpha = mask).
"No mask" = tag absent (None); "empty mask" = tag present with ``{}`` keyframes.

PyAV is import-guarded: the module-level ``av`` is ``None`` when PyAV is
unavailable, and the read/write functions raise ``RuntimeError`` in that case.
"""

import json
import math
import ntpath
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

try:
    import av

    _HAS_AV = True
except ImportError:
    av = None  # type: ignore[assignment]
    _HAS_AV = False


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

_METADATA_KEY = "lanpaint-mask"
_PAYLOAD_VERSION = 1


def encode_payload(payload: dict) -> str:
    """Encode a payload dict as a compact JSON string (UTF-8)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_payload(raw: Optional[str]) -> Optional[dict]:
    """Decode a metadata value to a payload dict, or None on any error.

    ``raw`` may be ``None`` (tag absent), a JSON string, or something
    unexpected.  Malformed / missing input returns ``None`` gracefully.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Core read / write
# ---------------------------------------------------------------------------


def read_mask_metadata(path: str) -> Optional[dict]:
    """Read the ``lanpaint-mask`` metadata tag from an MP4 file.

    Returns the decoded payload dict, or ``None`` when the tag is absent or
    PyAV is unavailable.
    """
    if not _HAS_AV:
        raise RuntimeError("PyAV (av) is required to read mask metadata but is not installed")
    with av.open(path, "r") as container:
        raw = container.metadata.get(_METADATA_KEY, None)
    return decode_payload(raw)


def write_mask_metadata(
    input_path: str,
    output_path: Union[str, BinaryIO],
    payload: dict,
) -> None:
    """Remux *input_path* to *output_path*, attaching a ``lanpaint-mask``
    metadata tag.

    The video track is stream-copied (no re-encode) so the video content is
    preserved byte-for-byte.  Audio and other tracks are also preserved.

    The source file at *input_path* is **never** modified.

    *output_path* may be an already-open binary file. The export route uses
    an exclusively created file so concurrent exports cannot overwrite one
    another or follow an existing output symlink.

    Raises ``RuntimeError`` when PyAV is unavailable.
    """
    if not _HAS_AV:
        raise RuntimeError("PyAV (av) is required to write mask metadata but is not installed")

    json_str = encode_payload(payload)

    with av.open(input_path, "r") as in_container:
        # ``movflags=use_metadata_tags`` is MANDATORY — without it ffmpeg
        # silently drops custom metadata keys from the output.
        with av.open(
            output_path,
            "w",
            format="mp4",
            options={"movflags": "use_metadata_tags"},
        ) as out_container:
            # Copy any existing metadata (except our own key) from the source.
            for key, value in in_container.metadata.items():
                if key != _METADATA_KEY:
                    out_container.metadata[key] = value

            # Write the LanPaint payload.
            out_container.metadata[_METADATA_KEY] = json_str

            # Stream-copy every stream from the source.
            stream_map: Dict[int, Any] = {}
            for in_stream in in_container.streams:
                out_stream = out_container.add_stream_from_template(in_stream)
                stream_map[in_stream.index] = out_stream

            # Demux → mux every packet.
            for packet in in_container.demux():
                if packet.dts is None:
                    continue
                out_stream = stream_map[packet.stream.index]
                packet.stream = out_stream
                out_container.mux(packet)

    # The caller is responsible for the output file on disk.


# ---------------------------------------------------------------------------
# Helpers for server routes
# ---------------------------------------------------------------------------


def _input_path(input_dir: str, filename: str) -> Path:
    """Resolve a portable relative filename, confined to the input directory.

    Apply Windows path rules on every platform: drive/UNC paths, alternate
    data streams and backslash traversal must not become valid on POSIX.
    Resolve symlinks (and Windows junctions) before checking containment.
    The input directory is trusted; this does not sandbox a local process
    that can replace its directories while the request is being processed.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty relative path")
    normalized = filename.replace("\\", "/")
    parts = normalized.split("/")
    if (
        ntpath.splitdrive(filename)[0]
        or normalized.startswith("/")
        or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts)
        or any(ord(char) < 32 or char in ':<>"|?*' for char in normalized)
    ):
        raise ValueError("filename must be a relative path inside the input directory")
    try:
        root = Path(input_dir).resolve()
        path = root.joinpath(*parts).resolve()
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("filename must stay inside the input directory") from exc
    return path


def _unique_output_path(input_dir: str, base_name: str) -> str:
    """Return a non-clobbering output filename in *input_dir*.

    Appends ``_masked``, then ``_masked_2``, ``_masked_3``, ... until a name
    that does not already exist is found.

    Returns the relative filename, preserving an allowed input subfolder.
    The caller must still create the file exclusively to handle races.
    """
    _input_path(input_dir, base_name)
    stem, ext = os.path.splitext(base_name.replace("\\", "/"))
    candidate = f"{stem}_masked{ext}"
    if not os.path.lexists(os.path.join(input_dir, candidate)):
        return candidate
    n = 2
    while True:
        candidate = f"{stem}_masked_{n}{ext}"
        if not os.path.lexists(os.path.join(input_dir, candidate)):
            return candidate
        n += 1


def export_mask_video_from_request(
    input_dir: str,
    filename: str,
    keyframes: dict,
    audio_intervals: list,
    fps: float,
) -> str:
    """Remux a source video with mask metadata, returning the new filename.

    Parameters
    ----------
    input_dir:
        The ComfyUI input directory (``folder_paths.get_input_directory()``).
    filename:
        The source video filename (relative to *input_dir*).
    keyframes:
        Dict of ``{frame_idx: base64_png_string}``.
    audio_intervals:
        List of ``{"start": float, "end": float}`` dicts.
    fps:
        The video frame rate.

    Returns
    -------
    The relative filename of the exported MP4 (in *input_dir*).
    """
    payload: Dict[str, Any] = {
        "version": _PAYLOAD_VERSION,
        "video": filename,
        "fps": fps,
        "keyframes": keyframes,
        "audio_intervals": audio_intervals,
    }

    src = _input_path(input_dir, filename)
    if not src.is_file():
        raise FileNotFoundError("source video not found")

    while True:
        out_name = _unique_output_path(input_dir, filename)
        # Validate the resolved destination, then open the original path with
        # O_EXCL ("xb"). Opening the resolved target would follow a symlink
        # introduced between name selection and validation.
        _input_path(input_dir, out_name)
        out_path = os.path.join(input_dir, out_name)
        try:
            output = open(out_path, "xb")
        except FileExistsError:
            continue
        try:
            with output:
                write_mask_metadata(str(src), output, payload)
        except Exception:
            os.unlink(out_path)
            raise
        return out_name


def register_routes(server) -> None:
    """Register the LanPaint video-mask metadata routes on a ComfyUI
    PromptServer instance.

    Call this from ``__init__.py`` when running inside ComfyUI (the ``server``
    module is only importable in that environment).  Safe to call multiple
    times — routes are registered once.
    """
    import aiohttp
    import folder_paths
    from yarl import URL

    @server.routes.get("/lanpaint/video_mask_meta")
    async def video_mask_meta(request: aiohttp.web.Request) -> aiohttp.web.Response:
        filename = request.query.get("filename", "")
        if not filename:
            return aiohttp.web.json_response({"found": False})

        input_dir = folder_paths.get_input_directory()
        try:
            path = _input_path(input_dir, filename)
        except ValueError as e:
            return aiohttp.web.json_response({"error": str(e)}, status=400)
        if not path.is_file():
            return aiohttp.web.json_response({"found": False})

        try:
            payload = read_mask_metadata(str(path))
        except Exception:
            payload = None

        if payload is None:
            return aiohttp.web.json_response({"found": False})
        return aiohttp.web.json_response({"found": True, "payload": payload})

    @server.routes.post("/lanpaint/export_mask_video")
    async def export_mask_video(request: aiohttp.web.Request) -> aiohttp.web.Response:
        # This endpoint writes a file. Do not rely on ComfyUI's optional CORS
        # middleware: a simple cross-origin text/plain POST needs no preflight.
        if request.content_type != "application/json":
            return aiohttp.web.json_response({"error": "application/json required"}, status=415)
        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return aiohttp.web.json_response({"error": "same-origin request required"}, status=403)
        origin = request.headers.get("Origin")
        if origin is not None:
            try:
                origin_url = URL(origin)
                same_origin = (
                    origin_url.scheme in ("http", "https")
                    and origin_url.user is None
                    and origin_url.path == "/"
                    and not origin_url.query_string
                    and not origin_url.fragment
                    and origin_url.origin() == request.url.origin()
                )
            except ValueError:
                same_origin = False
            if not same_origin:
                return aiohttp.web.json_response({"error": "same-origin request required"}, status=403)
        try:
            body = await request.json()
        except Exception:
            return aiohttp.web.json_response({"error": "invalid JSON body"}, status=400)

        if not isinstance(body, dict):
            return aiohttp.web.json_response({"error": "JSON body must be an object"}, status=400)

        filename = body.get("filename", "")
        if not isinstance(filename, str) or not filename:
            return aiohttp.web.json_response({"error": "filename must be a non-empty string"}, status=400)

        keyframes = body.get("keyframes", {})
        if not isinstance(keyframes, dict):
            return aiohttp.web.json_response({"error": "keyframes must be a dict"}, status=400)

        audio_intervals = body.get("audio_intervals", [])
        if not isinstance(audio_intervals, list):
            return aiohttp.web.json_response({"error": "audio_intervals must be a list"}, status=400)

        fps = body.get("fps", 30.0)
        try:
            fps = float(fps)
            if isinstance(body.get("fps"), bool) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("invalid fps")
        except (TypeError, ValueError, OverflowError):
            return aiohttp.web.json_response({"error": "fps must be a finite positive number"}, status=400)

        input_dir = folder_paths.get_input_directory()
        try:
            out_name = export_mask_video_from_request(input_dir, filename, keyframes, audio_intervals, fps)
        except ValueError as e:
            return aiohttp.web.json_response({"error": str(e)}, status=400)
        except FileNotFoundError as e:
            return aiohttp.web.json_response({"error": str(e)}, status=404)
        except Exception:
            return aiohttp.web.json_response({"error": "video export failed"}, status=500)

        return aiohttp.web.json_response({"filename": out_name, "path": os.path.join(input_dir, out_name)})
