"""Path and HTTP regression tests; no PyAV, GPU or ComfyUI required."""

import asyncio
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.LanPaint import videometa


BAD_PATHS = [
    "../outside.mp4",
    "sub/../../outside.mp4",
    "..\\outside.mp4",
    "sub\\..\\..\\outside.mp4",
    "/outside.mp4",
    "\\outside.mp4",
    "C:\\outside.mp4",
    "C:/outside.mp4",
    "C:outside.mp4",
    "\\\\server\\share\\outside.mp4",
    "//server/share/outside.mp4",
    "\\\\?\\C:\\outside.mp4",
    "clip.mp4:stream",
    "sub/.. /outside.mp4",
    "sub./clip.mp4",
    "clip.mp4\x00",
]


@pytest.fixture
def input_dir(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"source")
    (tmp_path / "outside.mp4").write_bytes(b"outside")
    return root


@pytest.fixture
def media_calls(monkeypatch):
    calls = []

    def write(source, output, payload):
        calls.append((source, output.name, payload))
        output.write(b"export")

    monkeypatch.setattr(videometa, "write_mask_metadata", write)
    monkeypatch.setattr(videometa, "read_mask_metadata", lambda path: calls.append(path) or {"version": 1})
    return calls


@pytest.fixture
def request_api(input_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_input_directory=lambda: str(input_dir)))

    async def request(method, path, *, origin=None, **kwargs):
        routes = web.RouteTableDef()
        videometa.register_routes(SimpleNamespace(routes=routes))
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            if origin is not None:
                kwargs.setdefault("headers", {})["Origin"] = str(client.make_url("/").origin()) if origin == "self" else origin
            response = await client.request(method, path, **kwargs)
            return response.status, await response.json()

    return request


def export(root, filename="clip.mp4"):
    return videometa.export_mask_video_from_request(str(root), filename, {"0": "mask"}, [], 30)


@pytest.mark.parametrize("filename", BAD_PATHS)
def test_export_rejects_paths_before_media_access(input_dir, media_calls, filename):
    with pytest.raises(ValueError):
        export(input_dir, filename)
    assert media_calls == []
    assert sorted(p.name for p in input_dir.iterdir()) == ["clip.mp4"]


@pytest.mark.parametrize("filename", BAD_PATHS)
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_routes_reject_paths_before_media_access(request_api, media_calls, filename, method):
    if method == "GET":
        status, _ = asyncio.run(request_api(method, "/lanpaint/video_mask_meta", params={"filename": filename}))
    else:
        status, _ = asyncio.run(request_api(method, "/lanpaint/export_mask_video", json={"filename": filename}))
    assert status == 400
    assert media_calls == []


def test_encoded_traversal_is_rejected(request_api, media_calls):
    status, _ = asyncio.run(request_api("GET", "/lanpaint/video_mask_meta?filename=%2e%2e%2foutside.mp4"))
    assert status == 400
    assert media_calls == []


def test_containment_uses_path_components(input_dir, media_calls):
    sibling = input_dir.with_name("input-other")
    sibling.mkdir()
    (sibling / "clip.mp4").write_bytes(b"outside")
    with pytest.raises(ValueError):
        export(input_dir, str(sibling / "clip.mp4"))
    assert media_calls == []


@pytest.mark.parametrize("filename", ["sub/テスト.mp4", "sub\\テスト.mp4"])
def test_nested_unicode_export_preserves_source(input_dir, media_calls, filename):
    (input_dir / "sub").mkdir()
    source = input_dir / "sub" / "テスト.mp4"
    source.write_bytes(b"source")
    result = export(input_dir, filename)
    assert result == "sub/テスト_masked.mp4"
    assert (input_dir / result).read_bytes() == b"export"
    assert source.read_bytes() == b"source"
    assert Path(media_calls[0][0]) == source.resolve()


def make_symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")


def test_source_symlink_outside_input_is_rejected(input_dir, media_calls):
    make_symlink(input_dir / "link.mp4", input_dir.parent / "outside.mp4")
    with pytest.raises(ValueError):
        export(input_dir, "link.mp4")
    assert media_calls == []


def test_directory_link_cannot_escape_input(input_dir, media_calls):
    link = input_dir / "linked"
    if os.name == "nt":
        # Junction creation does not require Windows symlink privileges.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(input_dir.parent)], check=True, capture_output=True)
    else:
        make_symlink(link, input_dir.parent)
    with pytest.raises(ValueError):
        export(input_dir, "linked/outside.mp4")
    assert media_calls == []


def test_dangling_output_symlink_is_skipped(input_dir, media_calls):
    outside = input_dir.parent / "must-not-exist.mp4"
    make_symlink(input_dir / "clip_masked.mp4", outside)
    assert export(input_dir) == "clip_masked_2.mp4"
    assert not outside.exists()
    assert (input_dir / "clip_masked.mp4").is_symlink()


def test_exclusive_creation_handles_collision_after_name_selection(input_dir, media_calls, monkeypatch):
    target = input_dir / "clip_masked.mp4"
    raced = False

    def racing_open(path, mode):
        nonlocal raced
        assert mode == "xb"
        if not raced:
            raced = True
            target.write_bytes(b"other export")
        return builtins.open(path, mode)

    monkeypatch.setattr(videometa, "open", racing_open, raising=False)
    assert export(input_dir) == "clip_masked_2.mp4"
    assert target.read_bytes() == b"other export"
    assert (input_dir / "clip_masked_2.mp4").read_bytes() == b"export"


def test_failed_export_removes_only_its_partial_output(input_dir, monkeypatch):
    existing = input_dir / "clip_masked.mp4"
    existing.write_bytes(b"keep")

    def fail(source, output, payload):
        output.write(b"partial")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(videometa, "write_mask_metadata", fail)
    with pytest.raises(RuntimeError):
        export(input_dir)
    assert existing.read_bytes() == b"keep"
    assert (input_dir / "clip.mp4").read_bytes() == b"source"
    assert not (input_dir / "clip_masked_2.mp4").exists()


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:1", "http://[", "https://evil.example/path"])
def test_cross_origin_export_is_rejected(request_api, media_calls, origin):
    status, _ = asyncio.run(request_api("POST", "/lanpaint/export_mask_video", origin=origin, json={"filename": "clip.mp4"}))
    assert status == 403
    assert media_calls == []


@pytest.mark.parametrize("fetch_site", ["cross-site", "same-site"])
def test_fetch_metadata_blocks_cross_origin_without_origin_header(request_api, media_calls, fetch_site):
    status, _ = asyncio.run(
        request_api("POST", "/lanpaint/export_mask_video", headers={"Sec-Fetch-Site": fetch_site}, json={"filename": "clip.mp4"})
    )
    assert status == 403
    assert media_calls == []


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data"])
def test_simple_post_content_types_are_rejected(request_api, media_calls, content_type):
    status, _ = asyncio.run(
        request_api(
            "POST", "/lanpaint/export_mask_video", headers={"Content-Type": content_type}, data=json.dumps({"filename": "clip.mp4"})
        )
    )
    assert status == 415
    assert media_calls == []


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        "clip.mp4",
        {"filename": []},
        {"filename": 12},
        {"filename": ""},
        {"filename": "clip.mp4", "keyframes": []},
        {"filename": "clip.mp4", "audio_intervals": {}},
    ],
)
def test_invalid_json_shapes_return_client_errors(request_api, media_calls, body):
    status, _ = asyncio.run(
        request_api("POST", "/lanpaint/export_mask_video", headers={"Content-Type": "application/json"}, data=json.dumps(body))
    )
    assert status == 400
    assert media_calls == []


@pytest.mark.parametrize("fps", [0, -1, "nan", "inf", "-inf", True, None, []])
def test_invalid_fps_returns_client_error(request_api, media_calls, fps):
    status, _ = asyncio.run(request_api("POST", "/lanpaint/export_mask_video", json={"filename": "clip.mp4", "fps": fps}))
    assert status == 400
    assert media_calls == []


@pytest.mark.parametrize("origin", [None, "self"])
def test_browser_and_non_browser_json_exports_work(request_api, input_dir, media_calls, origin):
    status, body = asyncio.run(request_api("POST", "/lanpaint/export_mask_video", origin=origin, json={"filename": "clip.mp4"}))
    assert status == 200
    assert body["filename"] == "clip_masked.mp4"
    assert (input_dir / body["filename"]).read_bytes() == b"export"


def test_valid_get_and_missing_file_behavior(request_api, media_calls):
    status, body = asyncio.run(request_api("GET", "/lanpaint/video_mask_meta", params={"filename": "clip.mp4"}))
    assert status == 200 and body == {"found": True, "payload": {"version": 1}}
    status, body = asyncio.run(request_api("GET", "/lanpaint/video_mask_meta", params={"filename": "missing.mp4"}))
    assert status == 200 and body == {"found": False}
    status, body = asyncio.run(request_api("POST", "/lanpaint/export_mask_video", json={"filename": "missing.mp4"}))
    assert status == 404 and body == {"error": "source video not found"}


def test_invalid_json_returns_client_error(request_api, media_calls):
    status, _ = asyncio.run(request_api("POST", "/lanpaint/export_mask_video", headers={"Content-Type": "application/json"}, data="{"))
    assert status == 400
    assert media_calls == []
