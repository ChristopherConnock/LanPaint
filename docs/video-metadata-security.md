# Video metadata endpoint security

This fork hardens the video metadata endpoints introduced in LanPaint 2.0.
The sampler, node interfaces and mask payload format are unchanged.

Both `/lanpaint/video_mask_meta` and `/lanpaint/export_mask_video` accept only
relative filenames inside ComfyUI's input directory. Subfolders and Unicode
filenames work; traversal, absolute paths, Windows drive/UNC paths and alternate
data streams are rejected. Symlinks and Windows junctions are resolved before
checking containment. Invalid paths return HTTP 400 before media is opened.

Exports preserve the existing `_masked`, `_masked_2`, ... naming convention.
Files are opened exclusively and passed to PyAV as an open stream, so an existing
file or output symlink cannot be overwritten, including concurrent name
collisions. A failed export removes its partial output.

The export endpoint requires `Content-Type: application/json`. If an `Origin`
header is supplied, it must match the request's scheme, host and port. Cross-site
or same-site (but cross-origin) Fetch Metadata is rejected. Existing browser
exports already use JSON; non-browser JSON clients without an Origin header
remain supported. These checks prevent browser cross-origin writes; they do not
authenticate clients. Reverse proxies must preserve the public request origin
in the application's request context. Do not trust arbitrary forwarded headers.

The input directory is trusted local storage. This is not a sandbox against a
local process with permission to swap its directories during a request, nor a
replacement for ComfyUI's access controls. Reading arbitrary media still uses
PyAV and its codec libraries.

Run `pytest` and `ruff check .` after installing the development dependencies
and CPU PyTorch. The security tests exercise real aiohttp requests without a GPU
or ComfyUI. Existing metadata tests also create, export and reread small MP4s.
The CI matrix covers Linux and Windows; file-symlink tests skip on Windows when
the account lacks symlink privileges, while the Windows junction test still runs.
