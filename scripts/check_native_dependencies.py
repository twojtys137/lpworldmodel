"""Check native dependencies, validating decord's known wheel-tag mismatch at runtime."""
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


DECORD_WARNING = "decord 0.6.0 is not supported on this platform"
DECORD_WHEEL_TAG = "Tag: cp36-cp36m-manylinux2010_x86_64"


def decode_test_video():
    """Exercise the bundled decoder and indexed batch reads, not just its import."""
    import decord
    import imageio.v2 as imageio
    import numpy as np

    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index] = 24 + index * 24
    with tempfile.TemporaryDirectory(prefix="native-decord-check-") as directory:
        video = Path(directory) / "probe.mp4"
        imageio.mimwrite(video, frames, format="FFMPEG", fps=5, codec="libx264",
                         pixelformat="yuv420p", quality=9)
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
        if len(reader) != len(frames):
            raise RuntimeError(f"decord frame count mismatch: {len(reader)} != {len(frames)}")
        indices = [7, 0, 3, 3]
        decoded = reader.get_batch(indices).asnumpy()
        expected = frames[indices]
        if decoded.shape != expected.shape or decoded.dtype != np.uint8:
            raise RuntimeError(f"decord returned unexpected frames: {decoded.shape}, {decoded.dtype}")
        error = float(np.abs(decoded.astype(float) - expected.astype(float)).max())
        if error > 4:
            raise RuntimeError(f"decord video round-trip error is too high: {error}")
        del reader
    print(f"decord video decode OK: 8 frames, indexed batch including repeats, max error={error}",
          flush=True)


def check_dependencies():
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 1 or lines != [DECORD_WARNING]:
            raise RuntimeError("pip check found dependency errors; see output above")
        dist = metadata.distribution("decord")
        wheel = dist.read_text("WHEEL") or ""
        if (platform.system() != "Linux" or platform.machine() != "x86_64"
                or dist.version != "0.6.0" or DECORD_WHEEL_TAG not in wheel.splitlines()):
            raise RuntimeError("Unsupported decord platform or unrecognized wheel metadata")
        print("Known decord 0.6.0 wheel metadata mismatch; checking actual video decoding.",
              flush=True)
    decode_test_video()
    print("Native dependency checks passed.", flush=True)


if __name__ == "__main__":
    check_dependencies()
