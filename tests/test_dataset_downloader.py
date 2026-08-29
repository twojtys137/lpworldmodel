import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_lpwmdatasets.py"
SPEC = importlib.util.spec_from_file_location("download_lpwmdatasets", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DatasetDownloaderTest(unittest.TestCase):
    def test_manifest_matches_official_osf_files(self):
        pusht = downloader.DATASETS["pusht_noise"]
        wall = downloader.DATASETS["wall_single"]
        self.assertEqual(pusht["size"], 2_785_304_515)
        self.assertEqual(wall["size"], 1_668_205_895)
        self.assertEqual(
            pusht["sha256"],
            "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08",
        )
        self.assertEqual(
            wall["sha256"],
            "2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d",
        )

    def test_safe_extract_and_layout_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "pusht.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for marker in downloader.DATASETS["pusht_noise"]["markers"]:
                    if Path(marker).suffix:
                        archive.writestr(f"pusht_noise/{marker}", b"test")
                    else:
                        archive.writestr(f"pusht_noise/{marker}/placeholder", b"test")
            extract_dir = root / "extract"
            downloader.safe_extract_zip(archive_path, extract_dir)
            found = downloader.find_extracted_dataset(extract_dir, "pusht_noise")
            self.assertEqual(found, extract_dir / "pusht_noise")

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", b"no")
            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                downloader.safe_extract_zip(archive_path, root / "extract")
            self.assertFalse((root / "outside.txt").exists())

    def test_completed_partial_is_verified_without_another_request(self):
        payload = b"complete archive"
        name = "test_complete_partial"
        downloader.DATASETS[name] = {
            "url": "https://example.invalid/test.zip",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "markers": (),
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                archive_path = Path(directory) / "test.zip"
                archive_path.with_suffix(".zip.part").write_bytes(payload)
                with mock.patch.object(downloader.requests, "get") as request:
                    result = downloader.download_archive(name, archive_path)
                request.assert_not_called()
                self.assertEqual(result.read_bytes(), payload)
        finally:
            downloader.DATASETS.pop(name)


if __name__ == "__main__":
    unittest.main()
