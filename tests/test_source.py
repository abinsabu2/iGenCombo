import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import src.gen


class TestCollectRefImages(unittest.TestCase):
    def test_sorted_order(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["c.png", "a.jpg", "b.jpeg"]:
                Path(d, name).write_bytes(b"x")
            result = src.gen.collect_ref_images(d)
            self.assertEqual(result, [str(Path(d, n)) for n in ["a.jpg", "b.jpeg", "c.png"]])

    def test_case_insensitive(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.JPG", "b.JPEG", "c.PNG", "d.Png", "e.txt", "f.mp4"]:
                Path(d, name).write_bytes(b"x")
            result = src.gen.collect_ref_images(d)
            basenames = [os.path.basename(p) for p in result]
            self.assertEqual(basenames, ["a.JPG", "b.JPEG", "c.PNG", "d.Png"])

    def test_missing_dir(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            src.gen.collect_ref_images("/no/such/dir_xyz_12345")
        self.assertIn("/no/such/dir_xyz_12345", str(ctx.exception))

    def test_file_not_dir(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(FileNotFoundError) as ctx:
                src.gen.collect_ref_images(f.name)
            self.assertIn(f.name, str(ctx.exception))

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                src.gen.collect_ref_images(d)
            self.assertIn(d, str(ctx.exception))

    def test_no_images_only_non_image_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_bytes(b"x")
            Path(d, "b.gif").write_bytes(b"x")
            with self.assertRaises(FileNotFoundError) as ctx:
                src.gen.collect_ref_images(d)
            self.assertIn(d, str(ctx.exception))

    def test_cli_source_missing_exits(self):
        import sys
        import src.cli
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "missing_subdir")
            sys.argv = ["cli", "--prompt", "hi", "--source", missing]
            with self.assertRaises(SystemExit) as ctx:
                src.cli.main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_cli_source_empty_exits(self):
        import sys
        import src.cli
        with tempfile.TemporaryDirectory() as d:
            sys.argv = ["cli", "--prompt", "hi", "--source", d]
            with self.assertRaises(SystemExit) as ctx:
                src.cli.main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_cli_source_forwards_to_generate(self):
        import sys
        import src.cli
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.jpg").write_bytes(b"x")
            Path(d, "b.png").write_bytes(b"x")
            sys.argv = ["cli", "--prompt", "hi", "--source", d, "--ref-scale", "0.65"]
            with patch("src.cli.generate", return_value="/tmp/out.png") as mock_gen:
                src.cli.main()
            mock_gen.assert_called_once()
            kwargs = mock_gen.call_args.kwargs
            self.assertEqual(kwargs["ref_scale"], 0.65)
            self.assertEqual([os.path.basename(p) for p in kwargs["ref_images"]], ["a.jpg", "b.png"])

    def test_mcp_source_forwards(self):
        import importlib
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.jpg").write_bytes(b"x")
            if "src.mcp_server" in __import__("sys").modules:
                importlib.reload(__import__("sys").modules["src.mcp_server"])
            import src.mcp_server as mcp_mod
            with patch.object(mcp_mod, "generate", return_value="/tmp/out.png") as mock_gen:
                result = mcp_mod.generate_image(prompt="hi", source=d, ref_scale=0.55)
            self.assertEqual(result, "/tmp/out.png")
            kwargs = mock_gen.call_args.kwargs
            self.assertEqual(kwargs["ref_scale"], 0.55)
            self.assertEqual(len(kwargs["ref_images"]), 1)

    def test_mcp_source_missing_raises(self):
        import importlib, sys
        if "src.mcp_server" in sys.modules:
            importlib.reload(sys.modules["src.mcp_server"])
        import src.mcp_server as mcp_mod
        with self.assertRaises(FileNotFoundError):
            mcp_mod.generate_image(prompt="hi", source="/no/such/dir_xyz_12345")

    def test_mcp_source_none_no_collect(self):
        import importlib, sys
        if "src.mcp_server" in sys.modules:
            importlib.reload(sys.modules["src.mcp_server"])
        import src.mcp_server as mcp_mod
        with patch.object(mcp_mod, "generate", return_value="/tmp/out.png") as mock_gen:
            mcp_mod.generate_image(prompt="hi", source=None)
        self.assertIsNone(mock_gen.call_args.kwargs["ref_images"])
