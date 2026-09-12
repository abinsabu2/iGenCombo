import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import src.gen
from PIL import Image


class FakeResult:
    def __init__(self, image):
        self.images = [image]


class FakePipe:
    def __init__(self):
        self.calls = []
        self.load_calls = []
        self.scale_calls = []
        self.fake_image = Image.new("RGB", (8, 8))

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kw):
        pass

    def load_ip_adapter(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))

    def set_ip_adapter_scale(self, scale):
        self.scale_calls.append(scale)

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return FakeResult(self.fake_image)


class TestRefImages(unittest.TestCase):
    def setUp(self):
        try:
            src.gen._load_pipe.cache_clear()
        except Exception:
            pass

    def test_no_ref_path_unchanged(self):
        """ref_images=None must not load adapter or pass ip_adapter_image."""
        fake_pipe = FakePipe()
        with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
            path = src.gen.generate("a cat", steps=1, seed=0, out_dir="out_test_no_ref")
        self.assertEqual(fake_pipe.load_calls, [])
        self.assertEqual(fake_pipe.scale_calls, [])
        # no ip_adapter_image in call kwargs
        self.assertTrue(fake_pipe.calls)
        self.assertNotIn("ip_adapter_image", fake_pipe.calls[0])
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_empty_list_same_as_none(self):
        fake_pipe = FakePipe()
        with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
            path = src.gen.generate("a cat", steps=1, seed=1, out_dir="out_test_no_ref", ref_images=[])
        self.assertEqual(fake_pipe.load_calls, [])
        self.assertNotIn("ip_adapter_image", fake_pipe.calls[0])
        os.remove(path)

    def test_bad_ref_path_raises(self):
        fake_pipe = FakePipe()
        with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
            with self.assertRaises((FileNotFoundError, ValueError)) as ctx:
                src.gen.generate("a cat", steps=1, seed=0, out_dir="out_test_no_ref", ref_images=["/no/such/file.jpg"])
            self.assertIn("/no/such/file.jpg", str(ctx.exception))
        # no adapter should have been loaded
        self.assertEqual(fake_pipe.load_calls, [])

    def test_unreadable_image_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="w") as f:
            f.write("not an image")
            bad_path = f.name
        try:
            fake_pipe = FakePipe()
            with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
                with self.assertRaises(ValueError) as ctx:
                    src.gen.generate("a cat", steps=1, seed=0, out_dir="out_test_no_ref", ref_images=[bad_path])
                self.assertIn(bad_path, str(ctx.exception))
        finally:
            os.unlink(bad_path)

    def test_valid_ref_loads_adapter(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (8, 8), color="red").save(f, format="PNG")
            good_path = f.name
        try:
            fake_pipe = FakePipe()
            with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
                path = src.gen.generate("a cat", steps=1, seed=0, out_dir="out_test_no_ref", ref_images=[good_path], ref_scale=0.7)
            self.assertEqual(len(fake_pipe.load_calls), 1)
            args, kwargs = fake_pipe.load_calls[0]
            self.assertEqual(args[0], ["h94/IP-Adapter"])
            self.assertEqual(kwargs["subfolder"], ["sdxl_models"])
            self.assertEqual(kwargs["weight_name"], ["ip-adapter-plus-face_sdxl_vit-h.safetensors"])
            self.assertEqual(fake_pipe.scale_calls, [[0.7]])
            self.assertIn("ip_adapter_image", fake_pipe.calls[0])
            os.remove(path)
        finally:
            os.unlink(good_path)
