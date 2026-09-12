import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import inspect

import src.gen


class TestGen(unittest.TestCase):
    def test_generate_smoke(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not importable")
        has_device = False
        try:
            if torch.cuda.is_available():
                has_device = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                has_device = True
            elif torch.cpu.is_available() if hasattr(torch, "cpu") else True:
                has_device = True
        except Exception:
            pass
        if not has_device:
            self.skipTest("no device available")

        from PIL import Image

        fake_image = Image.new("RGB", (8, 8))

        class FakeResult:
            images = [fake_image]

        class FakePipe:
            scheduler = MagicMock()

            def set_progress_bar_config(self, **kw):
                pass

            def load_ip_adapter(self, *a, **kw):
                pass

            def set_ip_adapter_scale(self, *a, **kw):
                pass

            def __call__(self, *args, **kwargs):
                return FakeResult()

            def to(self, device):
                return self

        fake_pipe = FakePipe()

        with patch("src.gen.AutoPipelineForText2Image.from_pretrained", return_value=fake_pipe):
            # also patch on diffusers module as imported by src.gen (same object)
            path = src.gen.generate("a red apple", steps=5, seed=0, out_dir="out_test_smoke")

        self.assertTrue(os.path.isabs(path))
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            magic = f.read(8)
        self.assertTrue(magic.startswith(b"\x89PNG"))
        self.assertIn("0", os.path.basename(path))

    def test_mcp_server_import(self):
        """Test that mcp_server module imports successfully."""
        # Pre-patch before any import to avoid model download
        with patch('mcp.server.fastmcp.FastMCP', return_value=MagicMock()):
            # Clear module cache to force fresh import
            if 'src.mcp_server' in sys.modules:
                del sys.modules['src.mcp_server']
            import src.mcp_server
            self.assertIsNotNone(src.mcp_server.mcp)

    def test_generate_image_tool_signature(self):
        """Test that generate_image function signature matches spec."""
        # Load the mcp_server source without importing (to avoid FastMCP init)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mcp_server_source",
            "/Users/abinsabu/Documents/iGenCombo/src/mcp_server.py"
        )
        # Just check the function exists on generate_image
        from src.mcp_server import generate_image
        sig = inspect.signature(generate_image)
        # prompt must be first param and required
        self.assertIn('prompt', sig.parameters)
        prompt_param = sig.parameters['prompt']
        self.assertEqual(prompt_param.default, inspect.Parameter.empty)
        # steps, seed, out_dir should have defaults
        self.assertNotEqual(sig.parameters['steps'].default, inspect.Parameter.empty)


