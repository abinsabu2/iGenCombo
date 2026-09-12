import inspect, sys, tempfile, os
from unittest.mock import patch, MagicMock

import src.gen as gen_mod
import src.cli as cli_mod
import src.mcp_server as mcp_mod

# (a) defaults agree
gsig = inspect.signature(gen_mod.generate)
msig = inspect.signature(mcp_mod.generate_image)
assert gsig.parameters["steps"].default == 1
assert gsig.parameters["height"].default == 512
assert gsig.parameters["width"].default == 512
assert gsig.parameters["guidance"].default == 0.0
assert gsig.parameters["model"].default is None
assert msig.parameters["steps"].default == 1
assert msig.parameters["height"].default == 512
assert msig.parameters["width"].default == 512
assert msig.parameters["guidance"].default == 0.0
assert msig.parameters["model"].default is None

# cli defaults via mocking cli's generate binding
old_argv = sys.argv
sys.argv = ["cli", "--prompt", "hi"]
try:
    with patch.object(cli_mod, "generate") as mg:
        mg.return_value = "/tmp/x.png"
        cli_mod.main()
        kw = mg.call_args.kwargs
        assert kw["steps"] == 1, kw
        assert kw["height"] == 512
        assert kw["width"] == 512
        assert kw["model"] is None
finally:
    sys.argv = old_argv
print("a) defaults agree: ok")

# helper stub
def make_fake(capture):
    stub_image = MagicMock()
    stub_image.save = MagicMock()
    class Pipe:
        scheduler = MagicMock()
        def to(self, d): return self
        def set_progress_bar_config(self, **kw): pass
        def load_ip_adapter(self, *a, **kw): pass
        def set_ip_adapter_scale(self, *a, **kw): pass
        def __call__(self, *a, **kw):
            capture["height"] = kw.get("height")
            capture["width"] = kw.get("width")
            return MagicMock(images=[stub_image])
    pipe = Pipe()
    def fake(model, **kw):
        capture["model"] = model
        capture["kw"] = kw
        return pipe
    return fake

tmpdir = tempfile.mkdtemp()
ref_img = os.path.join(tmpdir, "ref.jpg")
from PIL import Image as PILImage
PILImage.new("RGB", (64, 64), "red").save(ref_img)

# (b) explicit non-SDXL with ref_images raises
capture = {}
with patch("src.gen.AutoPipelineForText2Image.from_pretrained", side_effect=make_fake(capture)):
    try:
        gen_mod.generate(prompt="hi", model="stabilityai/sd-turbo", ref_images=[ref_img], out_dir=tmpdir, seed=0)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "SDXL" in str(e), e
        print(f"b) explicit non-SDXL raises ValueError: ok ({e})")

# (c) default model + ref_images -> sdxl at 1024
capture = {}
with patch("src.gen.AutoPipelineForText2Image.from_pretrained", side_effect=make_fake(capture)):
    gen_mod.generate(prompt="hi", ref_images=[ref_img], out_dir=tmpdir, seed=0)
    assert capture["model"] == "stabilityai/sdxl-turbo", capture["model"]
    assert capture["height"] == 1024 and capture["width"] == 1024, f"{capture['height']}x{capture['width']}"
    print(f"c) default+ref resolves to sdxl 1024: ok (model={capture['model']} {capture['height']}x{capture['width']})")

# (d) cache hit — same args returns SAME object, from_pretrained called once
gen_mod._load_pipe.cache_clear()
_count = {"n": 0}
def count_fake(model, **kw):
    _count["n"] += 1
    m = MagicMock()
    m.save = MagicMock()
    class P:
        scheduler = MagicMock()
        def to(self, d): return self
        def set_progress_bar_config(self, **k): pass
        def load_ip_adapter(self, *a, **kw): pass
        def set_ip_adapter_scale(self, *a, **kw): pass
        def __call__(self, *a, **kw): return MagicMock(images=[m])
    pipe = P()
    return pipe

# resolve device/dtype same as generate does
import torch as _torch
_dev = "cuda" if _torch.cuda.is_available() else ("mps" if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available() else "cpu")
_dtype = _torch.float16 if _dev != "cpu" else _torch.float32
with patch("src.gen.AutoPipelineForText2Image.from_pretrained", side_effect=count_fake):
    p1 = gen_mod._load_pipe("stabilityai/sd-turbo", _dev, _dtype, False)
    p2 = gen_mod._load_pipe("stabilityai/sd-turbo", _dev, _dtype, False)
    assert p1 is p2, "cache miss: same args should return same object"
    assert _count["n"] == 1, f"from_pretrained called {_count['n']} times, expected 1"
    print("d) cache hit same object: ok")

# (e) adapter isolation — ref then text-only does NOT reuse adapter pipe
gen_mod._load_pipe.cache_clear()
_count2 = {"n": 0}
_pipes = {}
def count_fake2(model, **kw):
    _count2["n"] += 1
    m = MagicMock()
    m.save = MagicMock()
    class P2:
        scheduler = MagicMock()
        def to(self, d): return self
        def set_progress_bar_config(self, **k): pass
        def load_ip_adapter(self, *a, **kw): pass
        def set_ip_adapter_scale(self, *a, **kw): pass
        def __call__(self, *a, **kw): return MagicMock(images=[m])
    pipe = P2()
    _pipes[model + str(kw.get("torch_dtype", ""))] = pipe
    return pipe

with patch("src.gen.AutoPipelineForText2Image.from_pretrained", side_effect=count_fake2):
    gen_mod.generate(prompt="hi", ref_images=[ref_img], out_dir=tmpdir, seed=1)
    gen_mod.generate(prompt="hi", ref_images=None, out_dir=tmpdir, seed=2)
    assert _count2["n"] == 2, f"expected 2 loads (adapter vs text-only), got {_count2['n']}"
    # direct key test: same model, different with_ip_adapter => different objects
    gen_mod._load_pipe.cache_clear()
    _count2["n"] = 0
    with patch("src.gen.AutoPipelineForText2Image.from_pretrained", side_effect=count_fake2):
        a = gen_mod._load_pipe("stabilityai/sdxl-turbo", _dev, _dtype, True)
        b = gen_mod._load_pipe("stabilityai/sdxl-turbo", _dev, _dtype, False)
        assert a is not b, "adapter pipe reused for text-only — mutation trap"
        assert _count2["n"] == 2, f"expected 2 distinct cache entries, got {_count2['n']}"
    print("e) adapter isolation: ok")

print("all checks passed")
