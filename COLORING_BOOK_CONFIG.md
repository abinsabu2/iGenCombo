# iGenCombo — kids colouring book page config (validated 2026-09-12)

## Generator settings (the winning combination)
| param | value | why |
|---|---|---|
| `width` | `512` | sd-turbo is 512-native |
| `height` | `640` | 4:5 — closest page-ish ratio that stays near native |
| `steps` | `4` | **the single most important setting.** 8 floods the page with ripple/bark/grass hatching (~27% ink); 4 gives clean open line art (~12-18% ink) |
| `guidance` | `3.5` | 5.5+ collapses faces into black blobs; 0-2 ignores the prompt |
| `model` | default (`stabilityai/sd-turbo`) | |
| `seed` | one per page, recorded | reruns are pixel-identical |

### Sizes that DO NOT work
- **`816x1056` (8.5x11 @96dpi) is the trap.** sd-turbo is 512-native, so at page size it *tiles the subject*: every prompt comes back as a crowd of 8-20 duplicated figures. No amount of "only one", "no crowd", "isolated" fixes it. This was the long-standing "model duplicates Jesus" bug — it is a resolution problem, not a prompt problem.
- `1024x1024` on the default model returns a pure black image.
- `steps=20` never returns; the request is dropped.

Generate at 512x640 and upscale in PIL to page size instead.

## Prompt template
```
Toddler colouring book page. {SCENE}. Plain empty white background, no scenery,
no landscape, no grass, no texture lines, no hatching, no cross-hatching,
no stipple dots, no speckles, no wood grain. Extra-thick bold black outlines,
pure white background, no shading, no grey, no color, minimal internal detail,
huge empty white areas, flat vector sticker style.
```

### Scene-writing rules
- Name **one** main figure plus a small count of others ("two children", "three sheep").
- **Never name terrain or surfaces** — "river", "hills", "meadow grass", "cobblestones", "wood grain", "wool curls" all come back as dense hatching that is unusable for young children. Say "three simple wavy water lines" instead of "a wide river".
- Props work better than backgrounds: "three giant simple flowers and two soft round clouds" beats "a sunny meadow".
- Object counts in the prompt are approximate at best — the model will not honour an exact count. Anything that must be exact (numbers, letters, equations) has to be drawn in PIL, not prompted.

## Post-processing (required — the model always outputs colour/grey)
```python
from PIL import Image
import numpy as np
from scipy import ndimage

def clean(path, th=None, despeckle=45, fill_holes=25):
    a = np.array(Image.open(path).convert('L'))
    if th is None:                       # auto-pick threshold
        th = 80
        for t in (80, 65, 55, 45):
            if float((a < t).mean()) <= 0.30:
                th = t; break
    b = a < th                           # True = ink
    lab, n = ndimage.label(b)            # drop stipple dots / grain
    if n:
        sizes = ndimage.sum(b, lab, range(1, n + 1))
        b[np.isin(lab, np.nonzero(sizes < despeckle)[0] + 1)] = False
    w = ~b                               # fill tiny white pinholes
    lab, n = ndimage.label(w)
    if n:
        sizes = ndimage.sum(w, lab, range(1, n + 1))
        b[np.isin(lab, np.nonzero(sizes < fill_holes)[0] + 1)] = True
    return Image.fromarray(np.where(b, 0, 255).astype('uint8'))
```
- Threshold **80** is the default. Drop to 65/55/45 when a page has dark fills (hair, night sky) that would otherwise become solid black blobs. 150 is always too high; below 45 thin lines vanish.
- The despeckle step is what removes the stippled-beard and speckled-sky noise; plain thresholding leaves it in.
- **QC metric:** `(a < th).mean()` = ink coverage. `0.10-0.20` is a healthy toddler page. `> 0.32` means the page is a texture field — regenerate with a new seed, don't try to salvage it.

## Page assembly for KDP
1. Clean at 512x640, then LANCZOS-upscale and re-threshold to pure `[0, 255]`.
2. Paste centred on a **2550x3300** white canvas (8.5x11 @ 300dpi) with >=200px margins; leave the inner gutter wider.
3. Save PNG with `dpi=(300, 300)`; PDF via `convert('RGB').save(..., resolution=300)`.
4. Verify `np.unique(arr) == [0, 255]` before export.

## Throughput
At 512x640 / steps=4 a call returns in **~6-10s** and does not time out. Batch 5-6 calls at a time; 27 pages takes ~4 minutes. (The old 60s timeouts were entirely caused by the 816x1056 size.)
