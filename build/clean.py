from PIL import Image
import numpy as np
from scipy import ndimage

def clean(path, th=None, despeckle=45, fill_holes=25):
    a = np.array(Image.open(path).convert('L'))
    if th is None:
        th = 80
        for t in (80, 65, 55, 45):
            if float((a < t).mean()) <= 0.30:
                th = t; break
    b = a < th
    lab, n = ndimage.label(b)
    if n:
        sizes = ndimage.sum(b, lab, range(1, n + 1))
        b[np.isin(lab, np.nonzero(sizes < despeckle)[0] + 1)] = False
    w = ~b
    lab, n = ndimage.label(w)
    if n:
        sizes = ndimage.sum(w, lab, range(1, n + 1))
        b[np.isin(lab, np.nonzero(sizes < fill_holes)[0] + 1)] = True
    return Image.fromarray(np.where(b, 0, 255).astype('uint8')), th, float(b.mean())
