import glob, json, sys
sys.path.insert(0, __file__.rsplit('/',1)[0])
from clean import clean
from PIL import Image, ImageDraw
S = json.load(open(sys.argv[1]))
base, src, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]
grp = [int(x) for x in sys.argv[5].split(',')]
cw, ch, cols = 400, 500, 3
rows = (len(grp) + cols - 1) // cols
sheet = Image.new('L', (cw*cols, ch*rows), 255); dr = ImageDraw.Draw(sheet)
for j, i in enumerate(grp):
    g = glob.glob('%s/*_%d.png' % (src, base + i))
    if not g: print('MISSING', i); continue
    im, th, c = clean(g[0]); print(i, S[i-1][1], th, round(c, 3))
    x, y = (j % cols)*cw, (j // cols)*ch
    sheet.paste(im.resize((cw-10, ch-30), Image.LANCZOS), (x+5, y+25))
    dr.text((x+8, y+8), '%d %s' % (i, S[i-1][1]), fill=0)
    dr.rectangle([x, y, x+cw-1, y+ch-1], outline=0)
sheet.save(out); print('saved', out)
