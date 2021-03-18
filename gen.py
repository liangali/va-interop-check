import numpy as np 

w, h = 224, 224
nv12file = "build/out_224x224.nv12"

d = np.fromfile(nv12file, dtype=np.uint8)
y_plane = d[:w*h].reshape((h, w))
uv_plane = d[w*h:].reshape((h//2, w))

with open('ref.h', 'wt') as f:
    f.write('static unsigned char ref_nv12_y[] = {\n')
    for y in range(h):
        line = ''
        for x in range(w):
            line += '0x%02x, ' % y_plane[y][x]
        f.write(line+'\n')
    f.write('};\n')

    f.write('static unsigned char ref_nv12_uv[] = {\n')
    for y in range(h//2):
        line = ''
        for x in range(w):
            line += '0x%02x, ' % uv_plane[y][x]
        f.write(line+'\n')
    f.write('};\n')

print('done')