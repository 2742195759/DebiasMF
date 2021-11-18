import numpy as np
import sys
import os.path as osp
assert len(sys.argv) >= 2, "please input the args.: filename int|float|double"
name = sys.argv[1]
dtype = sys.argv[2]
tt = np.load("".join([name, '.npy'])).astype(dtype)
tt = tt.reshape([943, 1682])
tt[tt>5] = 5
np.savetxt("".join([name, '.ascii']), tt, fmt='%1d')
