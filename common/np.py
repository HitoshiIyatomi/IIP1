# np.py
# if GPU is specified and GPU is available, use cupy instead of numpy 
from common.config import GPU


if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    print ('GPU mode:  [cupy] is used instead of [numpy]')
else:
    import numpy as np