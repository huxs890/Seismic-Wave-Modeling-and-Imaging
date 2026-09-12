import numpy as np

def multi_shot_acquisition(
    ix_src,
    receivers,
):
    ix_src = np.asarray(ix_src, dtype=np.int64)
    receivers = np.asarray(receivers, dtype=np.int64)
    n_shot = len(ix_src)
    ix_receiver = np.tile(receivers, (n_shot, 1))

    return ix_receiver
