import numpy as np

def multi_shot_acquisition(
    ix_src,
    receivers,
):
    ix_src = np.asarray(ix_src, dtype=np.int64)
    receivers = np.asarray(receivers, dtype=np.int64)
    n_shot = len(ix_src)
    n_receiver = len(receivers)
    ix_receiver = np.zeros((n_shot, n_receiver-1),dtype=np.int64)
    for i_shot in range(n_shot):
        ix_receiver[i_shot,:] = receivers[receivers != ix_src[i_shot]]

    return ix_receiver