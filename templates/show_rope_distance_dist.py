'''
@author: Faizan-Uni-Stuttgart

Feb 10, 2020

3:00:57 PM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


def cmpt_dist(pts):

    # rows are points, cols are coords
    assert pts.ndim == 2

    n_pts, n_dims = pts.shape

    print('n_pts, n_dims:', n_pts, n_dims)

    n_dists = (n_pts * (n_pts - 1)) // 2
    dists = np.full(n_dists, np.nan, dtype=float)

    pt_ctr = 0
    for i in range(n_pts):
        for j in range(n_pts):
            if i <= j:
                continue

            dist = (((pts[i] - pts[j]) ** 2).sum()) ** 0.5

            dists[pt_ctr] = dist

            pt_ctr += 1

    assert pt_ctr == n_dists

    assert np.all(np.isfinite(dists))

    return dists


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    kf = 1
    h5_file = r"P:\Synchronize\IWS\QGIS_Neckar\neckar_four_cats_lump_09_ns\01_database\cat_420.hdf5"

    in_h5_hdl = h5py.File(h5_file, 'r')

    kf_str = f'kf_{kf:02d}'

    iter_prm_vecs = in_h5_hdl[ f'calib/{kf_str}/iter_prm_vecs'][...]

    n_iters, n_vecs, n_prms = iter_prm_vecs.shape

    print('n_iters, n_vecs, n_prms:', n_iters, n_vecs, n_prms)

    plt.figure(figsize=(20, 20))

    for i in range(n_iters):
        dists = cmpt_dist(iter_prm_vecs[i])

        dists = np.sort(dists)

        probs = np.arange(1.0, dists.shape[0] + 1.0)
        probs /= dists.shape[0] + 1

        plt.plot(dists, probs, alpha=0.7, label=i)

    plt.grid()
    plt.legend()

    plt.show()

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
