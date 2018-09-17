'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
from depth_funcs import gen_usph_vecs_norm_dist_mp as gen_uvecs

from hydmodeling import depth_ftn_cy, pre_depth_cy, post_depth_cy


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_ref = int(1e4)
    n_test = int(1e4)
    n_dims = 6
    n_uvecs = int(1e4)
    n_cpus = 7

    ref = np.random.random((n_ref, n_dims))
    test = np.random.random((n_test, n_dims))

    uvecs = gen_uvecs(n_uvecs, n_dims, n_cpus)

    dot_ref = np.empty((n_cpus, n_ref), dtype=np.float64)
    dot_test = np.empty((n_cpus, n_test), dtype=np.float64)
    dot_test_sort = np.empty((n_cpus, n_test), dtype=np.float64)

    dot_ref_sort_cy = np.empty((n_uvecs, n_ref), dtype=np.float64)
    dot_ref_sort_c = np.empty((n_uvecs, n_ref), dtype=np.float64)

    dot_test_post_cy = np.empty((n_cpus, n_test), dtype=np.float64)
    dot_test_post_c = np.empty((n_cpus, n_test), dtype=np.float64)

    dot_test_sort_post_cy = np.empty((n_cpus, n_test), dtype=np.float64)
    dot_test_sort_post_c = np.empty((n_cpus, n_test), dtype=np.float64)

    temp_mins_cy = np.full((n_cpus, n_test), n_ref, dtype=np.int32)
    mins_cy = np.full((n_cpus, n_test), n_ref, dtype=np.int32)
    depths_arr_cy = np.full(n_test, n_ref, dtype=np.int32)

    temp_mins_c = temp_mins_cy.copy()
    mins_c = mins_cy.copy()
    depths_arr_c = depths_arr_cy.copy()

    temp_mins_post_cy = np.full((n_cpus, n_test), n_ref, dtype=np.int32)
    mins_post_cy = np.full((n_cpus, n_test), n_ref, dtype=np.int32)
    depths_arr_post_cy = np.full(n_test, n_ref, dtype=np.int32)

    temp_mins_post_c = temp_mins_cy.copy()
    mins_post_c = mins_cy.copy()
    depths_arr_post_c = depths_arr_cy.copy()

    temp_mins_st = temp_mins_cy.copy()
    mins_st = mins_cy.copy()
    depths_arr_st = depths_arr_cy.copy()

    depth_ftn_cy(
        ref,
        test,
        uvecs,
        dot_ref,
        dot_test,
        dot_test_sort,
        temp_mins_cy,
        mins_cy,
        depths_arr_cy,
        n_cpus,
        0)

    depth_ftn_cy(
        ref,
        test,
        uvecs,
        dot_ref,
        dot_test,
        dot_test_sort,
        temp_mins_c,
        mins_c,
        depths_arr_c,
        n_cpus,
        1)

    assert np.all(np.isclose(depths_arr_c, depths_arr_cy))
    print('Pass: depth_ftn mp equality')

    depth_ftn_cy(
        ref,
        test,
        uvecs,
        dot_ref,
        dot_test,
        dot_test_sort,
        temp_mins_st,
        mins_st,
        depths_arr_st,
        1,
        0)

    assert np.all(np.isclose(depths_arr_st, depths_arr_cy))
    print('Pass: depth_ftn sp equality')

    pre_depth_cy(
        ref,
        uvecs,
        dot_ref_sort_cy,
        n_cpus,
        0)

    pre_depth_cy(
        ref,
        uvecs,
        dot_ref_sort_c,
        n_cpus,
        1)

    assert np.all(np.isclose(dot_ref_sort_cy, dot_ref_sort_c))
    print('Pass: pre_depth dot_ref_sort equality')

    post_depth_cy(
        test,
        uvecs,
        dot_ref_sort_cy,
        dot_test_post_cy,
        dot_test_sort_post_cy,
        temp_mins_post_cy,
        mins_post_cy,
        depths_arr_post_cy,
        n_cpus,
        0)

    if n_cpus == 1:
        assert np.all(np.isclose(dot_test, dot_test_post_cy))
        assert np.all(np.isclose(dot_test_sort, dot_test_sort_post_cy))

    post_depth_cy(
        test,
        uvecs,
        dot_ref_sort_c,
        dot_test_post_c,
        dot_test_sort_post_c,
        temp_mins_post_c,
        mins_post_c,
        depths_arr_post_c,
        n_cpus,
        1)

    if n_cpus == 1:
        assert np.all(np.isclose(dot_test, dot_test_post_c))
        assert np.all(np.isclose(dot_test_sort, dot_test_sort_post_c))

        assert np.all(np.isclose(dot_test_post_cy, dot_test_post_c))
        assert np.all(np.isclose(dot_test_sort_post_c, dot_test_sort_post_cy))

        assert np.all(np.isclose(temp_mins_cy, temp_mins_post_c))
        assert np.all(np.isclose(temp_mins_post_cy, temp_mins_cy))
#
        assert np.all(np.isclose(temp_mins_post_cy, temp_mins_post_c))
#
        assert np.all(np.isclose(mins_post_cy, mins_post_c))

    assert np.all(np.isclose(depths_arr_st, depths_arr_post_cy))
    assert np.all(np.isclose(depths_arr_post_c, depths_arr_st))

    assert np.all(np.isclose(depths_arr_post_c, depths_arr_post_cy))

    print('Pass: post_depth equality')
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
