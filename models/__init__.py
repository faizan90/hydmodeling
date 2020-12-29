import pyximport
pyximport.install()

from .miscs.misc_ftns import (
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy,

    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy,

    get_ns_var_res_cy,
    get_ln_ns_var_res_cy,

    get_asymms_sample,

    get_mean,
    get_demr,
    get_ln_mean,
    get_ln_demr,

    get_hargreaves_pet,
    )

from .miscs.misc_ftns_partial import (
    get_ns_prt_cy,
    get_ln_ns_prt_cy,
    get_pcorr_prt_cy,
    get_kge_prt_cy,
    get_ns_var_res_prt_cy,
    get_ln_ns_var_res_prt_cy)

from .miscs.fdcs import get_sim_probs_in_ref_cy, rank_sorted_arr_cy

from .opts.hbv_opt import hbv_opt

from .opts.test_depth_ftns import depth_ftn_cy, pre_depth_cy, post_depth_cy

from .hyds.py_ftns import (
    hbv_loop_py,
    hbv_c_loop_py,
    hbv_mult_cat_loop_py,
    tfm_opt_to_hbv_prms_py)

from .solve_cats_sys import solve_cats_sys
