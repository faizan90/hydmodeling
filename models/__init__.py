import pyximport
pyximport.install()

from .misc_ftns import (
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy,

    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy,

    get_ns_var_res_cy,
    get_ln_ns_var_res_cy)

from .misc_ftns_partial import (
    get_ns_prt_cy,
    get_ln_ns_prt_cy,
    get_pcorr_prt_cy,
    get_kge_prt_cy,
    get_ns_var_res_prt_cy,
    get_ln_ns_var_res_prt_cy)

from .hbv_opt import hbv_opt_de

from .py_ftns import (
    hbv_loop_py,
    hbv_c_loop_py,
    hbv_mult_cat_loop_py)

from .solve_cats_sys import solve_cats_sys
