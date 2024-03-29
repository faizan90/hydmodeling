'''
Created on 23 Jun 2018

@author: Faizan_TR
'''
# import matplotlib as mpl
# mpl.rc('font', size=16)

# matplotlib.use('Cairo')

# Due to shitty tkinter errors.
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .dem_hyd_ansys import (
    get_cumm_cats,
    get_stms,
    TauDEMAnalysis,
    crt_strms_rltn_tree,
    plot_strm_rltn,
    merge_same_id_shp_poly)

from .models import (
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy,

    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy,

    hbv_opt,

    depth_ftn_cy,
    pre_depth_cy,
    post_depth_cy,

    hbv_loop_py,
    hbv_c_loop_py,
    hbv_mult_cat_loop_py,
    tfm_opt_to_hbv_prms_py,

    solve_cats_sys,

    get_ns_var_res_cy,
    get_ln_ns_var_res_cy,

    get_ns_prt_cy,
    get_ln_ns_prt_cy,
    get_pcorr_prt_cy,
    get_kge_prt_cy,
    get_ns_var_res_prt_cy,
    get_ln_ns_var_res_prt_cy,

    get_hargreaves_pet,

    get_sim_probs_in_ref_cy,
    rank_sorted_arr_cy,
    )

from .plotting import (
    plot_cats_hbv_sim,
    plot_cats_qsims,

    plot_cats_kfold_effs,
    plot_cats_prms_transfer_perfs,
    plot_cats_ann_cycs_fdcs_comp,
    plot_cats_vars_errors,
    plot_cats_discharge_errors,

    plot_cats_prm_vecs,
    plot_cats_prm_vecs_evo,
    plot_cats_best_prms_1d,
    plot_cats_best_prms_2d,

    plot_cats_diags,
    )

from .templates import (
    get_data_dict_from_h5_with_time_and_cat,
    get_cell_vars_dict_from_h5)

from .forcings import solve_cats_sys_forcings

from .misc import change_pt_crs, ret_mp_idxs, get_n_cpus
