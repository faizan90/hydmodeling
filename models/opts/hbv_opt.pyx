# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import time
import timeit
import random

from libcpp.map cimport map as cmap
from cython.parallel import prange, threadid
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

from .hbv_obj_ftn cimport obj_ftn
from .de_ftns cimport pre_de, post_de
from .rope_ftns cimport get_new_chull_vecs, pre_rope, post_rope
from .brute_ftns cimport pre_brute, post_brute
from ..ft.dfti cimport cmpt_real_fourtrans_1d
from ..miscs.misc_ftns cimport (
    get_demr,
    get_ln_demr,
    get_mean,
    get_ln_mean,
    get_variance,
    del_idx,
    cmpt_resampled_arr)
from ..miscs.sec_miscs cimport update_obj_doubles
from ..miscs.misc_ftns_partial cimport (
    get_demr_prt,
    get_ln_demr_prt,
    get_mean_prt,
    get_ln_mean_prt,
    get_variance_prt,
    cmpt_resampled_arr_prt)
from ..miscs.dtypes cimport (
    DT_D,
    DT_UL,
    DT_ULL,
    DT_DC,
    ForFourTrans1DReal,
    ForFourTrans1DRealVec,
    fc_i,
    pwp_i,
    n_hbv_cols,
    n_hbv_prms,
    obj_longs_ct,
    obj_doubles_ct,
    err_val,
    off_idx_i,
    curr_us_stm_i,
    n_hm_prms_i,
    route_type_i,
    cat_no_i,
    n_stms_i,
    n_hbv_cols_i,
    use_obs_flow_flag_i,
    opt_flag_i,
    n_cells_i,
    n_hbv_prms_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    ft_beg_idx_i,
    ft_end_idx_i,
    use_res_cat_runoff_flag_i,
    rnof_q_conv_i,
    demr_i,
    ln_demr_i,
    mean_ref_i,
    act_std_dev_i,
    err_val_i,
    min_q_thresh_i,
    ft_demr_1_i,
    ft_demr_2_i)

DT_D_NP = np.float64
DT_UL_NP = np.int32


cdef extern from "complex.h" nogil:
    cdef:
        DT_D cabs(DT_DC)


cdef extern from "../miscs/rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime


warm_up()


cdef extern from "./data_depths.h" nogil:
    cdef:
        void gen_usph_vecs_norm_dist_c(
            unsigned long long *seeds_arr,
            DT_D *rn_ct_arr,
            DT_D *ndim_usph_vecs,
            DT_UL n_vecs,
            DT_UL n_dims,
            DT_UL n_cpus)


cpdef dict hbv_opt(args):

    '''The differential evolution algorithm for distributed HBV with routing
    '''

    cdef:
        Py_ssize_t i, j, k, m

        DT_UL t_i, tid

        #======================================================================
        # Basic optimization related parameters
        DT_UL n_prm_vecs, n_prm_vecs_ol, cont_opt_flag = 1
        DT_UL opt_flag = 1, use_obs_flow_flag, use_step_flag
        DT_UL max_iters, max_cont_iters, off_idx, n_prms, n_hm_prms
        DT_UL iter_curr = 0, last_succ_i = 0, n_succ = 0, cont_iter = 0
        DT_UL resamp_obj_ftns_flag, ft_beg_idx, ft_end_idx
        DT_UL use_res_cat_runoff_flag

        # size in bytes
        DT_UL q_sorig, q_sft, q_sampang

        DT_D obj_ftn_tol, res, prm_pcnt_tol
        DT_D tol_curr = np.inf, tol_pre = np.inf
        DT_D fval_pre_global = np.inf, fval_pre, fval_curr

        list idxs_shuff_list

        ForFourTrans1DRealVec q_ft_tfms

        DT_UL[::1] prm_opt_stop_arr, obj_longs, use_step_arr
        DT_UL[:, ::1] prms_flags, prms_span_idxs, f_var_infos
        DT_UL[:, :, ::1] prms_idxs

        DT_D[::1] pre_obj_vals, curr_obj_vals, best_prm_vec, prms_tmp
        DT_D[::1] obj_ftn_wts, obj_doubles, iobj_vals, gobj_vals
        DT_D[:, ::1] curr_opt_prms, bounds, bds_dfs
        DT_D[:, ::1] prm_vecs, temp_prm_vecs

        # temporarily holds some values in case of residual discharges
        DT_D[:, ::1] obj_res_mult_doubles 

        DT_D[:, :, ::1] iter_prm_vecs
        #======================================================================

        #======================================================================
        # Differential Evolution related parameters
        # Some are in other functions

        DT_UL[::1] idx_rng, r_r
        DT_UL[:, ::1] del_idx_rng, choice_arr

        DT_D[::1] mu_sc_fac_bds, cr_cnst_bds
        DT_D[:, ::1] v_j_g, u_j_gs

        # ROPE related parameters
        DT_UL chull_vecs_ctr = 0, n_temp_rope_prm_vecs, n_acc_prm_vecs
        DT_UL n_uvecs, max_chull_tries, depth_ftn_type, min_pts_in_chull

        DT_UL[::1] depths_arr
        DT_UL[:, ::1] temp_mins, mins

        DT_D[::1] sort_obj_vals, rn_ct_arr
        DT_D[:, ::1] acc_vecs, chull_vecs, rope_bds_dfs, uvecs
        DT_D[:, ::1] prms_mean_thrs_arr
        DT_D[:, ::1] dot_ref, dot_test, dot_test_sort, temp_rope_prm_vecs

        # Brute-Force related parameters
        DT_UL n_discretize
        DT_ULL n_poss_combs, comb_ctr = 0

        DT_UL[::1] use_prm_vec_flags
        DT_UL[::1] last_idxs_vec

        DT_D[:, :, ::1] iter_acc_prm_vecs
        #======================================================================

        #======================================================================
        # HBV related parameters
        DT_D rnof_q_conv

        DT_D[:, :, ::1] hbv_prms
        #======================================================================

        #======================================================================
        # Hydrological state and topology related parameters
        DT_UL curr_us_stm

        DT_UL n_cells, n_recs, cat, stm, n_route_prms, opt_schm
        DT_UL route_type, n_stms, cat_no, stm_idx, cat_idx
        DT_UL n_pts_ft  # always even

        DT_UL[::1] stms_idxs

        DT_ULL[::1] obj_ftn_resamp_tags_arr

        DT_D[::1] inflow_arr, qact_arr, f_vars, area_arr, qact_resamp_arr
        DT_D[:, ::1] qact_qres_mult_arr
        DT_D[:, ::1] inis_arr, temp_arr, prec_arr, petn_arr, route_prms
        DT_D[:, ::1] dem_net_arr, qsim_qres_mult_arr
        DT_D[:, ::1] qsim_mult_arr, inflow_mult_arr
        DT_D[:, ::1] qsim_resamp_mult_arr
        DT_D[:, :, :, ::1] outs_mult_arr

        cmap[long, long] cat_to_idx_map, stm_to_idx_map

        DT_D[::1, :] cats_outflow_arr, stms_inflow_arr, stms_outflow_arr
        DT_D[::1, :, :] stms_inflow_mult_arr, stms_outflow_mult_arr
        DT_D[::1, :, :] cats_outflow_mult_arr
        #======================================================================

        #======================================================================
        # All other variables
        DT_UL n_cpus, n_resamp_tags = 0, a_zero = 0

        DT_D min_q_thresh
#         DT_D mean_ref, ln_mean_ref, demr, ln_demr, act_std_dev
        DT_D ft_demr_1, ft_demr_2

        dict out_dict

        DT_UL[::1] n_calls
        DT_ULL[::1] seeds_arr
        #======================================================================

    bounds = args[0]

    obj_ftn_wts = args[1]

    if obj_ftn_wts[3]:
        ft_beg_idx, ft_end_idx = args[9]

    opt_schm = args[8]

    if opt_schm == 1:
        (mu_sc_fac_bds, 
         cr_cnst_bds, 
         n_prm_vecs,
         max_iters, 
         max_cont_iters, 
         obj_ftn_tol,
         prm_pcnt_tol) = args[2]

    elif opt_schm == 2:
        (n_temp_rope_prm_vecs,
         n_acc_prm_vecs,
         n_uvecs,
         max_chull_tries,
         depth_ftn_type,
         min_pts_in_chull,
         n_prm_vecs,
         max_iters, 
         max_cont_iters, 
         obj_ftn_tol,
         prm_pcnt_tol) = args[2]
    
    elif opt_schm == 3:
        (n_discretize,
         n_prm_vecs,) = args[2]
         
    (inis_arr,
     temp_arr,
     prec_arr,
     petn_arr,
     qact_arr) = args[3]

    (curr_us_stm,
     stms_idxs,
     cat_to_idx_dict,
     stm_to_idx_dict) = args[4]

    (dem_net_arr,
     cats_outflow_arr,
     stms_inflow_arr,
     stms_outflow_arr) = args[5]

    (off_idx,
     rnof_q_conv,
     route_type,
     cat_no,
     n_cpus,
     n_stms,
     n_cells,
     use_obs_flow_flag,
     n_hm_prms,
     use_step_flag,
     use_step_arr,
     min_q_thresh,
     use_res_cat_runoff_flag) = args[6]
    
    (area_arr,
     prms_idxs, 
     prms_flags,
     prms_span_idxs,
     f_vars,
     f_var_infos,
     resamp_obj_ftns_flag,
     obj_ftn_resamp_tags_arr) = args[7]

    n_prms = bounds.shape[0]
    n_recs = cats_outflow_arr.shape[0]
    n_prm_vecs_ol = n_prm_vecs - 1
    idxs_shuff_list = list(range(0,  n_prm_vecs))

    n_calls = np.zeros(n_cpus, dtype=DT_UL_NP)
    seeds_arr = np.zeros(n_cpus, dtype=np.uint64)
    accept_vars = []
    total_vars = []

    # prep the MP RNG
    for i in range(n_cpus):
        seeds_arr[i] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)

    warm_up_mp(&seeds_arr[0], n_cpus)

    pre_obj_vals = np.full(n_prm_vecs, np.inf, dtype=DT_D_NP)
    curr_obj_vals = np.full(n_prm_vecs, np.inf, dtype=DT_D_NP)
    best_prm_vec = np.full(n_prms, np.nan, dtype=DT_D_NP)

    prms_tmp = np.zeros(n_prm_vecs, dtype=DT_D_NP)

    curr_opt_prms = np.zeros((n_cpus, n_prms), dtype=DT_D_NP)
    prm_vecs = np.zeros((n_prm_vecs, n_prms), dtype=DT_D_NP)
    temp_prm_vecs = prm_vecs.copy()

    n_route_prms = n_prms - n_hm_prms

    if n_route_prms <= 0:
        n_route_prms = 1

    route_prms = np.full((n_cpus, n_route_prms), np.nan, dtype=DT_D_NP)

    inflow_arr = np.zeros(n_recs, dtype=DT_D_NP)

    qsim_mult_arr = np.zeros((n_cpus, n_recs), dtype=DT_D_NP)
    inflow_mult_arr = qsim_mult_arr.copy()

    cats_outflow_mult_arr = np.zeros(
        (cats_outflow_arr.shape[0], cats_outflow_arr.shape[1], n_cpus), 
        dtype=DT_D_NP, order='f')

    stms_inflow_mult_arr = np.zeros(
        (stms_inflow_arr.shape[0], stms_inflow_arr.shape[1], n_cpus),
        dtype=DT_D_NP, order='f')

    stms_outflow_mult_arr = np.zeros(
        (stms_inflow_arr.shape[0], stms_inflow_arr.shape[1], n_cpus),
        dtype=DT_D_NP, order='f')

    outs_mult_arr = np.zeros((n_cpus, n_cells, 1, n_hbv_cols), dtype=DT_D_NP)

    for cat in cat_to_idx_dict:
        cat_to_idx_map[cat] = cat_to_idx_dict[cat]

    for stm in stm_to_idx_dict:
        stm_to_idx_map[stm] = stm_to_idx_dict[stm]

    n_resamp_tags = obj_ftn_resamp_tags_arr.shape[0]

    qact_resamp_arr = np.zeros(n_resamp_tags - 1, dtype=DT_D_NP)

    qsim_resamp_mult_arr = np.zeros(
        (n_cpus, n_resamp_tags - 1), dtype=DT_D_NP)

    qact_qres_mult_arr = np.empty((n_cpus, n_recs), dtype=DT_D_NP)
    qsim_qres_mult_arr = np.empty((n_cpus, n_recs), dtype=DT_D_NP)

    bds_dfs = np.zeros((n_prms, 2), dtype=DT_D_NP)

    if opt_schm == 1:
        idx_rng = np.arange(0, n_prm_vecs, 1, dtype=DT_UL_NP)
        r_r = np.zeros(n_cpus, dtype=DT_UL_NP)
        del_idx_rng = np.zeros((n_cpus, n_prm_vecs_ol), dtype=DT_UL_NP)
        choice_arr = np.zeros((n_cpus, 3), dtype=DT_UL_NP)
        v_j_g = np.zeros((n_cpus, n_prms), dtype=DT_D_NP)
        u_j_gs = np.zeros((n_prm_vecs, n_prms), dtype=DT_D_NP)

        # mean, min thresh, max thresh
        prms_mean_thrs_arr = np.zeros((n_prms, 3), dtype=DT_D_NP)

        # for all with value 1, stop optimizing
        prm_opt_stop_arr = np.zeros(n_prms, dtype=DT_UL_NP)

    elif opt_schm == 2:
        # donot use shape of these six in any ftn
        mins = np.full(
            (n_cpus, n_temp_rope_prm_vecs),
            n_temp_rope_prm_vecs, 
            dtype=DT_UL_NP)

        temp_mins = mins.copy()
        depths_arr = np.zeros(n_temp_rope_prm_vecs, dtype=DT_UL_NP)

        if depth_ftn_type == 1:
            dot_ref = np.empty((n_cpus, n_temp_rope_prm_vecs), dtype=DT_D_NP)

        elif depth_ftn_type == 2:
            dot_ref = np.empty((n_uvecs, n_temp_rope_prm_vecs), dtype=DT_D_NP)

        else:
            raise RuntimeError(
                f'depth_ftn_type can be only 1 or 2 not {depth_ftn_type}!')

        dot_test = np.empty((n_cpus, n_temp_rope_prm_vecs), dtype=DT_D_NP)
        dot_test_sort = np.empty((n_cpus, n_temp_rope_prm_vecs), dtype=DT_D_NP)

        temp_rope_prm_vecs = np.empty(
            (n_temp_rope_prm_vecs, n_prms), dtype=DT_D_NP)

        sort_obj_vals = np.empty(n_prm_vecs, dtype=DT_D_NP)

        acc_vecs = np.full(
            (n_acc_prm_vecs, n_prms),
            np.nan,
            dtype=DT_D_NP)

        chull_vecs = acc_vecs.copy()

        rope_bds_dfs = np.empty((n_prms, 2), dtype=DT_D_NP)
        rope_bds_dfs[:, 0] = 0.0
        rope_bds_dfs[:, 1] = 1.0

        uvecs = np.empty((n_uvecs, n_prms), dtype=DT_D_NP)
        rn_ct_arr = np.zeros(n_cpus, dtype=DT_D_NP)

        gen_usph_vecs_norm_dist_c(
            &seeds_arr[0],
            &rn_ct_arr[0],
            &uvecs[0, 0],
            n_uvecs,
            n_prms,
            n_cpus)

    elif opt_schm == 3:
        comb_ctr = 0
        n_poss_combs = n_discretize ** n_prms

        max_iters = (n_poss_combs // n_prm_vecs) + (
            bool(n_poss_combs % n_prm_vecs))

        use_prm_vec_flags = np.zeros(n_prm_vecs, dtype=DT_UL_NP)

        last_idxs_vec = np.zeros(n_prms, dtype=DT_UL_NP)
        last_idxs_vec[n_prms - 1] = -1  # this lets us start with zero idxs

        iter_acc_prm_vecs = np.full(
            (1, max_iters, n_prms),
            np.nan,
            dtype=DT_D_NP)

        print('n_poss_combs:', n_poss_combs)
        print('max_iters:', max_iters)

    if (opt_schm == 1) or (opt_schm == 2):
        iter_prm_vecs = np.full(
            (max_iters + 1, n_prm_vecs, n_prms), np.nan, dtype=DT_D_NP)

    iobj_vals = np.full((max_iters + 1), np.nan, dtype=DT_D_NP)

    gobj_vals = np.full((max_iters + 1), np.nan, dtype=DT_D_NP)

    hbv_prms = np.zeros((n_cpus, n_cells, n_hbv_prms), dtype=DT_D_NP)

    # size and other constants needed in the obj_ftn
    # so that all are in a single variable
    obj_longs = np.zeros(obj_longs_ct, dtype=DT_UL_NP)
    obj_longs[off_idx_i] = off_idx
    obj_longs[curr_us_stm_i] = curr_us_stm
    obj_longs[n_hm_prms_i] = n_hm_prms
    obj_longs[route_type_i] = route_type
    obj_longs[cat_no_i] = cat_no
    obj_longs[n_stms_i] = n_stms
    obj_longs[n_hbv_cols_i] = n_hbv_cols
    obj_longs[use_obs_flow_flag_i] = use_obs_flow_flag
    obj_longs[opt_flag_i] = opt_flag
    obj_longs[n_cells_i] = n_cells
    obj_longs[n_hbv_prms_i] = n_hbv_prms
    obj_longs[use_step_flag_i] = use_step_flag
    obj_longs[resamp_obj_ftns_flag_i] = resamp_obj_ftns_flag
    obj_longs[a_zero_i] = a_zero
    obj_longs[use_res_cat_runoff_flag_i] = use_res_cat_runoff_flag

    obj_doubles = np.full(obj_doubles_ct, np.nan, dtype=DT_D_NP)
    obj_doubles[rnof_q_conv_i] = rnof_q_conv
    obj_doubles[err_val_i] = err_val
    obj_doubles[min_q_thresh_i] = min_q_thresh

    # this assigns some more values to the obj_doubles
    if obj_longs[use_step_flag_i]:
        cmpt_resampled_arr_prt(
            qact_arr,
            qact_resamp_arr,
            obj_ftn_resamp_tags_arr,
            use_step_arr)

    else:
        cmpt_resampled_arr(
            qact_arr,
            qact_resamp_arr,
            obj_ftn_resamp_tags_arr)

    update_obj_doubles(
        obj_longs,
        use_step_arr,
        obj_ftn_wts,
        obj_doubles,
        qact_arr,
        qact_resamp_arr)

    obj_res_mult_doubles = np.full(
        (n_cpus, obj_doubles_ct), np.nan, dtype=DT_D_NP)

    for i in range(n_cpus):
        obj_res_mult_doubles[i, rnof_q_conv_i] = rnof_q_conv
        obj_res_mult_doubles[i, err_val_i] = err_val
        obj_res_mult_doubles[i, min_q_thresh_i] = min_q_thresh

        if ((not obj_longs[use_res_cat_runoff_flag_i]) or 
            (obj_longs[curr_us_stm_i] == -2)):

            obj_res_mult_doubles[i, demr_i] = obj_doubles[demr_i]
            obj_res_mult_doubles[i, ln_demr_i] = obj_doubles[ln_demr_i]
            obj_res_mult_doubles[i, mean_ref_i] = obj_doubles[mean_ref_i]
            obj_res_mult_doubles[i, act_std_dev_i] = obj_doubles[act_std_dev_i]

    for k in range(n_prms):
        bds_dfs[k, 0] = bounds[k, 0]
        bds_dfs[k, 1] = bounds[k, 1] - bounds[k, 0]

    for i in range(n_cpus):
        for j in range(n_recs):
            for k in range(stms_inflow_arr.shape[1]):
                stms_inflow_mult_arr[j, k, i] = stms_inflow_arr[j, k]
                stms_outflow_mult_arr[j, k, i] = stms_outflow_arr[j, k]

            for k in range(cats_outflow_arr.shape[1]):
                cats_outflow_mult_arr[j, k, i] = cats_outflow_arr[j, k]

    if (opt_schm == 1) or (opt_schm == 2):
        for i in range(n_prm_vecs):
            for k in range(n_prms):
                temp_prm_vecs[i, k] = (<DT_D> i) / n_prm_vecs_ol

        for i in range(n_prms):
            random.shuffle(idxs_shuff_list)
            for j in range(n_prm_vecs):
                prms_tmp[j] = temp_prm_vecs[<DT_UL> idxs_shuff_list[j], i]

            for j in range(n_prm_vecs):
                prm_vecs[j, i] = prms_tmp[j]

        for i in range(n_prm_vecs):
            for j in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):
                if (prm_vecs[i, prms_span_idxs[pwp_i, 0] + j] <
                    prm_vecs[i, prms_span_idxs[fc_i, 0] + j]):

                    continue

                # arbitrary decrease
                prm_vecs[i, prms_span_idxs[pwp_i, 0] + j] = (
                    0.99 * rand_c() * prm_vecs[i, prms_span_idxs[fc_i, 0] + j])

            for j in range(n_hbv_prms):
                for m in range(3, 6):
                    if not prms_flags[j, m]:
                        continue

                    k = prms_idxs[j, m, 0]
                    if prm_vecs[i, k + 1] >= prm_vecs[i, k]:
                        continue 

                    prm_vecs[i, k] = 0.99 * rand_c() * prm_vecs[i, k + 1]

    elif opt_schm == 3:
        pre_brute(
            last_idxs_vec,
            use_prm_vec_flags,
            prms_flags,
            prms_span_idxs,
            prms_idxs,
            prm_vecs,
            n_poss_combs,
            n_discretize,
            n_hbv_prms,
            &comb_ctr)

    if obj_ftn_wts[3]:
        n_pts_ft = n_recs - off_idx
        if n_recs % 2:
            n_pts_ft -= 1

        assert n_pts_ft > 1

        q_sorig = n_pts_ft * sizeof(DT_D)
        q_sft = ((n_pts_ft // 2) + 1) * sizeof(DT_DC)
        q_sampang = (((n_pts_ft // 2) + 1) * sizeof(DT_D))

        for i in range(n_cpus + 1):
            q_ft_tfms.push_back(<ForFourTrans1DReal *> malloc(
                sizeof(ForFourTrans1DReal)))

            q_ft_tfms[i].orig = <DT_D *> malloc(q_sorig)

            q_ft_tfms[i].ft = <DT_DC *> malloc(q_sft)

            q_ft_tfms[i].amps = <DT_D *> malloc(q_sampang)

            q_ft_tfms[i].angs = <DT_D *> malloc(q_sampang)

            q_ft_tfms[i].n_pts = n_pts_ft

        for i in range(n_pts_ft):
            q_ft_tfms[0].orig[i] = qact_arr[off_idx + i]
            q_ft_tfms[1].orig[i] = obj_doubles[mean_ref_i]

        cmpt_real_fourtrans_1d(q_ft_tfms[0])
        cmpt_real_fourtrans_1d(q_ft_tfms[1])

        ft_demr_1 = 0.0
        for i in range(ft_beg_idx, ft_end_idx):
            ft_demr_1 += cabs(q_ft_tfms[0].ft[i] - q_ft_tfms[1].ft[i])**2

        ft_demr_2 = 0.0
        for i in range((n_pts_ft // 2) + 1):
            ft_demr_2 += cabs(q_ft_tfms[0].ft[i] - q_ft_tfms[1].ft[i])

        obj_longs[ft_beg_idx_i] = ft_beg_idx
        obj_longs[ft_end_idx_i] = ft_end_idx

        obj_doubles[ft_demr_1_i] = ft_demr_1
        obj_doubles[ft_demr_2_i] = ft_demr_2

    # for the selected parameters, get obj vals
    for i in prange(
        n_prm_vecs,
        schedule='dynamic', 
        nogil=True,
        num_threads=n_cpus):
#         num_threads=1):

        tid = threadid()

        for k in range(n_prms):
            curr_opt_prms[tid, k] = prm_vecs[i, k]

#         print(['%0.5f'  % _ for _ in curr_opt_prms[tid, :]])
        res = obj_ftn(
            &tid,
            n_calls,
            stms_idxs,
            obj_longs,
            use_step_arr,
            obj_ftn_resamp_tags_arr,
            prms_flags,
            f_var_infos,
            prms_idxs,
            obj_ftn_wts,
            curr_opt_prms[tid],
            qact_arr,
            qact_resamp_arr,
            qsim_resamp_mult_arr[tid],
            area_arr,
            qsim_mult_arr[tid],
            inflow_mult_arr[tid],
            f_vars,
            route_prms[tid],
            qact_qres_mult_arr[tid],
            qsim_qres_mult_arr[tid],
            obj_res_mult_doubles[tid],
            inis_arr,
            temp_arr,
            prec_arr,
            petn_arr,
            cats_outflow_mult_arr[:, :, tid],
            stms_inflow_mult_arr[:, :, tid],
            stms_outflow_mult_arr[:, :, tid],
            dem_net_arr,
            hbv_prms[tid],
            bds_dfs,
            outs_mult_arr[tid],
            cat_to_idx_map,
            stm_to_idx_map,
            q_ft_tfms)

#         print('%d Ini res:' % i, res)
#         raise Exception(res)
        if res == err_val:
            pre_obj_vals[i] = (2 + rand_c()) * err_val

        else:
            pre_obj_vals[i] = res

#     raise Exception(res)

    for i in range(n_prm_vecs):
#         print('%d Ini res:' % i, pre_obj_vals[i])
        if pre_obj_vals[i] > fval_pre_global:
            continue

        if np.isnan(pre_obj_vals[i]):
            raise RuntimeError('res is Nan!')

        fval_pre_global = pre_obj_vals[i]
        for k in range(n_prms):
            best_prm_vec[k] = prm_vecs[i, k]

    iobj_vals[0] = fval_pre_global
    gobj_vals[0] = fval_pre_global

    print('Initial min. obj. value:', fval_pre_global)
#     raise Exception('Stop!')

    if (opt_schm == 1) or (opt_schm == 2):
        for i in range(n_prm_vecs):
            for j in range(n_prms):
                iter_prm_vecs[0, i, j] = prm_vecs[i, j]

    cont_opt_flag = 1
    while cont_opt_flag:
        if opt_schm == 1:
            pre_de(
                idx_rng,
                r_r,
                prms_flags,
                prms_span_idxs,
                del_idx_rng,
                choice_arr,
                prms_idxs,
                seeds_arr, # a warmed up one
                mu_sc_fac_bds,
                cr_cnst_bds,
                prm_vecs,
                v_j_g,
                u_j_gs,
                n_hbv_prms,
                n_cpus,
                &cont_iter)

        elif opt_schm == 2:
            print(f'\niter_curr: {iter_curr}')
            pre_rope(
                prms_flags,
                prms_span_idxs,
                depths_arr,
                temp_mins,
                mins,
                prms_idxs,
                pre_obj_vals,
                sort_obj_vals,
                prm_vecs,
                uvecs,
                temp_rope_prm_vecs,
                acc_vecs,
                rope_bds_dfs,
                dot_ref,
                dot_test,
                dot_test_sort,
                chull_vecs,
                n_hbv_prms,
                n_cpus,
                &chull_vecs_ctr,
                &cont_iter,
                max_chull_tries,
                &cont_opt_flag,
                depth_ftn_type,
                min_pts_in_chull)

        elif opt_schm == 3:
            pre_brute(
                last_idxs_vec,
                use_prm_vec_flags,
                prms_flags,
                prms_span_idxs,
                prms_idxs,
                prm_vecs,
                n_poss_combs,
                n_discretize,
                n_hbv_prms,
                &comb_ctr)

        if not cont_opt_flag:
            # This is just for ROPE
            break

        if opt_schm == 1:
            for i in range(n_prm_vecs):
                for j in range(n_prms):
                    iter_prm_vecs[iter_curr + 1, i, j] = u_j_gs[i, j]

        elif opt_schm == 2:
            for i in range(n_prm_vecs):
                for j in range(n_prms):
                    iter_prm_vecs[iter_curr + 1, i, j] = prm_vecs[i, j]

        for t_i in prange(
            n_prm_vecs, 
            schedule='dynamic', 
            nogil=True, 
            num_threads=n_cpus):

            tid = threadid()

            if (opt_schm == 3) and (not use_prm_vec_flags[t_i]):

                res = (2 + rand_c_mp(&seeds_arr[tid])) * err_val
                curr_obj_vals[t_i] = res

                continue

            if opt_schm == 1:
                for k in range(n_prms):
                    curr_opt_prms[tid, k] = u_j_gs[t_i, k]

            elif (opt_schm == 2) or (opt_schm == 3):
                for k in range(n_prms):
                    curr_opt_prms[tid, k] = prm_vecs[t_i, k]

            res = obj_ftn(
                &tid,
                n_calls,
                stms_idxs,
                obj_longs,
                use_step_arr,
                obj_ftn_resamp_tags_arr,
                prms_flags,
                f_var_infos,
                prms_idxs,
                obj_ftn_wts,
                curr_opt_prms[tid],
                qact_arr,
                qact_resamp_arr,
                qsim_resamp_mult_arr[tid],
                area_arr,
                qsim_mult_arr[tid],
                inflow_mult_arr[tid],
                f_vars,
                route_prms[tid],
                qact_qres_mult_arr[tid],
                qsim_qres_mult_arr[tid],
                obj_res_mult_doubles[tid],
                inis_arr,
                temp_arr,
                prec_arr,
                petn_arr,
                cats_outflow_mult_arr[:, :, tid],
                stms_inflow_mult_arr[:, :, tid],
                stms_outflow_mult_arr[:, :, tid],
                dem_net_arr,
                hbv_prms[tid],
                bds_dfs,
                outs_mult_arr[tid],
                cat_to_idx_map,
                stm_to_idx_map,
                q_ft_tfms)

            if res == err_val:
                res = (2 + rand_c_mp(&seeds_arr[tid])) * err_val

            if (opt_schm == 1) or (opt_schm == 3):
                curr_obj_vals[t_i] = res

            elif opt_schm == 2:
                pre_obj_vals[t_i] = res

        for i in range(n_prm_vecs):
            if np.isnan(pre_obj_vals[i]):
                raise RuntimeError('res is Nan!')

        if opt_schm == 1:
            post_de(
                prm_opt_stop_arr,
                curr_obj_vals,
                pre_obj_vals,
                best_prm_vec,
                iobj_vals,
                u_j_gs,
                prm_vecs,
                prms_mean_thrs_arr,
                max_iters,
                max_cont_iters,
                &iter_curr,
                &last_succ_i,
                &n_succ,
                &cont_iter,
                &cont_opt_flag,
                obj_ftn_tol,
                &tol_curr,
                &tol_pre,
                &fval_pre_global,
                &prm_pcnt_tol)

        elif opt_schm == 2:
            post_rope(
                pre_obj_vals,
                best_prm_vec,
                iobj_vals,
                prm_vecs,
                max_iters,
                max_cont_iters,
                &iter_curr,
                &last_succ_i,
                &n_succ,
                &cont_iter,
                &cont_opt_flag,
                &fval_pre_global)

        elif opt_schm == 3:
            post_brute(
                use_prm_vec_flags,
                curr_obj_vals,
                pre_obj_vals,
                best_prm_vec,
                iobj_vals,
                prm_vecs,
                iter_acc_prm_vecs,
                n_poss_combs,
                &iter_curr,
                &cont_opt_flag,
                &comb_ctr,
                &fval_pre_global)

        gobj_vals[iter_curr] = fval_pre_global
#         print('iter_curr:', iter_curr, 'comb_ctr:', comb_ctr)

#     if opt_schm == 2:
#         get_new_chull_vecs(
#             depths_arr,
#             temp_mins,
#             mins,
#             pre_obj_vals,
#             sort_obj_vals,
#             acc_vecs,
#             prm_vecs,
#             uvecs,
#             dot_ref,
#             dot_test,
#             dot_test_sort,
#             chull_vecs,
#             n_cpus,
#             &chull_vecs_ctr)

    # it is important to call the obj_ftn to makes changes one last time
    # i.e. fill arrays with the best parameters
    tid = 0
    for k in range(n_prms):
        curr_opt_prms[tid, k] = best_prm_vec[k]

    res = obj_ftn(
        &tid,
        n_calls,
        stms_idxs,
        obj_longs,
        use_step_arr,
        obj_ftn_resamp_tags_arr,
        prms_flags,
        f_var_infos,
        prms_idxs,
        obj_ftn_wts,
        curr_opt_prms[tid],
        qact_arr,
        qact_resamp_arr,
        qsim_resamp_mult_arr[tid],
        area_arr,
        qsim_mult_arr[tid],
        inflow_mult_arr[tid],
        f_vars,
        route_prms[tid],
        qact_qres_mult_arr[tid],
        qsim_qres_mult_arr[tid],
        obj_res_mult_doubles[tid],
        inis_arr,
        temp_arr,
        prec_arr,
        petn_arr,
        cats_outflow_arr,  # no mult arr
        stms_inflow_arr,  # no mult arr
        stms_outflow_arr,  # no mult arr
        dem_net_arr,
        hbv_prms[tid],
        bds_dfs,
        outs_mult_arr[tid],
        cat_to_idx_map,
        stm_to_idx_map,
        q_ft_tfms)

    print('Number of iterations:', iter_curr)
    print('Final min. obj. value:', fval_pre_global)
    print('Successful tries:', n_succ)
    print('Last successful try at:', last_succ_i)
    print('cont_iter:', cont_iter)
    print('n_calls:', np.array(n_calls))
    print('Best parameters:')
    print(['%0.3f' % prm for prm in np.array(best_prm_vec).ravel()])

    if opt_schm == 1:
        print('Final tolerance:', 0.5 * (tol_pre + tol_curr))

    if obj_ftn_wts[3]:
        for i in range(n_cpus + 1):
            free(q_ft_tfms[i].orig)

            free(q_ft_tfms[i].ft)

            free(q_ft_tfms[i].amps)

            free(q_ft_tfms[i].angs)

            free(q_ft_tfms[i])

        q_ft_tfms.clear()

    tid = 0
    out_dict = {
        'hbv_prms': np.asarray(hbv_prms[tid]),
        'route_prms': np.asarray(route_prms[tid]),
        'opt_prms': np.asarray(best_prm_vec),
        'fmin': fval_pre_global,
        'n_gens': iter_curr,
        'n_succ': n_succ,
        'lst_succ_try': last_succ_i,
        'cont_iter': cont_iter,
        'accept_vars': accept_vars,
        'total_vars': total_vars,
        'n_calls': np.asarray(n_calls),
        'qsim_arr': np.asarray(qsim_mult_arr[tid]),
        'pre_obj_vals' : np.asarray(pre_obj_vals),
        'gobj_vals': np.asarray(gobj_vals),
        'iobj_vals': np.asarray(iobj_vals)}

    if opt_schm == 1:
        out_dict['iter_prm_vecs'] = np.asarray(iter_prm_vecs[:iter_curr])
        out_dict['prm_vecs'] = np.asarray(prm_vecs)
        out_dict['fin_tol'] = 0.5 * (tol_pre + tol_curr)
        out_dict['curr_obj_vals'] = np.asarray(curr_obj_vals)

    elif opt_schm == 2:
        out_dict['iter_prm_vecs'] = np.asarray(iter_prm_vecs[:iter_curr])
        out_dict['prm_vecs'] = np.asarray(acc_vecs)
        out_dict['curr_obj_vals'] = np.asarray(
            sort_obj_vals[:acc_vecs.shape[0]])

    elif opt_schm == 3:
        out_dict['iter_prm_vecs'] = np.asarray(iter_acc_prm_vecs)
        out_dict['prm_vecs'] = np.asarray(prm_vecs)
        out_dict['curr_obj_vals'] = np.asarray(curr_obj_vals)

    else:
        raise NotImplementedError

    return out_dict
